package main

// The server's records, held in memory and written through to storage.
//
// Reads never leave the process. A change is made to the in-memory state inside update(),
// which then writes every object the change touched in one atomic commit. If the commit
// fails, or the change gave up half-way, the in-memory state is thrown away and read again
// from storage, so memory can never get ahead of what is durable.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func now() int64 { return time.Now().Unix() }

func mustJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// --- what is stored ---------------------------------------------------------------------------

type Account struct {
	ID              int64           `json:"id"`
	Handle          string          `json:"handle"`
	Pubkey          string          `json:"pubkey"`
	Github          string          `json:"github,omitempty"`
	GithubID        int64           `json:"github_id,omitempty"`
	GithubCreatedAt int64           `json:"github_created_at,omitempty"`
	CreatedAt       int64           `json:"created_at"`
	Agreed          int             `json:"agreed,omitempty"`  // grades that matched established accounts' grades of the same script, ever
	Streak          int             `json:"streak,omitempty"`  // and in a row
	Strikes         int             `json:"strikes,omitempty"` // grades that did not, ever
	LastStrikeAt    int64           `json:"last_strike_at,omitempty"`
	Runs            int             `json:"runs,omitempty"` // editors run for the leaderboard, ever
	Hardware        json.RawMessage `json:"hardware,omitempty"`
	GPUSeconds      float64         `json:"gpu_seconds,omitempty"`
	LastSeen        int64           `json:"last_seen,omitempty"`
	// RunnerSeen is when it last asked for work as a machine that may run editors; zero if
	// the last time it asked, it was not one. Backend is the kind of GPU it had then.
	RunnerSeen int64  `json:"runner_seen,omitempty"`
	Backend    string `json:"backend,omitempty"`
	// Anchor marks an organisers' account. It comes from configuration, never from storage:
	// the organisers' accounts are named by the people who run the server, and a database
	// row cannot make one. Such an account starts out established and administers the
	// season; no score depends on it.
	Anchor bool `json:"-"`
	// Worker marks an organisers' account that is a machine they rent (Config.organiser).
	// Anchor is set too: to the judge it is one of the organisers' machines. It administers
	// nothing.
	Worker bool `json:"-"`
}

// administers reports whether an account may act for the organisers: draw, publish, look
// up keys, and keep or fetch the held-out exam. Their rented machines may not.
func (c *Account) administers() bool { return c.Anchor && !c.Worker }

type Editor struct {
	ID          string       `json:"id"` // sha256 of the bundle
	AccountID   int64        `json:"account"`
	Name        string       `json:"name"`
	Version     int          `json:"version"`
	Description string       `json:"description,omitempty"`
	Technique   string       `json:"technique,omitempty"`
	Parent      string       `json:"parent,omitempty"`
	Entry       string       `json:"entry"`
	Files       []BundleFile `json:"files"`
	Size        int64        `json:"size"`
	CreatedAt   int64        `json:"created_at"`
	Status      string       `json:"status"`              // queued | running | retired | withdrawn
	GithubID    int64        `json:"github_id,omitempty"` // whose it was when submitted
	Public      bool         `json:"public,omitempty"`
	Assets      []Asset      `json:"assets,omitempty"`      // files it needs by reference: model weights
	AssetBytes  int64        `json:"asset_bytes,omitempty"` // and what they come to
}

type Round struct {
	Idx          int                `json:"idx"`
	Status       string             `json:"status"` // scheduled | open | settled
	Scale        string             `json:"scale"`
	DrawAt       int64              `json:"draw_at"`
	DrandRound   int64              `json:"drand_round"`
	Randomness   string             `json:"randomness,omitempty"`
	Seed         string             `json:"seed,omitempty"`
	Crawl        *Crawl             `json:"crawl,omitempty"`
	DrawnAt      int64              `json:"drawn_at,omitempty"`
	SettledAt    int64              `json:"settled_at,omitempty"`
	CorpusSHA    string             `json:"corpus_sha256,omitempty"`
	ExamSHA      string             `json:"exam_sha256,omitempty"`
	BaselineLoss *float64           `json:"baseline_loss,omitempty"`
	Baseline     map[string]float64 `json:"baseline,omitempty"`
	BaselineN    int                `json:"baseline_n,omitempty"`
	// Was lists the beacon rounds this round was announced for before its draw moved.
	// No other round is ever announced for one of them: a verifier that saw the first
	// announcement would rightly refuse the second.
	Was []int64 `json:"was,omitempty"`
	// Postponed says the draw was put off because the round before was still being graded.
	// It is then made at the first grid mark after that round settles.
	Postponed bool `json:"postponed,omitempty"`
	// Verifiers whose grades were turned away because their machine built a different
	// round or measured a different unedited loss. Published when the round settles.
	Differs []Differ `json:"differs,omitempty"`
	// ActiveAt is when something last moved the round on: a run, a grade by an established
	// account, a job one of them took, handed back or let run out, an entry given up. How
	// long the round waits for people is counted from here.
	ActiveAt int64 `json:"active_at,omitempty"`
}

// EntryDetail is the public summary of a settled entry: what the grade its score rests on
// reported, and who graded it. Every run and every grade is in the entry's Runs.
type EntryDetail struct {
	Script    json.RawMessage `json:"script,omitempty"`
	Hardware  json.RawMessage `json:"hardware,omitempty"`
	Train     *TrainInfo      `json:"train,omitempty"`
	Verifiers []string        `json:"verifiers"`          // whose grades the score is made of, then everyone else whose grade agreed
	Disputed  int             `json:"disputed,omitempty"` // grades that differed from what the score is made of
	Audited   bool            `json:"audited,omitempty"`  // an established account graded it (always, for an entry with a score)
	// Replications is how entries settled before harness 0.3.0 listed volunteers' grades,
	// beside the grade of the organisers' machine that was then the score.
	Replications []Replication `json:"replications,omitempty"`
	// Two established accounts graded one of its scripts, disagreed, and nobody was there
	// to break the tie: the score is their mean, and one of them is wrong.
	DisputedTogether bool `json:"disputed_together,omitempty"`
}

// Replication is one volunteer's grade of an entry settled before harness 0.3.0.
type Replication struct {
	Handle   string   `json:"handle"`
	Verdict  string   `json:"verdict"`              // agreed | differs | other data | baseline differs
	Valid    bool     `json:"valid"`                // whether this volunteer found the script valid
	Loss     *float64 `json:"loss,omitempty"`       // its exam loss
	Diff     *float64 `json:"difference,omitempty"` // its loss minus the score's
	BaseDiff *float64 `json:"baseline_difference,omitempty"`
	First    bool     `json:"first,omitempty"` // this account had no agreed grade before this round
}

// Run is one execution of an entry's editor on one verifier's machine, and the script it
// wrote. What follows At is filled in when the entry settles, and is what is published.
type Run struct {
	N          int         `json:"n"`                   // 1, 2, 3: the order they arrived in
	By         int64       `json:"by"`                  // the account that ran it
	GithubID   int64       `json:"github_id,omitempty"` // and the GitHub identity it had then
	Claim      int64       `json:"claim"`               // the job it was
	Script     string      `json:"script,omitempty"`    // sha256 of the script; empty if the editor wrote none
	ScriptSize int64       `json:"script_size,omitempty"`
	Edit       *EditReport `json:"edit,omitempty"`
	At         int64       `json:"at"`
	GPU        string      `json:"gpu,omitempty"`
	Backend    string      `json:"backend,omitempty"` // cuda | rocm

	Valid  bool    `json:"valid,omitempty"`  // it wrote a script, and the script is valid
	Gain   float64 `json:"gain,omitempty"`   // its script's gain; 0 for a run that failed
	Reason string  `json:"reason,omitempty"` // why it failed, or why it was dropped
	// Self: the only established grade of its script is by the account that ran it.
	Self bool `json:"self,omitempty"`
	// Dropped: no established account could grade its script, so it counts for nothing.
	Dropped bool    `json:"dropped,omitempty"`
	Grades  []Grade `json:"grades,omitempty"`
	// Owed: its runner has not been credited with it yet. That happens when the entry
	// settles, so that nothing public moves when a run arrives.
	Owed bool `json:"owed,omitempty"`
}

// Grade is one verifier's grade of a script, as published when the round settles.
type Grade struct {
	Handle string `json:"by"`
	// agreed | outlier | disputed | other data | baseline differs | no established grade
	Verdict     string   `json:"verdict"`
	Valid       bool     `json:"valid"`          // whether this grader found the script valid
	Loss        *float64 `json:"loss,omitempty"` // its exam loss
	Gain        *float64 `json:"gain,omitempty"` // its own unedited loss minus that
	First       bool     `json:"first,omitempty"`
	Self        bool     `json:"self,omitempty"`        // it graded a script that it alone wrote
	Counted     bool     `json:"counted,omitempty"`     // the script's gain is the median of these
	Established bool     `json:"established,omitempty"` // by an established account
}

// Differ is a verifier whose grade was turned away because its machine built a different
// round or measured a different unedited loss. Kept so that it is on show with the rest.
type Differ struct {
	Handle string  `json:"handle"`
	What   string  `json:"what"` // corpus | baseline
	Diff   float64 `json:"difference,omitempty"`
	Est    bool    `json:"established,omitempty"` // the account's work would have counted towards scores
}

type Entry struct {
	RoundIdx int                `json:"round"`
	EditorID string             `json:"editor"`
	Status   string             `json:"status"` // open | settled
	Valid    bool               `json:"valid,omitempty"`
	Loss     *float64           `json:"loss,omitempty"`
	Gain     float64            `json:"gain,omitempty"`
	BySource map[string]float64 `json:"by_source,omitempty"`
	Detail   *EntryDetail       `json:"detail,omitempty"`
	NResults int                `json:"n_results,omitempty"`
	Reason   string             `json:"reason,omitempty"`
	// Void: the verifiers could not run the editor at all, or could not grade what it wrote
	// (their machines' trouble, not the editor's). The round does not count for it, for or
	// against. Unattended: and that was because nobody was there to do it, which is not
	// held against the entry in any way.
	Void       bool  `json:"void,omitempty"`
	Unattended bool  `json:"unattended,omitempty"`
	SettledAt  int64 `json:"settled_at,omitempty"`
	// The runs of the editor in this round. Sealed until the round settles.
	Runs []*Run `json:"runs,omitempty"`
	// Of those, the one the score is nearest to, copied here when the entry settles. (An
	// entry settled before harness 0.3.0 has only these: its one run, by the organisers'
	// machine.)
	Script     string      `json:"script,omitempty"` // sha256 of the edit script
	ScriptSize int64       `json:"script_size,omitempty"`
	Edit       *EditReport `json:"edit,omitempty"`
	EditedAt   int64       `json:"edited_at,omitempty"`
	EditedBy   int64       `json:"edited_by,omitempty"`
	NRuns      int         `json:"n_runs,omitempty"`      // runs that counted
	Confirmed  bool        `json:"confirmed,omitempty"`   // two people's runs agree, and none rests on its own runner's grade
	RunsDiffer bool        `json:"runs_differ,omitempty"` // two or more runs, and no two of them agree
	OneGrader  bool        `json:"one_grader,omitempty"`  // its runs agree, and one person's grades decide them all
	OneRunner  bool        `json:"one_runner,omitempty"`  // its runs agree, and they are all one person's machines' (the organisers')
}

type Claim struct {
	ID        int64  `json:"id"`
	RoundIdx  int    `json:"round"`
	EditorID  string `json:"editor"`
	AccountID int64  `json:"account"`
	Kind      string `json:"kind"`             // edit | grade
	Script    string `json:"script,omitempty"` // a grade job: which of the entry's scripts
	ClaimedAt int64  `json:"claimed_at"`
	ExpiresAt int64  `json:"expires_at"`
	// claimed | done | released | refused | expired (the entry settled first) |
	// lapsed (its time ran out with no report and no release)
	Status string `json:"status"`
}

func (c *Claim) live(t int64) bool { return c.Status == "claimed" && c.ExpiresAt > t }

type ResultSummary struct {
	Script   json.RawMessage `json:"script,omitempty"`
	Hardware json.RawMessage `json:"hardware,omitempty"`
	Train    *TrainInfo      `json:"train,omitempty"`
}

type Result struct {
	ID            int64              `json:"id"`
	ClaimID       int64              `json:"claim"`
	RoundIdx      int                `json:"round"`
	EditorID      string             `json:"editor"`
	AccountID     int64              `json:"account"`
	Script        string             `json:"script,omitempty"` // which of the entry's scripts it graded
	ReceivedAt    int64              `json:"received_at"`
	Valid         bool               `json:"valid,omitempty"`
	Loss          float64            `json:"loss,omitempty"`
	BySource      map[string]float64 `json:"by_source,omitempty"`
	InvalidReason string             `json:"invalid_reason,omitempty"`
	CorpusSHA     string             `json:"corpus_sha256"`
	ExamSHA       string             `json:"exam_sha256"`
	Seconds       float64            `json:"seconds,omitempty"`
	// pending | counted (an established grade that stood alone as its script's gain) |
	// agreed | same (agreed, on a script that earns no standing) | outlier | disputed (one
	// of two established grades that disagree) | void
	Verdict string `json:"verdict"`
	// Summary is kept only until the entry settles; the entry's detail keeps the one that
	// came to define its score.
	Summary *ResultSummary `json:"summary,omitempty"`
}

type Baseline struct {
	RoundIdx   int                `json:"round"`
	AccountID  int64              `json:"account"`
	Loss       float64            `json:"loss"`
	BySource   map[string]float64 `json:"by_source"`
	CorpusSHA  string             `json:"corpus_sha256"`
	ExamSHA    string             `json:"exam_sha256"`
	ReceivedAt int64              `json:"received_at"`
	// Established says the account was established when it measured this, so the measurement
	// is one of those the round's unedited loss is the median of.
	Established bool `json:"established,omitempty"`
}

type Record struct {
	N          int      `json:"n"`
	EditorID   string   `json:"editor"`
	RoundIdx   int      `json:"round"`
	At         int64    `json:"at"`
	Rating     float64  `json:"rating"`
	Multiplier *float64 `json:"multiplier,omitempty"`
	Capped     bool     `json:"capped,omitempty"`
	Rounds     int      `json:"rounds"`    // rounds shared with the previous holder
	Advantage  float64  `json:"advantage"` // mean per-round advantage over it, nats
	P          float64  `json:"p_value"`
	Beat       string   `json:"beat,omitempty"` // the previous holder; empty for the first record
}

// Look is one of a challenger's few chances at the record. See checkRecord.
type Look struct {
	Challenger string  `json:"challenger"`
	Holder     string  `json:"holder"`
	Look       int     `json:"look"`
	Shared     int     `json:"shared"`
	Advantage  float64 `json:"advantage"`
	P          float64 `json:"p_value"`
	At         int64   `json:"at"`
}

type Event struct {
	ID   int64          `json:"id"`
	At   int64          `json:"at"`
	Kind string         `json:"kind"`
	Data map[string]any `json:"data"`
}

type Meta struct {
	NextAccount int64 `json:"next_account"`
	NextClaim   int64 `json:"next_claim"`
	NextResult  int64 `json:"next_result"`
	NextEvent   int64 `json:"next_event"`
}

type SavedCrawl struct {
	Crawl
	CheckedAt int64 `json:"checked_at"`
}

// --- keys -------------------------------------------------------------------------------------

func accountKey(id int64) string             { return fmt.Sprintf("account/%012d", id) }
func editorKey(id string) string             { return "editor/" + id }
func roundKey(idx int) string                { return fmt.Sprintf("round/%08d", idx) }
func entryKey(idx int, editor string) string { return fmt.Sprintf("entry/%08d/%s", idx, editor) }
func claimKey(id int64) string               { return fmt.Sprintf("claim/%012d", id) }
func resultKey(id int64) string              { return fmt.Sprintf("result/%012d", id) }
func baselineKey(idx int, acct int64) string { return fmt.Sprintf("baseline/%08d/%012d", idx, acct) }
func recordKey(n int) string                 { return fmt.Sprintf("record/%06d", n) }
func eventKey(id int64) string               { return fmt.Sprintf("event/%012d", id) }
func lookKey(challenger, holder string, look int) string {
	if holder == "" {
		holder = "-"
	}
	return fmt.Sprintf("look/%s/%s/%03d", challenger, holder, look)
}

const (
	metaKey     = "meta/counters"
	crawlKey    = "meta/crawl"
	eventsKept  = 400
	storeBudget = 90 * time.Second
)

var loadScans = []KVScan{
	{Prefix: "account/"}, {Prefix: "editor/"}, {Prefix: "round/"}, {Prefix: "entry/"}, {Prefix: "claim/"},
	{Prefix: "result/"}, {Prefix: "baseline/"}, {Prefix: "record/"}, {Prefix: "look/"}, {Prefix: "meta/"}, {Prefix: "shadow/"},
	{Prefix: "event/", Reverse: true, Limit: eventsKept},
}

// --- the state --------------------------------------------------------------------------------

type State struct {
	Accounts  map[int64]*Account
	Editors   map[string]*Editor
	Rounds    map[int]*Round
	Entries   map[string]*Entry
	Claims    map[int64]*Claim
	Results   map[int64]*Result
	Baselines map[string]*Baseline
	Records   []*Record // oldest first
	Looks     map[string]*Look
	Shadows   map[int]*Shadow // the next exam's verdict on settled rounds, by round
	Events    []*Event        // oldest first; only the most recent are kept in memory
	Meta      Meta
	Crawl     *SavedCrawl

	byPubkey  map[string]*Account
	byHandle  map[string]*Account
	inRound   map[int][]*Entry
	ofEditor  map[string][]*Entry // ordered by round
	resultsOf map[string][]*Result
	ownedBy   map[int64][]*Editor
}

func newState() *State {
	return &State{
		Accounts: map[int64]*Account{}, Editors: map[string]*Editor{}, Rounds: map[int]*Round{}, Entries: map[string]*Entry{},
		Claims: map[int64]*Claim{}, Results: map[int64]*Result{}, Baselines: map[string]*Baseline{}, Looks: map[string]*Look{},
		Shadows:  map[int]*Shadow{},
		Meta:     Meta{NextAccount: 1, NextClaim: 1, NextResult: 1, NextEvent: 1},
		byPubkey: map[string]*Account{}, byHandle: map[string]*Account{}, inRound: map[int][]*Entry{}, ofEditor: map[string][]*Entry{},
		resultsOf: map[string][]*Result{}, ownedBy: map[int64][]*Editor{},
	}
}

func (s *State) addAccount(a *Account) {
	s.Accounts[a.ID], s.byPubkey[a.Pubkey], s.byHandle[a.Handle] = a, a, a
}

func (s *State) addEditor(e *Editor) {
	s.Editors[e.ID] = e
	s.ownedBy[e.AccountID] = append(s.ownedBy[e.AccountID], e)
}

func (s *State) addEntry(e *Entry) {
	s.Entries[entryKey(e.RoundIdx, e.EditorID)] = e
	s.inRound[e.RoundIdx] = append(s.inRound[e.RoundIdx], e)
	s.ofEditor[e.EditorID] = append(s.ofEditor[e.EditorID], e)
}

func (s *State) addResult(r *Result) {
	s.Results[r.ID] = r
	k := entryKey(r.RoundIdx, r.EditorID)
	s.resultsOf[k] = append(s.resultsOf[k], r)
}

func (s *State) entry(idx int, editor string) *Entry { return s.Entries[entryKey(idx, editor)] }

func (s *State) results(idx int, editor string) []*Result { return s.resultsOf[entryKey(idx, editor)] }

// owner is the account an editor belongs to. Every editor has one.
func (s *State) owner(e *Editor) *Account {
	if a := s.Accounts[e.AccountID]; a != nil {
		return a
	}
	return &Account{Handle: "unknown"}
}

func (s *State) roundWithStatus(status string) *Round {
	var best *Round
	for _, r := range s.Rounds {
		if r.Status == status && (best == nil || r.Idx < best.Idx) {
			best = r
		}
	}
	return best
}

// roundsDesc lists rounds, newest first.
func (s *State) roundsDesc(limit int) []*Round {
	out := make([]*Round, 0, len(s.Rounds))
	for _, r := range s.Rounds {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Idx > out[j].Idx })
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

func (s *State) maxRound() int {
	m := 0
	for idx := range s.Rounds {
		if idx > m {
			m = idx
		}
	}
	return m
}

func (s *State) holder() *Record {
	if len(s.Records) == 0 {
		return nil
	}
	return s.Records[len(s.Records)-1]
}

// decode puts one stored item into the state.
func (s *State) decode(it KVItem, organiser func(key string) (anchor, worker bool)) error {
	into := func(v any) error {
		if err := json.Unmarshal(it.Value, v); err != nil {
			return fmt.Errorf("%s: %w", it.Key, err)
		}
		return nil
	}
	switch prefix, _, _ := strings.Cut(it.Key, "/"); prefix {
	case "account":
		a := &Account{}
		if err := into(a); err != nil {
			return err
		}
		a.Anchor, a.Worker = organiser(a.Pubkey)
		s.addAccount(a)
	case "editor":
		e := &Editor{}
		if err := into(e); err != nil {
			return err
		}
		s.addEditor(e)
	case "round":
		r := &Round{}
		if err := into(r); err != nil {
			return err
		}
		s.Rounds[r.Idx] = r
	case "entry":
		e := &Entry{}
		if err := into(e); err != nil {
			return err
		}
		s.addEntry(e)
	case "claim":
		c := &Claim{}
		if err := into(c); err != nil {
			return err
		}
		s.Claims[c.ID] = c
	case "result":
		r := &Result{}
		if err := into(r); err != nil {
			return err
		}
		s.addResult(r)
	case "baseline":
		b := &Baseline{}
		if err := into(b); err != nil {
			return err
		}
		s.Baselines[baselineKey(b.RoundIdx, b.AccountID)] = b
	case "record":
		r := &Record{}
		if err := into(r); err != nil {
			return err
		}
		s.Records = append(s.Records, r)
	case "look":
		l := &Look{}
		if err := into(l); err != nil {
			return err
		}
		s.Looks[it.Key] = l
	case "shadow":
		sh := &Shadow{}
		if err := into(sh); err != nil {
			return err
		}
		s.Shadows[sh.RoundIdx] = sh
	case "event":
		e := &Event{}
		if err := into(e); err != nil {
			return err
		}
		s.Events = append(s.Events, e)
	case "meta":
		switch it.Key {
		case metaKey:
			return into(&s.Meta)
		case crawlKey:
			s.Crawl = &SavedCrawl{}
			return into(s.Crawl)
		}
	}
	return nil
}

func loadState(ctx context.Context, kv KV, organiser func(key string) (anchor, worker bool)) (*State, int64, error) {
	pages, rev, err := kv.Load(ctx, loadScans)
	if err != nil {
		return nil, 0, err
	}
	s := newState()
	for _, items := range pages {
		for _, it := range items {
			if err := s.decode(it, organiser); err != nil {
				return nil, 0, err
			}
		}
	}
	sort.Slice(s.Records, func(i, j int) bool { return s.Records[i].N < s.Records[j].N })
	sort.Slice(s.Events, func(i, j int) bool { return s.Events[i].ID < s.Events[j].ID })
	for _, es := range s.ofEditor {
		sort.Slice(es, func(i, j int) bool { return es[i].RoundIdx < es[j].RoundIdx })
	}
	for _, rs := range s.resultsOf {
		sort.Slice(rs, func(i, j int) bool { return rs[i].ID < rs[j].ID })
	}
	for _, eds := range s.ownedBy {
		sort.Slice(eds, func(i, j int) bool { return eds[i].CreatedAt < eds[j].CreatedAt })
	}
	return s, rev, nil
}

// --- the store --------------------------------------------------------------------------------

type Store struct {
	mu    sync.RWMutex
	kv    KV
	blobs *blobCache
	// organiser says which keys are the organisers' (Config.organiser)
	organiser func(key string) (anchor, worker bool)
	s         *State
	rev       int64
	version   uint64         // bumps whenever the state changes; caches key on it
	lazy      map[string]any // changes not worth a commit of their own (a verifier's last-seen time)
	stale     bool           // memory could not be reconciled with storage; fixed before the next change
	down      atomic.Bool    // stale, for whoever must not wait for the lock to learn it
}

func openStore(ctx context.Context, kv KV, blobs Blobs, organiser func(key string) (anchor, worker bool)) (*Store, error) {
	st := &Store{kv: kv, blobs: newBlobCache(blobs, 128<<20), organiser: organiser, lazy: map[string]any{}}
	s, rev, err := loadState(ctx, kv, organiser)
	if err != nil {
		return nil, fmt.Errorf("reading the database: %w", err)
	}
	st.s, st.rev = s, rev
	return st, nil
}

// Tx is one change in progress. Everything it saves is written together or not at all.
type Tx struct {
	s      *State
	writes map[string]any // nil value: delete
	lazy   map[string]any
}

func (tx *Tx) put(key string, v any) { tx.writes[key] = v }

func (tx *Tx) saveAccount(a *Account) {
	if tx.s.Accounts[a.ID] == nil {
		tx.s.addAccount(a)
	}
	tx.put(accountKey(a.ID), a)
}

// touch records a change that can wait for the next commit.
func (tx *Tx) touch(a *Account) { tx.lazy[accountKey(a.ID)] = a }

func (tx *Tx) saveEditor(e *Editor) {
	if tx.s.Editors[e.ID] == nil {
		tx.s.addEditor(e)
	}
	tx.put(editorKey(e.ID), e)
}

func (tx *Tx) saveRound(r *Round) {
	tx.s.Rounds[r.Idx] = r
	tx.put(roundKey(r.Idx), r)
}

func (tx *Tx) saveEntry(e *Entry) {
	if tx.s.entry(e.RoundIdx, e.EditorID) == nil {
		tx.s.addEntry(e)
	}
	tx.put(entryKey(e.RoundIdx, e.EditorID), e)
}

func (tx *Tx) newClaim(c *Claim) {
	c.ID = tx.s.Meta.NextClaim
	tx.s.Meta.NextClaim++
	tx.put(metaKey, &tx.s.Meta)
	tx.saveClaim(c)
}

func (tx *Tx) saveClaim(c *Claim) {
	tx.s.Claims[c.ID] = c
	tx.put(claimKey(c.ID), c)
}

func (tx *Tx) dropClaim(c *Claim) {
	delete(tx.s.Claims, c.ID)
	tx.put(claimKey(c.ID), nil)
}

func (tx *Tx) newResult(r *Result) {
	r.ID = tx.s.Meta.NextResult
	tx.s.Meta.NextResult++
	tx.put(metaKey, &tx.s.Meta)
	tx.s.addResult(r)
	tx.put(resultKey(r.ID), r)
}

func (tx *Tx) saveResult(r *Result) { tx.put(resultKey(r.ID), r) }

func (tx *Tx) saveBaseline(b *Baseline) {
	k := baselineKey(b.RoundIdx, b.AccountID)
	tx.s.Baselines[k] = b
	tx.put(k, b)
}

func (tx *Tx) addRecord(r *Record) {
	tx.s.Records = append(tx.s.Records, r)
	tx.put(recordKey(r.N), r)
}

func (tx *Tx) saveLook(l *Look) {
	k := lookKey(l.Challenger, l.Holder, l.Look)
	tx.s.Looks[k] = l
	tx.put(k, l)
}

func (tx *Tx) saveCrawl(c *SavedCrawl) {
	tx.s.Crawl = c
	tx.put(crawlKey, c)
}

func (tx *Tx) event(kind string, data map[string]any) {
	e := &Event{ID: tx.s.Meta.NextEvent, At: now(), Kind: kind, Data: data}
	tx.s.Meta.NextEvent++
	tx.put(metaKey, &tx.s.Meta)
	tx.s.Events = append(tx.s.Events, e)
	if len(tx.s.Events) > eventsKept {
		tx.s.Events = tx.s.Events[len(tx.s.Events)-eventsKept:]
	}
	tx.put(eventKey(e.ID), e)
}

func encodeWrites(objects map[string]any) ([]KVWrite, error) {
	keys := make([]string, 0, len(objects))
	for k := range objects {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]KVWrite, 0, len(keys))
	for _, k := range keys {
		if objects[k] == nil {
			out = append(out, KVWrite{Key: k, Delete: true})
			continue
		}
		raw, err := json.Marshal(objects[k])
		if err != nil {
			return nil, fmt.Errorf("%s: %w", k, err)
		}
		out = append(out, KVWrite{Key: k, Value: raw})
	}
	return out, nil
}

var errUnavailable = fail(503, "the database is not answering; try again in a moment")

// reload replaces the in-memory state with what storage holds. Caller holds the lock. It
// takes its own time: the caller's may already be spent on the commit that failed, and
// until a reload succeeds the state in memory may hold a change that was never stored.
func (st *Store) reload(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), storeBudget)
	defer cancel()
	s, rev, err := loadState(ctx, st.kv, st.organiser)
	if err != nil {
		st.stale = true
		st.down.Store(true)
		slog.Error("could not re-read the database", "err", err)
		return err
	}
	st.s, st.rev, st.stale = s, rev, false
	st.down.Store(false)
	st.lazy = map[string]any{}
	st.version++
	return nil
}

// view runs fn with the state, which it must not change or keep.
func (st *Store) view(fn func(s *State)) {
	st.mu.RLock()
	defer st.mu.RUnlock()
	fn(st.s)
}

// update runs one change and makes it durable before returning. If fn returns an error
// nothing is kept: fn should check everything it can before it starts changing things,
// because backing out of a half-made change costs a full re-read.
func (st *Store) update(ctx context.Context, fn func(tx *Tx) error) error {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), storeBudget)
	defer cancel()
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.stale && st.reload(ctx) != nil {
		return errUnavailable
	}
	tx := &Tx{s: st.s, writes: map[string]any{}, lazy: map[string]any{}}
	if err := fn(tx); err != nil {
		if len(tx.writes) > 0 || len(tx.lazy) > 0 {
			st.reload(ctx)
		}
		return err
	}
	for k, v := range tx.lazy {
		st.lazy[k] = v
	}
	if len(tx.writes) == 0 {
		return nil
	}
	for k, v := range st.lazy {
		if _, ok := tx.writes[k]; !ok {
			tx.writes[k] = v
		}
	}
	return st.commit(ctx, tx.writes)
}

// commit writes objects. Caller holds the lock.
func (st *Store) commit(ctx context.Context, objects map[string]any) error {
	writes, err := encodeWrites(objects)
	if err == nil {
		var rev int64
		if rev, err = st.kv.Commit(ctx, st.rev, writes); err == nil {
			st.rev, st.lazy = rev, map[string]any{}
			st.version++
			return nil
		}
	}
	// Refused, or the answer never came. Either way storage decides what happened.
	if errors.Is(err, errConflict) {
		slog.Warn("another writer changed the database; reading it again")
	} else {
		slog.Error("commit failed", "err", err)
	}
	st.reload(ctx)
	return errUnavailable
}

// sync re-reads storage if another server has written to it. One server writes in normal
// running, but during a deploy the old one and the new one overlap for a few seconds, and
// this is what stops the old one answering from what it remembers.
func (st *Store) sync(ctx context.Context) {
	rev, err := st.kv.Revision(ctx)
	if err != nil {
		slog.Warn("could not ask the database for its revision", "err", err)
		return
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	// Only ever forwards: a commit of our own may have landed since the question was asked.
	if rev > st.rev || st.stale {
		slog.Info("the database moved on; reading it again", "ours", st.rev, "theirs", rev)
		st.reload(ctx)
	}
}

// flush writes out changes that were waiting for company.
func (st *Store) flush(ctx context.Context) {
	st.mu.Lock()
	defer st.mu.Unlock()
	if len(st.lazy) == 0 || st.stale {
		return
	}
	pending := st.lazy
	st.commit(ctx, pending)
}

// ready is asked on every request, and while the database is down the lock is held for
// most of every retry. The answer must not wait behind it.
func (st *Store) ready() bool { return !st.down.Load() }

// audit checks that memory and storage agree, object for object. Tests call it after every
// step: a change that forgot to save something it touched fails here.
func (st *Store) audit(ctx context.Context) error {
	st.flush(ctx)
	st.mu.RLock()
	defer st.mu.RUnlock()
	mem := map[string]any{metaKey: &st.s.Meta}
	if st.s.Crawl != nil {
		mem[crawlKey] = st.s.Crawl
	}
	for _, a := range st.s.Accounts {
		mem[accountKey(a.ID)] = a
	}
	for _, e := range st.s.Editors {
		mem[editorKey(e.ID)] = e
	}
	for _, r := range st.s.Rounds {
		mem[roundKey(r.Idx)] = r
	}
	for k, e := range st.s.Entries {
		mem[k] = e
	}
	for _, c := range st.s.Claims {
		mem[claimKey(c.ID)] = c
	}
	for _, r := range st.s.Results {
		mem[resultKey(r.ID)] = r
	}
	for k, b := range st.s.Baselines {
		mem[k] = b
	}
	for _, r := range st.s.Records {
		mem[recordKey(r.N)] = r
	}
	for k, l := range st.s.Looks {
		mem[k] = l
	}
	for _, sh := range st.s.Shadows {
		mem[shadowKey(sh.RoundIdx)] = sh
	}
	for _, e := range st.s.Events {
		mem[eventKey(e.ID)] = e
	}
	pages, rev, err := st.kv.Load(ctx, loadScans)
	if err != nil {
		return err
	}
	if rev != st.rev {
		return fmt.Errorf("memory is at revision %d, storage at %d", st.rev, rev)
	}
	seen := 0
	for _, items := range pages {
		for _, it := range items {
			if it.Key == metaKey && st.rev == 0 {
				continue
			}
			obj, ok := mem[it.Key]
			if !ok {
				return fmt.Errorf("%s is stored but not in memory", it.Key)
			}
			if raw, _ := json.Marshal(obj); string(raw) != string(it.Value) {
				return fmt.Errorf("%s differs: memory %s, storage %s", it.Key, raw, it.Value)
			}
			seen++
		}
	}
	if _, stored := mem[metaKey]; stored && st.rev == 0 {
		delete(mem, metaKey)
	}
	if missing := len(mem) - seen; missing > 0 {
		for k := range mem {
			found := false
			for _, items := range pages {
				for _, it := range items {
					found = found || it.Key == k
				}
			}
			if !found && !(k == metaKey) {
				return fmt.Errorf("%s is in memory but was never stored", k)
			}
		}
	}
	return nil
}
