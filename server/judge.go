package main

// Jobs, results and settlement: how a number becomes a score.
//
// An entry in a round goes through two stages.
//
//   - EDIT. An anchor (a machine the organisers run) executes the editor in a sandbox and
//     uploads the edit script it wrote. The script is plain data, stored and published under
//     its sha256. Only anchors are ever given an entrant's code.
//   - GRADE. Any verifier downloads the script, trains the fixed student on what it says
//     and reports the exam loss. Grading runs no entrant code, and every grader of an entry
//     trains on exactly the same tokens, so their results can be compared directly.
//
// The rules for grades, in one place:
//
//   - An anchor grades every entry, and the anchor's grade is the score. Nobody else's
//     grade can move a score, so nobody gains by reporting a false one.
//   - Volunteers grade the same scripts on their own GPUs. Their grades are replications:
//     each is compared with the anchor's when the round's results are published, and the
//     outcome is shown next to the score.
//   - Grades are sealed. Nothing about an entry's grades is visible until its round
//     settles, and no grade is accepted afterwards, so a replication cannot be copied.
//   - A round settles as a whole: when an anchor has graded every entry, and every
//     volunteer who had started has reported or a fixed wait has passed.
//   - Two grades agree if both say the script is invalid, or both are valid with exam
//     losses within the season's tolerance.
//   - A volunteer never chooses its job, never grades its own editor, and is offered an
//     entry once. An account whose grades keep disagreeing stops being given work.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"sort"
	"strings"
	"time"

	"github.com/klauspost/compress/zstd"
)

const (
	maxReplicas        = 3  // volunteers' grades of one entry, reported or under way
	muteAfterStrikes   = 3  // disagreements before an account stops being given work...
	muteUnlessAgreed   = 10 // ...unless it has this many agreements for each of them
	retireAfterInvalid = 2
	maxEditAttempts    = 3
	maxGradeAttempts   = 5          // anchor grade jobs handed out for one entry before it is given up
	muteFor            = 14 * 86400 // how long a mute lasts after the latest disagreement
	lapseRest          = 86400      // no new work for this long after letting a job run out
)

// muted reports whether a volunteer is being given no grading work: three grades that
// differed from the anchor's with nobody else finding the same, and not ten agreements
// for each. It lifts by itself, muteFor after the latest such grade, because the machine
// may be honest and merely different, and the only way to find out is to let it try again.
func muted(c *Account, t int64) bool {
	return !c.Anchor && c.Strikes >= muteAfterStrikes && c.Agreed < muteUnlessAgreed*c.Strikes && t < c.LastStrikeAt+muteFor
}

type apiError struct {
	Status  int
	Message string
}

func (e *apiError) Error() string { return e.Message }

func fail(status int, format string, args ...any) *apiError {
	return &apiError{Status: status, Message: fmt.Sprintf(format, args...)}
}

// Hello is what a verifier says about itself when asking for work.
type Hello struct {
	Harness      string          `json:"harness"`
	Season       string          `json:"season"`
	SeasonDigest string          `json:"season_digest"`
	Hardware     json.RawMessage `json:"hardware"`
	Edits        bool            `json:"edits"` // willing and able to run edit jobs in the sandbox
	Sandbox      string          `json:"sandbox"`
}

// deviceOf reads the kind of device out of a hardware description.
func deviceOf(hardware json.RawMessage) string {
	var h struct {
		Device string `json:"device"`
	}
	json.Unmarshal(hardware, &h)
	return h.Device
}

// ResultPayload is what a grader reports. Mirrors stet.client.slim_result.
type ResultPayload struct {
	Harness       string          `json:"harness"`
	HarnessSHA256 string          `json:"harness_sha256"`
	Season        string          `json:"season"`
	SeasonDigest  string          `json:"season_digest"`
	Round         string          `json:"round"`
	Scale         string          `json:"scale"`
	CorpusSHA256  string          `json:"corpus_sha256"`
	ExamSHA256    string          `json:"exam_sha256"`
	ScriptSHA256  string          `json:"script_sha256"`
	Valid         bool            `json:"valid"`
	InvalidReason string          `json:"invalid_reason"`
	Script        json.RawMessage `json:"script"`
	Hardware      json.RawMessage `json:"hardware"`
	StartedAt     int64           `json:"started_at"`
	FinishedAt    int64           `json:"finished_at"`
	Baseline      *ExamScore      `json:"baseline"`
	Exam          *ExamScore      `json:"exam"`
	Train         *TrainInfo      `json:"train"`
}

type ExamScore struct {
	Loss     float64            `json:"loss"`
	BySource map[string]float64 `json:"by_source"`
}

type TrainInfo struct {
	Seconds         float64     `json:"seconds"`
	TokensPerSecond float64     `json:"tokens_per_second"`
	FinalTrainLoss  float64     `json:"final_train_loss"`
	InitSHA256      string      `json:"init_sha256,omitempty"`
	Precision       string      `json:"precision,omitempty"`
	Curve           [][]float64 `json:"curve,omitempty"`
}

// EditReport is what an anchor says about an edit run. Mirrors stet.runner.EditRun.
type EditReport struct {
	Status    string         `json:"status"`
	Seconds   float64        `json:"seconds"`
	TimeLimit float64        `json:"time_limit"`
	ExitCode  *int           `json:"exit_code,omitempty"`
	Sandbox   string         `json:"sandbox,omitempty"`
	Limits    map[string]any `json:"limits,omitempty"`
	Reason    string         `json:"reason,omitempty"`
	LogTail   string         `json:"log_tail,omitempty"`
}

// ScriptPayload is an anchor's report of an edit job, script included.
type ScriptPayload struct {
	Harness      string     `json:"harness"`
	Season       string     `json:"season"`
	SeasonDigest string     `json:"season_digest"`
	Round        string     `json:"round"`
	Scale        string     `json:"scale"`
	CorpusSHA256 string     `json:"corpus_sha256"`
	Edit         EditReport `json:"edit"`
	ScriptSHA256 string     `json:"script_sha256"`
	ScriptSize   int64      `json:"script_size"`
	ScriptZst    string     `json:"script_zst"` // base64 of the zstd-compressed script
}

func (a *App) tolerance() Tolerance {
	if t, ok := a.season.Tolerance[a.cfg.Scale]; ok {
		return t
	}
	return Tolerance{Grade: 0.1, Baseline: 0.1}
}

// recentAgreed counts each account's grades that agreed in the last two weeks: what the
// verifier board shows, and what puts a volunteer's own entries ahead in the queue.
func recentAgreed(s *State) map[int64]int {
	since := now() - 14*86400
	out := map[int64]int{}
	for _, r := range s.Results {
		if r.Verdict == "agreed" && r.ReceivedAt > since {
			out[r.AccountID]++
		}
	}
	return out
}

// agree reports whether two grades of the same entry are consistent with honest runs.
func (a *App) agree(x, y *Result) bool {
	if x.CorpusSHA != y.CorpusSHA || x.ExamSHA != y.ExamSHA || x.Valid != y.Valid {
		return false
	}
	return !x.Valid || math.Abs(x.Loss-y.Loss) <= a.tolerance().Grade
}

// --- claiming -------------------------------------------------------------------------------

type entryState struct {
	Entry      *Entry
	OwnerID    int64
	Editing    bool // an anchor holds a live edit claim
	EditTries  int  // edit jobs handed out for it so far
	Graded     bool // an anchor's grade is in
	Grading    bool // an anchor holds a live grade claim
	GradeTries int  // grade jobs handed to anchors for it so far
	Replicas   int  // volunteers' grades, reported or under way
	Newcomer   bool // one of those volunteers has no agreed grade to its name yet
	Mine       bool // this verifier has had this entry before, whatever became of it
	Closing    bool // an anchor has graded every entry in the round: it is about to settle
}

// openEntryStates describes every entry that still needs work, as one verifier sees it.
func openEntryStates(s *State, acctID int64) []*entryState {
	t := now()
	index := map[string]*entryState{}
	var states []*entryState
	for _, r := range s.Rounds {
		if r.Status != "open" {
			continue
		}
		for _, e := range s.inRound[r.Idx] {
			if e.Status != "open" {
				continue
			}
			st := &entryState{Entry: e, OwnerID: s.Editors[e.EditorID].AccountID}
			index[entryKey(e.RoundIdx, e.EditorID)] = st
			states = append(states, st)
		}
	}
	sort.Slice(states, func(i, j int) bool {
		x, y := states[i].Entry, states[j].Entry
		return x.RoundIdx < y.RoundIdx || (x.RoundIdx == y.RoundIdx && x.EditorID < y.EditorID)
	})
	replica := func(st *entryState, acct *Account) {
		st.Replicas++
		st.Newcomer = st.Newcomer || acct.Agreed == 0
	}
	for k, st := range index {
		for _, r := range s.resultsOf[k] {
			acct := s.Accounts[r.AccountID]
			if acct == nil {
				continue
			}
			if r.AccountID == acctID {
				st.Mine = true
			}
			if acct.Anchor {
				st.Graded = true
			} else {
				replica(st, acct)
			}
		}
	}
	for _, c := range s.Claims {
		st, acct := index[entryKey(c.RoundIdx, c.EditorID)], s.Accounts[c.AccountID]
		if st == nil || acct == nil {
			continue
		}
		if c.Kind == "edit" {
			st.EditTries++
			st.Editing = st.Editing || c.live(t)
			continue
		}
		if c.AccountID == acctID {
			st.Mine = true // released and lapsed jobs count: an entry is offered once
		}
		if acct.Anchor {
			st.GradeTries++
		}
		switch {
		case !c.live(t):
		case acct.Anchor:
			st.Grading = true
		default:
			replica(st, acct)
		}
	}
	ungraded := map[int]bool{}
	for _, st := range states {
		ungraded[st.Entry.RoundIdx] = ungraded[st.Entry.RoundIdx] || !st.Graded
	}
	for _, st := range states {
		st.Closing = !ungraded[st.Entry.RoundIdx]
	}
	return states
}

// wantsGrade decides whether this verifier should grade this entry now.
//
// An anchor grades every entry, once. A volunteer replicates: any entry that is not its
// own, that it has not been offered before, and that does not already have enough
// volunteers on it. At most one of those may be a newcomer (an account with no agreed
// grade yet), so that fresh accounts cannot crowd established volunteers out of the work.
func wantsGrade(st *entryState, acct, owner *Account) bool {
	if st.Entry.Script == "" {
		return false
	}
	if acct.Anchor { // including the reference editors the organisers enter themselves
		return !st.Graded && !st.Grading && st.GradeTries < maxGradeAttempts
	}
	if st.Mine || st.Closing || st.OwnerID == acct.ID || st.Replicas >= maxReplicas {
		return false
	}
	if owner != nil && owner.GithubID != 0 && owner.GithubID == acct.GithubID {
		return false // the same person under another key
	}
	return acct.Agreed > 0 || !st.Newcomer
}

type ScriptRef struct {
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

type Job struct {
	ID        int64          `json:"id"`
	Kind      string         `json:"kind"` // edit | grade
	Round     map[string]any `json:"round"`
	Scale     string         `json:"scale"`
	Editor    map[string]any `json:"editor"`
	Script    *ScriptRef     `json:"script,omitempty"`
	ExpiresAt int64          `json:"expires_at"`
}

func (a *App) jobFor(s *State, c *Claim) *Job {
	rd, ed := s.Rounds[c.RoundIdx], s.Editors[c.EditorID]
	job := &Job{
		ID: c.ID, Kind: c.Kind, Round: rd.description(a.season), Scale: rd.Scale, ExpiresAt: c.ExpiresAt,
		Editor: map[string]any{"id": ed.ID, "name": ed.Name, "version": ed.Version, "owner": s.owner(ed).Handle, "size": ed.Size},
	}
	if c.Kind == "grade" {
		e := s.entry(c.RoundIdx, c.EditorID)
		job.Script = &ScriptRef{SHA256: e.Script, Size: e.ScriptSize}
	}
	return job
}

func (a *App) checkHarness(harness, season, digest string) error {
	if season != a.season.ID || digest != a.season.Digest || versionLess(harness, a.season.Harness) {
		return fail(426, "this verifier is out of date (season %s, harness %s is current). Upgrade: curl -fsSL %s/install.sh | sh (on an AMD GPU, with STET_TORCH_INDEX set as it was when you installed)", a.season.ID, a.season.Harness, a.cfg.BaseURL)
	}
	return nil
}

// claim hands a verifier its next job, or nil if there is nothing it should do right now.
func (a *App) claim(ctx context.Context, acctID int64, h *Hello) (*Job, error) {
	if err := a.checkHarness(h.Harness, h.Season, h.SeasonDigest); err != nil {
		return nil, err
	}
	if !a.cfg.Dev && !a.season.gradesOn(deviceOf(h.Hardware)) {
		return nil, fail(400, "grading for the leaderboard needs an NVIDIA (CUDA) or AMD (ROCm) GPU, so that every grade uses the same arithmetic")
	}
	var job *Job
	err := a.st.update(ctx, func(tx *Tx) error {
		s := tx.s
		acct := s.Accounts[acctID]
		if acct == nil {
			return fail(401, "no such account")
		}
		if muted(acct, now()) {
			return fail(403, "this account is given no grading work until %s: its grades kept differing from the anchor's, and no other volunteer found what it found. If this machine is honest, the organisers want to hear about it",
				time.Unix(acct.LastStrikeAt+muteFor, 0).UTC().Format("2 January"))
		}
		if !acct.Anchor {
			// Taking a job and neither reporting nor handing it back holds a volunteer's place
			// on that entry for hours. After that, a day's rest.
			for _, c := range s.Claims {
				if c.AccountID == acctID && c.Status == "lapsed" && c.ExpiresAt > now()-lapseRest {
					return fail(403, "a job this verifier took was given up on at %s: no result had come, and it had not been handed back. It is offered work again a day after that. (Ctrl-C hands a job back at once.)",
						time.Unix(c.ExpiresAt, 0).UTC().Format("15:04 UTC"))
				}
			}
		}
		canEdit := h.Edits && acct.Anchor && (h.Sandbox == "bwrap" || a.cfg.Dev)
		if hw := clip(string(h.Hardware), 2000); json.Valid([]byte(hw)) {
			acct.Hardware = json.RawMessage(hw)
		}
		acct.LastSeen = now()
		tx.touch(acct)

		// A verifier that restarts picks its unfinished job back up.
		var mine *Claim
		for _, c := range s.Claims {
			if c.AccountID != acctID || !c.live(now()) || (c.Kind == "edit" && !canEdit) {
				continue
			}
			if e := s.entry(c.RoundIdx, c.EditorID); e == nil || e.Status != "open" {
				continue
			}
			if mine == nil || c.ID < mine.ID {
				mine = c
			}
		}
		if mine != nil {
			job = a.jobFor(s, mine)
			return nil
		}
		var pick *entryState
		pickKind := ""
		best, n := math.MaxInt, 0
		consider := func(st *entryState, kind string, score int) {
			// Lowest score first; ties at random so that nobody can predict or steer what they
			// will be given.
			switch {
			case score < best:
				best, pick, pickKind, n = score, st, kind, 1
			case score == best:
				n++
				if rand.IntN(n) == 0 {
					pick, pickKind = st, kind
				}
			}
		}
		// A machine whose corpus or baseline for a round has twice differed from the anchor's
		// will get the same answer a third time. It is given no more of that round.
		differs := map[int]int{}
		for _, c := range s.Claims {
			if c.AccountID == acctID && c.Status == "refused" {
				differs[c.RoundIdx]++
			}
		}
		for _, st := range openEntryStates(s, acctID) {
			if differs[st.Entry.RoundIdx] >= 2 && !acct.Anchor {
				continue
			}
			if canEdit && st.Entry.EditedAt == 0 && !st.Editing && st.EditTries < maxEditAttempts {
				consider(st, "edit", st.Entry.RoundIdx*1000-500) // edits first: they unblock everyone else
			}
			if wantsGrade(st, acct, s.Accounts[st.OwnerID]) {
				consider(st, "grade", st.Entry.RoundIdx*1000+st.Replicas) // oldest round, least attention
			}
		}
		if pick == nil {
			return nil
		}
		// Far more than any GPU can do. It bounds what one account can add to the records.
		if !acct.Anchor && !a.limits.allow(fmt.Sprintf("claim:%d", acctID), 30, 3600) {
			return fail(429, "this account has been handed a great many jobs in the last hour; try again later")
		}
		c := &Claim{RoundIdx: pick.Entry.RoundIdx, EditorID: pick.Entry.EditorID, AccountID: acctID, Kind: pickKind,
			ClaimedAt: now(), ExpiresAt: now() + int64(a.cfg.Lease.Seconds()), Status: "claimed"}
		tx.newClaim(c)
		job = a.jobFor(s, c)
		return nil
	})
	return job, err
}

func (a *App) release(ctx context.Context, acctID, claimID int64) error {
	return a.st.update(ctx, func(tx *Tx) error {
		if c := tx.s.Claims[claimID]; c != nil && c.AccountID == acctID && c.Status == "claimed" {
			c.Status = "released"
			tx.saveClaim(c)
			a.settleReady(tx) // the round may have been waiting for this job alone
		}
		return nil
	})
}

func article(kind string) string {
	if kind == "edit" {
		return "an edit"
	}
	return "a " + kind
}

// findClaim checks that a job is this account's to report on. A volunteer's job ends with
// its lease and with the round: a grade that arrives after the results are public could
// have been copied from them.
func findClaim(s *State, acct *Account, claimID int64, kind string) (*Claim, error) {
	c := s.Claims[claimID]
	if c == nil || c.AccountID != acct.ID {
		return nil, fail(404, "no such job for this account (it may have been finished long ago)")
	}
	if c.Kind != kind {
		return nil, fail(400, "this is %s job, not %s job", article(c.Kind), article(kind))
	}
	switch {
	case c.Status == "done" || c.Status == "released" || c.Status == "refused":
		return nil, fail(409, "this job was already %s", c.Status)
	case !acct.Anchor && !c.live(now()):
		return nil, fail(409, "this job ended before your result arrived: its time ran out or the round settled. It earns no credit, and it does not count against you")
	}
	return c, nil
}

// --- edit results ---------------------------------------------------------------------------

// unzstd decompresses at most limit bytes, refusing anything larger.
func unzstd(blob []byte, limit int64) ([]byte, error) {
	dec, err := zstd.NewReader(bytes.NewReader(blob), zstd.WithDecoderMaxMemory(uint64(limit)+(64<<20)), zstd.WithDecoderConcurrency(1))
	if err != nil {
		return nil, err
	}
	defer dec.Close()
	raw, err := io.ReadAll(io.LimitReader(dec, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) > limit {
		return nil, errors.New("too large")
	}
	return raw, nil
}

func (a *App) readScript(ctx context.Context, sha string) ([]byte, error) {
	if !sha256Re.MatchString(sha) {
		return nil, errNoBlob
	}
	return a.st.blobs.Read(ctx, scriptKey(sha), a.season.Script.MaxBytes)
}

type Ack struct {
	Status   string         `json:"status"` // settled | pending | edited
	Message  string         `json:"message,omitempty"`
	Verifier map[string]any `json:"verifier,omitempty"`
}

// submitScript records the outcome of an edit job. With a script, the entry moves on to
// grading. Without one, the editor failed and the entry settles at once as invalid.
func (a *App) submitScript(ctx context.Context, acctID, claimID int64, p *ScriptPayload) (*Ack, error) {
	if err := a.checkHarness(p.Harness, p.Season, p.SeasonDigest); err != nil {
		return nil, err
	}
	if !sha256Re.MatchString(p.CorpusSHA256) {
		return nil, fail(400, "the edit does not say which corpus it ran on")
	}
	check := func(s *State) (*Claim, error) {
		acct := s.Accounts[acctID]
		if acct == nil || !acct.Anchor {
			return nil, fail(403, "edit jobs are run by anchors only")
		}
		c, err := findClaim(s, acct, claimID, "edit")
		if err != nil {
			return nil, err
		}
		rd := s.Rounds[c.RoundIdx]
		if p.Round != roundID(a.season.ID, c.RoundIdx) || p.Scale != rd.Scale {
			return nil, fail(400, "this edit is for %s at %s, but the job is %s at %s", p.Round, p.Scale, roundID(a.season.ID, c.RoundIdx), rd.Scale)
		}
		if rd.CorpusSHA != "" && rd.CorpusSHA != p.CorpusSHA256 {
			return nil, fail(409, "your corpus for %s differs from the round's", p.Round)
		}
		if e := s.entry(c.RoundIdx, c.EditorID); e == nil || e.Status != "open" || e.EditedAt != 0 {
			return nil, fail(409, "this entry has already been edited")
		}
		return c, nil
	}
	var pre error
	a.st.view(func(s *State) { _, pre = check(s) })
	if pre != nil {
		return nil, pre
	}
	p.Edit.LogTail = clip(strings.ToValidUTF8(p.Edit.LogTail, "?"), 2000)
	p.Edit.Reason = clip(strings.ToValidUTF8(p.Edit.Reason, "?"), 300)
	seconds := p.Edit.Seconds
	if !finite(seconds) || seconds < 0 {
		seconds = 0
	}
	seconds = math.Min(seconds, 2*float64(a.season.Scales[a.cfg.Scale].EditSeconds)) // an edit has a time limit
	sha := ""
	var size int64
	if p.ScriptSHA256 != "" {
		if !sha256Re.MatchString(p.ScriptSHA256) {
			return nil, fail(400, "malformed script hash")
		}
		blob, err := base64.StdEncoding.DecodeString(p.ScriptZst)
		if err != nil {
			return nil, fail(400, "the script is not valid base64")
		}
		raw, err := unzstd(blob, a.season.Script.MaxBytes)
		if err != nil {
			return nil, fail(400, "the script could not be decompressed, or is over the size limit")
		}
		sum := sha256.Sum256(raw)
		if hex.EncodeToString(sum[:]) != p.ScriptSHA256 {
			return nil, fail(400, "the script does not match its hash")
		}
		sha, size = p.ScriptSHA256, int64(len(raw))
		if err := a.st.blobs.Put(ctx, scriptKey(sha), blob, "application/zstd"); err != nil {
			return nil, err
		}
	}
	ack := &Ack{Status: "edited", Message: "the script is published and waiting for grades"}
	err := a.st.update(ctx, func(tx *Tx) error {
		s := tx.s
		c, err := check(s)
		if err != nil {
			return err
		}
		acct, rd, e := s.Accounts[acctID], s.Rounds[c.RoundIdx], s.entry(c.RoundIdx, c.EditorID)
		edit := p.Edit
		e.Script, e.ScriptSize, e.Edit, e.EditedAt, e.EditedBy = sha, size, &edit, now(), acctID
		tx.saveEntry(e)
		if rd.CorpusSHA == "" {
			rd.CorpusSHA = p.CorpusSHA256
			tx.saveRound(rd)
		}
		c.Status = "done"
		tx.saveClaim(c)
		acct.GPUSeconds += seconds
		acct.LastSeen = now()
		tx.saveAccount(acct)
		if sha == "" {
			reason := edit.Reason
			if reason == "" {
				reason = fmt.Sprintf("the editor produced no script (%s)", edit.Status)
			}
			a.settleInvalid(tx, e, reason, &EntryDetail{Verifiers: []string{acct.Handle}})
			a.settleReady(tx)
			ack = &Ack{Status: "settled", Message: "no script: the entry scores as the unedited corpus"}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return ack, nil
}

// expireClaims ends the outstanding jobs on an entry that has settled.
func expireClaims(tx *Tx, e *Entry) {
	for _, c := range tx.s.Claims {
		if c.RoundIdx == e.RoundIdx && c.EditorID == e.EditorID && c.Status == "claimed" {
			c.Status = "expired"
			tx.saveClaim(c)
		}
	}
}

// settleInvalid closes an entry with gain 0.
func (a *App) settleInvalid(tx *Tx, e *Entry, reason string, detail *EntryDetail) {
	e.Status, e.Valid, e.Loss, e.Gain, e.BySource = "settled", false, nil, 0, nil
	e.Detail, e.Reason = detail, clip(reason, 300)
	if e.SettledAt == 0 {
		e.SettledAt = now()
	}
	tx.saveEntry(e)
	expireClaims(tx, e)
	a.afterInvalid(tx, e.EditorID)
}

// failing says an editor keeps failing: two of its last five rounds were invalid. Such an
// editor is not scheduled again, whether it is still qualifying or sits on the ladder; an
// anchor's time is better spent on someone else's entry.
//
// A round in which the anchors could not run an entry is void for it and is not held
// against it. Three of those in a row are, when the anchors ran other entries in each:
// whatever the cause, it is this entry they cannot run. A round in which nothing ran,
// or a void long ago, says nothing about the editor.
func failing(s *State, editorID string) bool {
	es := s.ofEditor[editorID]
	seen, invalid, void := 0, 0, 0
	unbroken := true // void rounds count only in a run, back from the latest
	for i := len(es) - 1; i >= 0; i-- {
		switch e := es[i]; {
		case e.Void:
			if unbroken && othersRan(s, e) {
				void++
			}
		case e.Status == "settled":
			unbroken = false
			if seen < 5 {
				seen++
				if !e.Valid {
					invalid++
				}
			}
		}
	}
	return invalid >= retireAfterInvalid || void >= maxEditAttempts
}

// othersRan reports whether the anchors ran some other entry of the same round. A round in
// which nothing ran says the anchor was in trouble, not the entry.
func othersRan(s *State, e *Entry) bool {
	for _, o := range s.inRound[e.RoundIdx] {
		if o != e && o.Status == "settled" && !o.Void {
			return true
		}
	}
	return false
}

// roundsRun is how many rounds count for an editor: all it was drawn into, except those
// the anchors could not run.
func roundsRun(s *State, editorID string) int {
	n := 0
	for _, e := range s.ofEditor[editorID] {
		if !e.Void {
			n++
		}
	}
	return n
}

// afterInvalid retires a qualifying editor that keeps failing. (One on the ladder is
// already retired from qualifying; computeBoard takes it off the ladder.)
func (a *App) afterInvalid(tx *Tx, editorID string) {
	if ed := tx.s.Editors[editorID]; failing(tx.s, editorID) && (ed.Status == "queued" || ed.Status == "running") {
		ed.Status = "retired"
		tx.saveEditor(ed)
	}
}

// --- grade results --------------------------------------------------------------------------

func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

func (a *App) checkScore(sc *ExamScore, what string) error {
	if sc == nil || !finite(sc.Loss) || sc.Loss <= 0 || sc.Loss > 30 {
		return fail(400, "the result has no usable %s score", what)
	}
	// Exactly the season's sources and nothing else: what is sent here is stored.
	if len(sc.BySource) != len(a.season.ExamSources) {
		return fail(400, "the %s score names %d sources; the season has %d", what, len(sc.BySource), len(a.season.ExamSources))
	}
	sum := 0.0
	for _, src := range a.season.ExamSources {
		v, ok := sc.BySource[src]
		if !ok || !finite(v) || v <= 0 || v > 30 {
			return fail(400, "the %s score is missing source %q", what, src)
		}
		sum += v
	}
	if math.Abs(sum/float64(len(a.season.ExamSources))-sc.Loss) > 1e-3 {
		return fail(400, "the %s score is not the mean of its sources", what)
	}
	return nil
}

func clipJSON(raw json.RawMessage, limit int) json.RawMessage {
	if len(raw) == 0 || len(raw) > limit || !json.Valid(raw) {
		return nil
	}
	return raw
}

func (a *App) submitResult(ctx context.Context, acctID, claimID int64, p *ResultPayload) (*Ack, error) {
	if err := a.checkHarness(p.Harness, p.Season, p.SeasonDigest); err != nil {
		return nil, err
	}
	if !sha256Re.MatchString(p.CorpusSHA256) || !sha256Re.MatchString(p.ExamSHA256) {
		return nil, fail(400, "the result does not say which corpus and exam it used")
	}
	if !a.cfg.Dev && !a.season.gradesOn(deviceOf(p.Hardware)) {
		return nil, fail(400, "grades for the leaderboard must come from an NVIDIA (CUDA) or AMD (ROCm) GPU")
	}
	if err := a.checkScore(p.Baseline, "baseline"); err != nil {
		return nil, err
	}
	if p.Valid {
		if err := a.checkScore(p.Exam, "exam"); err != nil {
			return nil, err
		}
	}
	p.Train = cleanTrain(p.Train)
	var ack *Ack
	var refused error // the job is over, and that has to be stored, so it is not the update's error
	err := a.st.update(ctx, func(tx *Tx) error {
		s := tx.s
		acct := s.Accounts[acctID]
		if acct == nil {
			return fail(401, "no such account")
		}
		if muted(acct, now()) {
			return fail(403, "this account is given no grading work for now")
		}
		c, err := findClaim(s, acct, claimID, "grade")
		if err != nil {
			return err
		}
		rd, e := s.Rounds[c.RoundIdx], s.entry(c.RoundIdx, c.EditorID)
		if p.Round != roundID(a.season.ID, c.RoundIdx) || p.Scale != rd.Scale {
			return fail(400, "this result is for %s at %s, but the job is %s at %s", p.Round, p.Scale, roundID(a.season.ID, c.RoundIdx), rd.Scale)
		}
		if e == nil || e.Status != "open" {
			return fail(409, "this entry settled before your result arrived. It earns no credit, and it does not count against you")
		}
		if e.Script == "" || p.ScriptSHA256 != e.Script {
			return fail(400, "this result is for a different edit script than the one the job named")
		}
		for _, r := range s.results(c.RoundIdx, c.EditorID) {
			if other := s.Accounts[r.AccountID]; r.AccountID == acctID || (acct.Anchor && other != nil && other.Anchor) {
				return fail(409, "this entry already has that grade")
			}
		}
		refuse := func(format string, args ...any) error {
			c.Status = "refused"
			tx.saveClaim(c)
			refused = fail(409, format, args...)
			return nil
		}
		// Everyone must be looking at the same corpus and exam, and must measure the same
		// unedited baseline on them. The first anchor to report fixes all three for the round.
		// A volunteer who reports before that is checked when the round settles instead.
		if (rd.CorpusSHA != "" && rd.CorpusSHA != p.CorpusSHA256) || (rd.ExamSHA != "" && rd.ExamSHA != p.ExamSHA256) {
			if acct.Anchor {
				tx.event("anchors-differ", map[string]any{"round": p.Round, "anchor": acct.Handle})
			}
			if !acct.Anchor && len(rd.Differs) < 50 {
				rd.Differs = append(rd.Differs, Differ{Handle: acct.Handle, What: "corpus"})
				tx.saveRound(rd)
			}
			return refuse("your corpus or exam for %s differs from the round's. Delete ~/.cache/stet/rounds/%s and the pool cache, then try again", p.Round, p.Round)
		}
		if tol := a.tolerance(); rd.BaselineLoss != nil && math.Abs(*rd.BaselineLoss-p.Baseline.Loss) > tol.Baseline {
			if acct.Anchor {
				tx.event("anchors-differ", map[string]any{"round": p.Round, "anchor": acct.Handle})
			}
			if !acct.Anchor && len(rd.Differs) < 50 {
				rd.Differs = append(rd.Differs, Differ{Handle: acct.Handle, What: "baseline", Diff: places5(p.Baseline.Loss - *rd.BaselineLoss)})
				tx.saveRound(rd)
			}
			return refuse("your unedited baseline for %s is further from the anchor's than the tolerance of %.2f nats allows, so this machine's setup or arithmetic differs. That is published with the round when it settles; nothing counts against you", p.Round, tol.Baseline)
		}

		// Everything is checked. From here on the change is made.
		tx.saveBaseline(&Baseline{RoundIdx: rd.Idx, AccountID: acctID, Loss: p.Baseline.Loss, BySource: p.Baseline.BySource,
			CorpusSHA: p.CorpusSHA256, ExamSHA: p.ExamSHA256, ReceivedAt: now()})
		if acct.Anchor {
			rd.CorpusSHA, rd.ExamSHA = p.CorpusSHA256, p.ExamSHA256
			tx.saveRound(rd)
			a.refreshBaseline(tx, rd)
		}
		// GPU time is what the grader says, but never more than the job was out for.
		seconds := math.Min(float64(p.FinishedAt-p.StartedAt), float64(now()-c.ClaimedAt))
		if !finite(seconds) || seconds < 0 {
			seconds = 0
		}
		res := &Result{ClaimID: claimID, RoundIdx: rd.Idx, EditorID: e.EditorID, AccountID: acctID, ReceivedAt: now(),
			Valid: p.Valid, CorpusSHA: p.CorpusSHA256, ExamSHA: p.ExamSHA256, Seconds: seconds, Verdict: "pending",
			InvalidReason: clip(strings.ToValidUTF8(p.InvalidReason, "?"), 300),
			Summary:       &ResultSummary{Script: clipJSON(p.Script, 16<<10), Hardware: clipJSON(p.Hardware, 2000), Train: p.Train}}
		if p.Valid {
			res.Loss, res.BySource = p.Exam.Loss, p.Exam.BySource
		}
		tx.newResult(res)
		c.Status = "done"
		tx.saveClaim(c)
		acct.LastSeen = now()
		if acct.Anchor {
			acct.GPUSeconds += seconds // a volunteer's time counts once its grade has agreed
		}
		tx.saveAccount(acct)

		a.settleReady(tx)
		ack = &Ack{Status: "pending", Verifier: map[string]any{"validated": acct.Agreed, "streak": acct.Streak, "trusted": acct.Anchor}}
		switch {
		case e.Status == "settled":
			ack.Status = "settled"
		case acct.Anchor:
			ack.Message = "sealed until the round's other entries are graded"
		default:
			ack.Message = "sealed until the round settles, then compared with the anchor's grade"
		}
		return nil
	})
	if err == nil && refused != nil {
		return nil, refused
	}
	return ack, err
}

// cleanTrain keeps the part of a training report that is safe to store and draw.
func cleanTrain(t *TrainInfo) *TrainInfo {
	if t == nil {
		return nil
	}
	c := *t
	c.InitSHA256, c.Precision = clip(c.InitSHA256, 64), clip(c.Precision, 16)
	for _, v := range []*float64{&c.Seconds, &c.TokensPerSecond, &c.FinalTrainLoss} {
		if !finite(*v) || *v < 0 || *v > 1e12 {
			*v = 0
		}
	}
	if len(c.Curve) > 200 {
		c.Curve = nil
	}
	for _, pt := range c.Curve {
		if len(pt) != 2 || !finite(pt[0]) || !finite(pt[1]) || pt[0] < 0 || pt[0] > 1e12 || pt[1] < 0 || pt[1] > 30 {
			c.Curve = nil
			break
		}
	}
	return &c
}

// refreshBaseline sets the round's baseline to what its anchors measured (the median, if
// there are several).
func (a *App) refreshBaseline(tx *Tx, rd *Round) {
	var losses []float64
	per := map[string][]float64{}
	for _, b := range tx.s.Baselines {
		acct := tx.s.Accounts[b.AccountID]
		if b.RoundIdx != rd.Idx || acct == nil || !acct.Anchor || b.CorpusSHA != rd.CorpusSHA || b.ExamSHA != rd.ExamSHA {
			continue
		}
		losses = append(losses, b.Loss)
		for k, v := range b.BySource {
			per[k] = append(per[k], v)
		}
	}
	if len(losses) == 0 {
		return
	}
	by := map[string]float64{}
	for k, v := range per {
		by[k] = median(v)
	}
	m := median(losses)
	rd.BaselineLoss, rd.Baseline = &m, by
	// How many machines measured it, volunteers included.
	rd.BaselineN = 0
	for _, b := range tx.s.Baselines {
		if b.RoundIdx == rd.Idx && b.CorpusSHA == rd.CorpusSHA && b.ExamSHA == rd.ExamSHA && math.Abs(b.Loss-m) <= a.tolerance().Baseline {
			rd.BaselineN++
		}
	}
	tx.saveRound(rd)
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	n := len(s)
	if n == 0 {
		return math.NaN()
	}
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// --- settlement -----------------------------------------------------------------------------

// anchorResult is the grade that defines an entry's score: the first an anchor reported.
func anchorResult(s *State, e *Entry) *Result {
	var first *Result
	for _, r := range s.results(e.RoundIdx, e.EditorID) {
		if acct := s.Accounts[r.AccountID]; acct != nil && acct.Anchor && (first == nil || r.ID < first.ID) {
			first = r
		}
	}
	return first
}

// settleReady publishes the grades of every round that is ready: an anchor has graded each
// of its entries, and no volunteer is still at work on one. A volunteer who has started
// gets ReplicaWait after the anchor's last grade to finish; after that the round settles
// without them.
func (a *App) settleReady(tx *Tx) {
	s, t := tx.s, now()
	for _, rd := range s.Rounds {
		if rd.Status != "open" {
			continue
		}
		var open []*Entry
		last, graded := int64(0), true
		for _, e := range s.inRound[rd.Idx] {
			if e.Status != "open" {
				continue
			}
			r := anchorResult(s, e)
			if r == nil {
				graded = false
				break
			}
			open = append(open, e)
			last = max(last, r.ReceivedAt)
		}
		if !graded {
			continue
		}
		// The round waits a while for volunteers who are at work on it, but only for those
		// who have finished a job before: an account that has never reported a grade that
		// agreed could otherwise hold every round back just by taking a job.
		waiting := false
		if t < last+int64(a.cfg.ReplicaWait.Seconds()) {
			for _, c := range s.Claims {
				if acct := s.Accounts[c.AccountID]; c.RoundIdx == rd.Idx && c.Kind == "grade" && c.live(t) && acct != nil && !acct.Anchor && acct.Agreed > 0 {
					waiting = true
					break
				}
			}
		}
		if waiting {
			continue
		}
		// A volunteer the round waited for to the end, and who never reported, has let a job
		// run out as surely as one whose lease ended.
		for _, c := range s.Claims {
			if acct := s.Accounts[c.AccountID]; c.RoundIdx == rd.Idx && c.Kind == "grade" && c.live(t) && acct != nil && !acct.Anchor && acct.Agreed > 0 {
				c.Status, c.ExpiresAt = "lapsed", t
				tx.saveClaim(c)
			}
		}
		// Who had standing before this round: settling one entry must not change what a
		// volunteer's grade of the next one is taken for.
		standing := map[int64]bool{}
		for id, acct := range s.Accounts {
			standing[id] = acct.Agreed > 0
		}
		sort.Slice(open, func(i, j int) bool { return open[i].EditorID < open[j].EditorID })
		for _, e := range open {
			a.settleEntry(tx, e, standing)
		}
	}
	a.settleRounds(tx)
}

// abandonStuck settles, as not run, any entry that anchors have tried and failed to edit
// maxEditAttempts times, or to grade maxGradeAttempts times. Without this one bundle or
// one script that upsets the harness would hold its round open for ever, and no later
// round could be drawn.
func (a *App) abandonStuck(tx *Tx) {
	t := now()
	for _, st := range openEntryStates(tx.s, 0) {
		e, why := st.Entry, ""
		switch {
		case e.EditedAt == 0 && !st.Editing && st.EditTries >= maxEditAttempts:
			why = fmt.Sprintf("the anchor could not run this entry in %d attempts; the round does not count for it", maxEditAttempts)
		case e.EditedAt != 0 && !st.Graded && !st.Grading && st.GradeTries >= maxGradeAttempts:
			why = fmt.Sprintf("the anchor could not grade this entry's script in %d attempts; the round does not count for it", maxGradeAttempts)
		default:
			continue
		}
		e.Void = true
		a.settleInvalid(tx, e, why, &EntryDetail{})
		tx.event("abandoned", map[string]any{"round": roundID(a.season.ID, e.RoundIdx), "editor": e.EditorID, "at": t})
	}
}

// settleEntry makes an anchor-graded entry's score public and compares every volunteer's
// grade of it with the anchor's. standing says which accounts had a grade that agreed
// before the round settled.
func (a *App) settleEntry(tx *Tx, e *Entry, standing map[int64]bool) {
	s := tx.s
	rd := s.Rounds[e.RoundIdx]
	rep := anchorResult(s, e)
	if rep == nil || e.Status != "open" {
		return
	}
	baselines := map[int64]*Baseline{}
	for _, b := range s.Baselines {
		if b.RoundIdx == e.RoundIdx {
			baselines[b.AccountID] = b
		}
	}
	tol := a.tolerance()
	detail := &EntryDetail{Verifiers: []string{s.Accounts[rep.AccountID].Handle}, Audited: true}
	if rep.Summary != nil {
		detail.Script, detail.Hardware, detail.Train = rep.Summary.Script, rep.Summary.Hardware, rep.Summary.Train
	}
	// An entry that is the unedited corpus itself has the round's baseline for its loss, and
	// every grader reports the baseline with every grade. Agreeing about it shows nothing
	// that was not already shown, so it earns no standing.
	plain := rep.Valid && rd.BaselineLoss != nil && math.Abs(rep.Loss-*rd.BaselineLoss) < 1e-9
	type differing struct {
		r    *Result
		acct *Account
	}
	var differ []differing
	for _, r := range s.results(e.RoundIdx, e.EditorID) {
		acct := s.Accounts[r.AccountID]
		if r.Verdict != "pending" || acct == nil {
			continue
		}
		if r == rep || acct.Anchor {
			r.Verdict, r.Summary = "agreed", nil
			tx.saveResult(r)
			continue
		}
		// Every volunteer's grade is published, whatever it says and whoever sent it.
		row := Replication{Handle: acct.Handle, Valid: r.Valid, First: !standing[acct.ID]}
		if r.Valid {
			loss := r.Loss
			row.Loss = &loss
			if rep.Valid {
				d := places5(r.Loss - rep.Loss)
				row.Diff = &d
			}
		}
		b := baselines[r.AccountID]
		if b != nil && rd.BaselineLoss != nil {
			d := places5(b.Loss - *rd.BaselineLoss)
			row.BaseDiff = &d
		}
		switch {
		case r.CorpusSHA != rep.CorpusSHA || r.ExamSHA != rep.ExamSHA:
			// It graded different data. A setup problem, and no evidence about this entry.
			r.Verdict, row.Verdict = "void", "other data"
		case b == nil || rd.BaselineLoss == nil || math.Abs(b.Loss-*rd.BaselineLoss) > tol.Baseline:
			// Its arithmetic differs on the unedited corpus too.
			r.Verdict, row.Verdict = "void", "baseline differs"
		case a.agree(rep, r):
			r.Verdict, row.Verdict = "agreed", "agreed"
			if plain {
				r.Verdict = "same" // agreed, and counted for nothing
			} else {
				acct.Agreed++
				acct.Streak++
			}
			acct.GPUSeconds += r.Seconds
			tx.saveAccount(acct)
			detail.Verifiers = append(detail.Verifiers, acct.Handle)
		default:
			// Same script, same corpus, same code, same baseline: a real difference.
			r.Verdict, row.Verdict = "outlier", "differs"
			detail.Disputed++
			differ = append(differ, differing{r, acct})
			tx.event("outlier", map[string]any{"round": roundID(a.season.ID, e.RoundIdx), "editor": e.EditorID, "verifier": acct.Handle, "difference": row.Diff})
		}
		detail.Replications = append(detail.Replications, row)
		r.Summary = nil // it has done its job; the entry keeps the one that counts
		tx.saveResult(r)
	}
	// A volunteer who differs from the anchor alone has a strike against it. Two who differ
	// from the anchor and agree with each other do not: then it is the anchor's grade that
	// is in doubt, and the entry says so.
	for i, v := range differ {
		backed := false
		for j, w := range differ {
			// Both must have agreed with an anchor before, or a second key would be all it
			// took to stand behind an invented number.
			same := v.acct.GithubID != 0 && v.acct.GithubID == w.acct.GithubID
			backed = backed || (i != j && !same && standing[v.acct.ID] && standing[w.acct.ID] && a.agree(v.r, w.r))
		}
		if backed {
			detail.DisputedTogether = true
			continue
		}
		v.acct.Streak = 0
		v.acct.Strikes++
		v.acct.LastStrikeAt = now()
		tx.saveAccount(v.acct)
	}
	if detail.DisputedTogether {
		tx.event("disputed", map[string]any{"round": roundID(a.season.ID, e.RoundIdx), "editor": e.EditorID})
	}
	e.Status, e.Valid, e.Detail, e.NResults, e.SettledAt = "settled", rep.Valid, detail, len(detail.Verifiers), now()
	e.Loss, e.BySource, e.Reason = nil, nil, ""
	if rep.Valid {
		loss := rep.Loss
		e.Loss, e.BySource = &loss, rep.BySource
	} else {
		e.Reason = rep.InvalidReason
	}
	tx.saveEntry(e)
	expireClaims(tx, e)
	a.regain(tx, rd)
	if !e.Valid {
		a.afterInvalid(tx, e.EditorID)
	}
}

// regain recomputes every settled entry's gain in a round from the round's current baseline.
func (a *App) regain(tx *Tx, rd *Round) {
	for _, e := range tx.s.inRound[rd.Idx] {
		if e.Status != "settled" {
			continue
		}
		gain := 0.0
		if e.Valid && e.Loss != nil && rd.BaselineLoss != nil {
			// Losses are reported to five places; so is their difference (not 0.9124699999999999).
			gain = math.Round((*rd.BaselineLoss-*e.Loss)*1e5) / 1e5
		}
		if gain != e.Gain {
			e.Gain = gain
			tx.saveEntry(e)
		}
	}
}

// versionLess compares dotted versions numerically ("0.10.0" is newer than "0.9.1").
func versionLess(x, y string) bool {
	xs, ys := strings.Split(x, "."), strings.Split(y, ".")
	for i := 0; i < len(xs) || i < len(ys); i++ {
		var xi, yi int
		if i < len(xs) {
			fmt.Sscanf(xs[i], "%d", &xi)
		}
		if i < len(ys) {
			fmt.Sscanf(ys[i], "%d", &yi)
		}
		if xi != yi {
			return xi < yi
		}
	}
	return false
}
