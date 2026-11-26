package main

// Jobs, results and settlement: how a number becomes a score.
//
// An entry in a round is worked on in two kinds of job.
//
//   - EDIT. A verifier executes the editor in the sandbox and uploads the edit script it
//     wrote. That is a run. The script is plain data, stored under its sha256.
//   - GRADE. A verifier downloads a script, trains the fixed student on what it says, sits
//     it on the round's exam and reports its score on every item, and, when its job asks
//     for it, the round's unedited baseline's the same way: a few machines measure that
//     for the whole round (exam.go). Grading runs no entrant code, and every grader of a
//     script trains on exactly the same tokens, so their results can be compared directly.
//     The exam is made after the draw (exam.go): grading starts when it is posted.
//
// No score depends on one machine, and none on a machine the organisers run. The
// organisers' accounts start out established, may run their own reference editors, and
// administer the season (the draws, the rounds' exams). In judging they are verifiers like
// any other, with the two exceptions marked (*) below. Both exist so that the competition
// can run while one machine is all there is, and the first is printed on every score it
// touches.
//
// Who may do what:
//
//   - An account is ESTABLISHED when enough of its grades (five, unless configured) have
//     agreed with established accounts' grades of the same scripts, it is not muted, and
//     it has the GitHub identity that entering needs. Standing is earned from accounts that have it,
//     beginning with the organisers'. Once volunteers have it, nobody needs the organisers
//     to earn it.
//   - A RUNNER is an established account whose machine has the sandbox and is at least as
//     fast as the season's floor. Only runners are handed entrants' code.
//   - Anyone may grade. Only an established account's grade can move a score, so nobody
//     else gains by reporting a false one. An account that is not established is handed a
//     script only once its entry is closed to runs (below).
//
// Runs:
//
//   - An entry is run twice, by two different people, and a third time if the first two
//     differ: one failed and the other did not, or the gains of their scripts are further
//     apart than the season's tolerance for runs.
//   - A runner who could give an entry the run that settles it, should its runs differ,
//     is kept clear of its scripts while someone else who is around can grade them.
//   - Nobody runs their own entry (an organisers' machine runs the organisers' reference
//     editors, and grades their scripts, only when nobody else who is around could), nobody
//     runs an entry twice, and nobody who has been handed one of an entry's scripts to
//     grade is then handed the entry to run, or has a run of it taken: they could hand the
//     script back as their own. "Nobody" is a person: an account, and any other key of the
//     same GitHub identity.
//   - A verifier holds one job at a time, except that a grade whose student it trained
//     before the round's exam, and which waits only to sit it, leaves it free to take
//     another grade (pace.go). One in the middle of running an entry is handed nothing else
//     until that job is reported, handed back or out of time, and none of the entry's
//     scripts goes to another key of theirs meanwhile.
//   - An entry is CLOSED TO RUNS once one of its scripts has been handed to an account that
//     is not established. Such an account is a key, and keys are free: whoever made it
//     could be the entry's next runner. So it is handed a script only when the entry has
//     the runs it wants, or nobody who is around could be handed it to run and nobody is
//     running it; and from then on the entry takes no more runs, a third included. An
//     account that is not established and hands a script back, or lets the job run out,
//     is handed no more scripts of that round: asking and handing back would otherwise
//     show one key every script there is.
//   - A run that was given up (nobody could check it) does not use up one of the entry's
//     runs: a runner who comes later may give it another. Three runs is the most.
//   - First runs come before second runs, and a second run goes to the other make of GPU
//     when there is one.
//   - A job is offered once. Handing it back does not buy another go. (*) An organisers'
//     machine is offered a job again after one that came to nothing, a few times over:
//     while it is the only machine there is, a restart must not cost an entry its round.
//
// Grades:
//
//   - A grade names one script. Two runs that wrote the same bytes share one set of grades.
//   - A script's gain is the median, over the established accounts that graded it and
//     found it valid, of each one's score against the round's baseline: the exam's
//     aggregate of its item scores less the baseline's. If most of them find it invalid,
//     or as many as do not, it is invalid.
//   - A grade of a script by the only person who wrote it is a SELF grade: that person
//     would control both the script and the number said about it. A volunteer is never
//     handed its own script. (*) An organisers' machine is, when nobody else who is around
//     could grade it; the grade counts only until someone else's exists, and an entry
//     resting on one is never called confirmed.
//   - Two established grades of a script that disagree get a third, and the median decides.
//   - A grade that differs from a script's gain is a strike only when that gain is one two
//     established grades agree on, or an organisers' machine's. Against one volunteer's
//     grade, or two that disagree, it is published as UNRESOLVED and held against nobody.
//   - A script that two runs wrote is the entry's only script, so one grade of it would
//     be one machine's word for the whole score: it gets a second, from someone else, when
//     someone else who is around could give it. An entry's scripts go to different graders
//     when there is other work for them. An entry is called CONFIRMED only when two
//     people's runs agree and more than one person's grades decide them; when one
//     person's do, it says so (ONE GRADER), and so it does when the runs that agree are
//     all one person's (ONE RUNNER). The organisers' machines are one person's.
//   - Everyone else's grade is compared with the script's gain when the round settles. One
//     that agrees earns standing; one that does not is a strike. Agreeing about the
//     unedited corpus, or that a script is invalid, earns nothing: neither takes training.
//   - Nobody chooses their job, grades their own entry, or is offered a script twice. An
//     account whose grades keep disagreeing stops being given work.
//
// Scores:
//
//   - A run's gain is its script's gain, or nothing if it failed (no script, or an invalid
//     one). An entry's gain is the median of its runs' gains (the mean of two), in bits per
//     byte. The round's baseline is the mean, item by item, of what its established graders
//     measured on the unedited corpus: the season says how many of them are asked to. A
//     grade is not scored, and settles nothing, until the round has one.
//   - An entry is CONFIRMED when two people's runs agree and neither rests on a self
//     grade. Otherwise its score says which it is: run once, runs differed, or graded by
//     its own runner.
//
// Sealing and settling:
//
//   - Nothing about an entry's runs, scripts or grades is public until its round settles,
//     and nothing is accepted afterwards. That includes whether the entry has been given
//     up early, and what its runners are credited with: a run is added to its runner's
//     count, and its time to their GPU time, when the entry settles and only if the run
//     counted.
//   - A script's hash is told only to whoever is handed it to grade, and until the round
//     settles its bytes are served only to the account that holds it as a grade job (one
//     under way, or one it has reported), in a signed request. After that it is public.
//     So a second run cannot be a copy of the first, unless two established accounts work
//     together: an established account may be handed a script while its entry can still
//     be run, and nothing stops it passing the script to the runner. That is NOT caught.
//     (Nor is it new: two established accounts that agree with each other already decide
//     a score.)
//   - A round settles as a whole: when every entry has the runs and grades it wants; or
//     when every entry has at least one graded run, no established verifier is at work on
//     the round, and nobody who is around could be handed a job a score is waiting for.
//     So one machine alone settles its rounds as it finishes them, and the moment a second
//     is there every entry gets its second run.
//   - A verifier is around if it asked for work a short while ago, or holds a job: one
//     that is training says nothing, and is not gone. An established verifier's job holds
//     the round until it is reported, handed back, or out of time. For anyone else who
//     has had a grade agree the round waits a fixed while, and for a newcomer not at all:
//     keys are free.
//   - What nobody is there to do is given up after the same fixed while: a script nobody
//     else could grade, a failed run nobody else could repeat, and an entry with no run
//     that nobody around may run, when other entries' scores are waiting behind it or a
//     verifier who runs editors is there (it is their own entry, or they took it and did
//     not finish). Without the last a round whose only runners may not run what is left
//     would stay open for ever. A round nobody who runs editors has come to still waits.
//     A round in which that leaves an entry with nothing is void for it: counted neither
//     for nor against.

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
	maxReplicas        = 3  // grades of one script by accounts not yet established, reported or under way
	muteAfterStrikes   = 3  // disagreements before an account stops being given work...
	muteUnlessAgreed   = 10 // ...unless it has this many agreements for each of them
	retireAfterInvalid = 2
	maxRuns            = 3          // runs of one entry in one round, at most
	maxEditAttempts    = 3          // edit jobs on one entry that came to nothing before it is given up
	maxGradeAttempts   = 5          // grade jobs on one script, handed to established accounts, that came to nothing
	muteFor            = 14 * 86400 // how long a mute lasts after the latest disagreement
	lapseRest          = 86400      // no new work for this long after letting a job run out
)

// muted reports whether an account is being given no work: three grades that differed
// from what established accounts found, and not ten agreements for each. It lifts by
// itself, muteFor after the latest such grade, because the machine may be honest and
// merely different, and the only way to find out is to let it try again.
func muted(c *Account, t int64) bool {
	return !c.Anchor && c.Strikes >= muteAfterStrikes && c.Agreed < muteUnlessAgreed*c.Strikes && t < c.LastStrikeAt+muteFor
}

// established reports whether an account's grades can move a score, and whether it may be
// handed entrants' code: the organisers' accounts, and anyone who has earned it.
func (a *App) established(acct *Account) bool {
	if acct == nil {
		return false
	}
	if acct.Anchor {
		return true
	}
	return !muted(acct, now()) && acct.Agreed >= a.cfg.EstablishAfter && a.mayEnter(acct) == nil
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
	// Early says that it trains before the exam: handed a grade before the round's exam is
	// posted, it trains the student, and the round's unedited one if the job asks for it,
	// waits for the exam, and sits it. A verifier that does not say so is handed grades once
	// the exam is there.
	Early bool `json:"early"`
	// CuratorsDeclined is why its last `stet verify --edits` ran no curators on this machine,
	// in the client's words, said by a verifier that does not ask to run them (stet 0.4.6).
	CuratorsDeclined string `json:"curators_declined,omitempty"`
	// Trained lists the grade jobs it holds whose students it has trained, and which wait
	// only for the exam (stet 0.4.4 and later). It is not handed them again, and while all it
	// holds are such, it may be handed another grade before the exam (claim).
	Trained []int64 `json:"trained,omitempty"`
	// PassSeconds is how long this machine last took to train a grade's student, kept on its
	// disk from its latest grade (stet 0.4.4 and later); 0 until it has graded.
	PassSeconds float64 `json:"pass_seconds,omitempty"`
}

// hardwareFacts is the part of a hardware description the server reads. All of it is
// whatever the client sent.
type hardwareFacts struct {
	Device  string `json:"device"`
	GPU     string `json:"gpu"`
	Backend string `json:"backend"`
	// Bench is the machine's measurement of itself against the season's floor: see
	// src/stet/bench.py.
	Bench *struct {
		GPUTokensPerSecond float64 `json:"gpu_tokens_per_second"`
		CPUMBPerSecond     float64 `json:"cpu_mb_per_second"`
		Cores              float64 `json:"cores"`
		MemoryGB           float64 `json:"memory_gb"`
		GPUMemoryGB        float64 `json:"gpu_memory_gb"`
		DiskGB             float64 `json:"disk_gb"` // free under its cache (stet 0.4.6 and later)
	} `json:"bench"`
}

func readHardware(raw json.RawMessage) hardwareFacts {
	var h hardwareFacts
	json.Unmarshal(raw, &h)
	return h
}

// gpuName is the GPU's name if it looks like one. It is shown on the site.
func (h hardwareFacts) gpuName() string {
	switch {
	case gpuNameRe.MatchString(h.GPU):
		return h.GPU
	case gpuNameRe.MatchString(h.Device):
		return h.Device
	}
	return ""
}

// gpuMake is which maker's arithmetic the GPU does: cuda (NVIDIA) or rocm (AMD).
func (h hardwareFacts) gpuMake() string {
	if h.Backend == "cuda" || h.Backend == "rocm" {
		return h.Backend
	}
	return ""
}

// deviceOf reads the kind of device out of a hardware description.
func deviceOf(hardware json.RawMessage) string { return readHardware(hardware).Device }

// A machine sold with 16 GB reports about 15.5 of it: firmware and the kernel keep some
// back. The floor is in the gigabytes people know their machine by, so this much is
// allowed for. (The client makes the same allowance: bench.py.)
const memoryAllowanceGB = 1.0

// meetsFloor reports whether a machine says it is at least the slowest one that may run
// editors. An editor has a time limit in seconds, so every machine that runs one has to
// be at least that fast.
func (a *App) meetsFloor(hardware json.RawMessage) bool { return a.shortOfFloor(hardware) == "" }

// shortOfFloor says where a machine falls short of the season's floor for running editors,
// as it measured itself, e.g. "CPU 95 MB/s, floor 100"; empty if it does not. Organisers'
// tools read the word CPU in it to tell a host's slow processor from the rest.
func (a *App) shortOfFloor(hardware json.RawMessage) string {
	f := a.season.Runner
	if a.cfg.Dev || (f.GPUTokensPerSecond == 0 && f.CPUMBPerSecond == 0 && f.GPUMemoryGB == 0 && f.MemoryGB == 0 && f.Cores == 0) {
		return ""
	}
	b := readHardware(hardware).Bench
	if b == nil {
		return "it sent no measurement of itself (stet bench)"
	}
	var out []string
	if b.GPUTokensPerSecond < f.GPUTokensPerSecond {
		out = append(out, fmt.Sprintf("GPU %.0f tokens/s, floor %.0f", b.GPUTokensPerSecond, f.GPUTokensPerSecond))
	}
	if b.GPUMemoryGB < f.GPUMemoryGB {
		out = append(out, fmt.Sprintf("GPU memory %.0f GB, floor %.0f", b.GPUMemoryGB, f.GPUMemoryGB))
	}
	if b.CPUMBPerSecond < f.CPUMBPerSecond {
		out = append(out, fmt.Sprintf("CPU %.0f MB/s, floor %.0f", b.CPUMBPerSecond, f.CPUMBPerSecond))
	}
	if b.Cores < float64(f.Cores) {
		out = append(out, fmt.Sprintf("CPU %.0f cores, floor %d", b.Cores, f.Cores))
	}
	if b.MemoryGB < f.MemoryGB-memoryAllowanceGB {
		out = append(out, fmt.Sprintf("memory %.1f GB, floor %.0f", b.MemoryGB, f.MemoryGB))
	}
	if b.DiskGB > 0 && b.DiskGB < f.DiskGB { // said from stet 0.4.6; checked by the client before
		out = append(out, fmt.Sprintf("disk %.0f GB free, floor %.0f", b.DiskGB, f.DiskGB))
	}
	return strings.Join(out, "; ")
}

// runsEditors says whether an account asking for work may be handed editors to run, and
// if it asked to and may not, why.
func (a *App) runsEditors(acct *Account, h *Hello) (bool, string) {
	switch {
	case !h.Edits:
		return false, ""
	case muted(acct, now()):
		return false, "this account is given no work for now"
	case a.mayEnter(acct) != nil:
		return false, "running curators needs a linked GitHub account, as entering does: stet login --github"
	case !a.established(acct):
		return false, fmt.Sprintf("curators are run by established verifiers: accounts with %d grades that agreed. This one has %d", a.cfg.EstablishAfter, acct.Agreed)
	case h.Sandbox == "container" && !acct.Anchor && !a.cfg.Dev:
		// The container sandbox (stet's cage.py) is for a container that runs stet and
		// nothing else, which the organisers' rented machines are and a volunteer's
		// machine may not be. It is weaker than bubblewrap: the curator sees the machine's
		// other processes, and its scratch space is on disk.
		return false, "the container sandbox is for the organisers' rented machines; a volunteer's runs curators in bubblewrap"
	case h.Sandbox != "bwrap" && h.Sandbox != "container" && !a.cfg.Dev:
		return false, "running curators needs the bubblewrap sandbox"
	}
	if short := a.shortOfFloor(h.Hardware); short != "" {
		return false, floorWhy + short
	}
	return true, ""
}

const floorWhy = "this machine is under the season's floor for running curators: "

// curatorsWhy is why a machine that asked for work as h did may not be handed curators, for
// the organisers' tools (/admin/work). If it asked to run them and may not, why. If it did
// not ask: where its own measurement of itself falls short of the floor, when it sent one
// under it, or else that it did not ask; then, after a colon, why it said its last
// `stet verify --edits` ran none (stet 0.4.6), or what it sent. The server's own words come
// first, so that the organisers' tools can read the word CPU in them. Empty if it may.
func (a *App) curatorsWhy(canEdit bool, why string, h *Hello) string {
	switch {
	case canEdit:
		return ""
	case h.Edits:
		return why
	}
	out, short := "it did not ask to run curators", ""
	bench := readHardware(h.Hardware).Bench != nil
	if bench {
		if short = a.shortOfFloor(h.Hardware); short != "" {
			out = floorWhy + short
		}
	}
	switch said := clip(strings.ToValidUTF8(h.CuratorsDeclined, "?"), 200); {
	case said != "":
		out += ": " + said
	case !bench:
		out += ": it sent no measurement of itself"
	case short == "":
		out += ": the speeds it sent meet the floor"
	}
	return out
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
	// Baseline is the round's unedited student's score, from a machine whose job asked for
	// it (Job.Baseline), or that had the round's on its disk anyway. Absent otherwise.
	Baseline *ExamScore `json:"baseline"`
	Exam     *ExamScore `json:"exam"`
	Train    *TrainInfo `json:"train"`
}

// ExamScore is a student's score on the round's exam: one score for each of its items (bits
// per byte, higher is better), and their mean.
type ExamScore struct {
	Mean       float64   `json:"mean"`
	ItemScores []float64 `json:"item_scores"`
}

type TrainInfo struct {
	Seconds         float64     `json:"seconds"`
	TokensPerSecond float64     `json:"tokens_per_second"`
	FinalTrainLoss  float64     `json:"final_train_loss"`
	InitSHA256      string      `json:"init_sha256,omitempty"`
	Precision       string      `json:"precision,omitempty"`
	Curve           [][]float64 `json:"curve,omitempty"`
}

// EditReport is what a runner says about an edit run. Mirrors stet.runner.EditRun.
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

// ScriptPayload is a runner's report of an edit job, script included.
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
	t, ok := a.season.Tolerance[a.cfg.Scale]
	if !ok {
		t = Tolerance{Grade: 0.1, Baseline: 0.1}
	}
	if t.Run <= 0 { // a season from before runs were compared
		t.Run = 1.5 * t.Grade
	}
	return t
}

// recentAgreed counts each account's grades that agreed in the last two weeks: what the
// verifier board shows.
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

// recentWork counts what each account did for other people's entries in the last two
// weeks: grades that agreed, grades that stood alone as a script's gain, and editors run.
// It is what puts a volunteer's own entries ahead in the queue.
func recentWork(s *State) map[int64]int {
	since := now() - 14*86400
	out := recentAgreed(s)
	for _, r := range s.Results {
		if r.Verdict == "counted" && r.ReceivedAt > since {
			out[r.AccountID]++
		}
	}
	for _, es := range s.inRound {
		for _, e := range es {
			for _, run := range e.Runs {
				if e.Status == "settled" && !run.Dropped && run.At > since {
					out[run.By]++
				}
			}
		}
	}
	return out
}

// within reports whether a difference is inside a tolerance. Scores are reported to five
// places, and 4.48954 - 4.40954 is 0.08000000000000007 in floating point: a difference of
// exactly the tolerance agrees, as the page that prints both numbers says it does.
func within(d, tol float64) bool { return math.Abs(d) <= tol+1e-9 }

// samePerson reports whether two accounts are one person: one account, or two keys linked
// to one GitHub identity.
func samePerson(x, y *Account) bool {
	return x.ID == y.ID || (x.GithubID != 0 && x.GithubID == y.GithubID)
}

// --- what an entry has, and what it still wants ----------------------------------------------

// judge is one look at the state: who is established, who is around and who is being
// given no work, all as they were when the look began. Settling a round changes accounts'
// standing as it goes, and settling one entry must not change what the grades of the next
// are taken for.
type judge struct {
	a        *App
	s        *State
	t        int64
	est      map[int64]bool        // established
	standing map[int64]bool        // has had a grade agree
	busy     map[int64]bool        // holds a job
	training map[int64]bool        // holds a job other than a grade it has trained, waiting for the exam
	resting  map[int64]bool        // let a job run out within the last day
	differs  map[int64]map[int]int // jobs turned away for building a round differently, by round
	// quit: accounts that were handed a script of a round and handed it back or let it run
	// out. One that is not established is handed no more of that round's scripts: asking
	// and handing back would otherwise show it every script there is.
	quit     map[int64]map[int]bool
	around   []*asker                      // found when first asked for
	passes   map[string]map[string]float64 // a training pass's seconds, by scale and GPU: worked out when first asked for
	measures map[int]map[int64]bool        // whose baseline each round has or is promised (measuring): worked out when first asked for
	left     map[int]map[int64]bool        // whose baseline each round leaves out of its own (oddOut): worked out when first asked for
}

func (a *App) newJudge(s *State) *judge {
	j := &judge{a: a, s: s, t: now(), est: map[int64]bool{}, standing: map[int64]bool{}, busy: map[int64]bool{}, training: map[int64]bool{},
		resting: map[int64]bool{}, differs: map[int64]map[int]int{}, quit: map[int64]map[int]bool{}, measures: map[int]map[int64]bool{}, left: map[int]map[int64]bool{}}
	for id, acct := range s.Accounts {
		j.est[id], j.standing[id] = a.established(acct), acct.Agreed > 0
	}
	for _, c := range s.Claims {
		if c.Kind == "grade" && (c.Status == "released" || c.Status == "lapsed" || (c.Status == "claimed" && !c.live(j.t))) {
			if j.quit[c.AccountID] == nil {
				j.quit[c.AccountID] = map[int]bool{}
			}
			j.quit[c.AccountID][c.RoundIdx] = true
		}
		switch {
		case c.live(j.t):
			j.busy[c.AccountID] = true
			j.training[c.AccountID] = j.training[c.AccountID] || c.TrainedAt == 0
		case c.Status == "lapsed" && c.ExpiresAt > j.t-lapseRest:
			j.resting[c.AccountID] = true
		case c.Status == "refused":
			if j.differs[c.AccountID] == nil {
				j.differs[c.AccountID] = map[int]int{}
			}
			j.differs[c.AccountID][c.RoundIdx]++
		}
	}
	return j
}

// barred reports whether an account is being given none of a round's work: it is muted, it
// let a job run out within the last day, or its machine has twice built this round
// differently from everyone else and would do so a third time.
func (j *judge) barred(acct *Account, round int) bool {
	if muted(acct, j.t) {
		return true
	}
	return !acct.Anchor && (j.resting[acct.ID] || j.differs[acct.ID][round] >= 2)
}

// asker is an account as it stands when it asks for work, or as it stood when it last did.
type asker struct {
	acct    *Account
	est     bool
	canEdit bool   // it may be handed editors to run
	gpuMake string // cuda | rocm
	early   bool   // it trains before the exam, and may be handed a grade before it is posted
}

// active lists the accounts that are around: those that asked for work a short while ago,
// and those at work on a job now. A verifier says nothing to the server while it trains,
// so a silent one that holds a job is busy, not gone.
func (j *judge) active() []*asker {
	if j.around == nil {
		j.around = []*asker{}
		since := j.t - int64(j.a.cfg.ActiveWindow.Seconds())
		for id, acct := range j.s.Accounts {
			if acct.LastSeen > since || j.busy[id] {
				j.around = append(j.around, &asker{acct: acct, est: j.est[id], canEdit: j.est[id] && acct.RunnerSeen != 0, gpuMake: acct.Backend})
			}
		}
		sort.Slice(j.around, func(x, y int) bool { return j.around[x].acct.ID < j.around[y].acct.ID })
	}
	return j.around
}

// scriptState is one script an entry's runs wrote, and what has been said about it.
type scriptState struct {
	SHA     string
	Size    int64
	Run     int            // the first run that wrote it
	Writers []writer       // whose runs wrote it
	Results []*Result      // grades of it, in the order they came
	Live    []*Claim       // grade jobs on it under way
	Offered map[int64]bool // accounts that have been handed it to grade, whatever came of it
	Graded  map[int64]bool // and those that reported
	Tries   int            // grade jobs on it, handed to established accounts, that came to nothing
	verdict *verdict       // worked out once
}

// writer is whoever's run wrote a script: the account, and the GitHub identity it had at
// the time, which may since have moved to another key.
type writer struct {
	acct *Account
	gh   int64
}

func (wr writer) is(acct *Account) bool {
	return samePerson(wr.acct, acct) || (wr.gh != 0 && wr.gh == acct.GithubID)
}

// wroteAlone reports whether a script is this person's own: their run wrote it, and
// nobody else's did. A script that two people's runs both wrote is nobody's own: neither
// chose its bytes.
func (sc *scriptState) wroteAlone(acct *Account) bool {
	for _, wr := range sc.Writers {
		if !wr.is(acct) {
			return false
		}
	}
	return len(sc.Writers) > 0
}

// entryState is everything about an open entry that deciding its next job needs.
type entryState struct {
	Entry       *Entry
	Round       *Round
	OwnerID     int64
	OwnerGH     int64          // the GitHub identity the editor was entered under, wherever it is linked now
	Runs        []*Run         // those that count
	Dropped     int            // and those that were given up
	Scripts     []*scriptState // what the runs that count wrote, in order
	byScript    map[string]*scriptState
	LiveEdits   []*Claim       // edit jobs under way
	EditOffered map[int64]int  // edit jobs each account has been handed for it
	EditFailed  int            // edit jobs that came to nothing
	GradeSeen   map[int64]bool // accounts that have been handed any of its scripts
	Shown       bool           // one of them is not established: the entry takes no more runs
	runOpen     *bool          // somebody is running it, or somebody who is around could be handed it to run: worked out once
}

// states describes the open entries of a round.
func (j *judge) states(rd *Round) []*entryState {
	s := j.s
	index := map[string]*entryState{}
	var out []*entryState
	for _, e := range s.inRound[rd.Idx] {
		if e.Status != "open" {
			continue
		}
		ed := s.Editors[e.EditorID]
		st := &entryState{Entry: e, Round: rd, OwnerID: ed.AccountID, OwnerGH: ed.GithubID, byScript: map[string]*scriptState{},
			EditOffered: map[int64]int{}, GradeSeen: map[int64]bool{}}
		for _, run := range e.Runs {
			if run.Dropped {
				st.Dropped++
				continue
			}
			st.Runs = append(st.Runs, run)
			if run.Script == "" {
				continue
			}
			sc := st.byScript[run.Script]
			if sc == nil {
				sc = &scriptState{SHA: run.Script, Size: run.ScriptSize, Run: run.N, Offered: map[int64]bool{}, Graded: map[int64]bool{}}
				st.byScript[run.Script] = sc
				st.Scripts = append(st.Scripts, sc)
			}
			if by := s.Accounts[run.By]; by != nil {
				sc.Writers = append(sc.Writers, writer{by, run.GithubID})
			}
		}
		index[entryKey(e.RoundIdx, e.EditorID)] = st
		out = append(out, st)
	}
	sort.Slice(out, func(x, y int) bool { return out[x].Entry.EditorID < out[y].Entry.EditorID })
	for k, st := range index {
		for _, r := range s.resultsOf[k] {
			st.GradeSeen[r.AccountID] = true
			st.Shown = st.Shown || !j.est[r.AccountID]
			if sc := st.byScript[r.Script]; sc != nil {
				sc.Results = append(sc.Results, r)
				sc.Offered[r.AccountID], sc.Graded[r.AccountID] = true, true
			}
		}
	}
	// In the order the jobs were handed out, so that nothing below depends on a map's.
	ids := make([]int64, 0, len(s.Claims))
	for id, c := range s.Claims {
		if c.RoundIdx == rd.Idx {
			ids = append(ids, id)
		}
	}
	sort.Slice(ids, func(x, y int) bool { return ids[x] < ids[y] })
	for _, id := range ids {
		c := s.Claims[id]
		st := index[entryKey(c.RoundIdx, c.EditorID)]
		if st == nil {
			continue
		}
		if c.Kind == "edit" {
			st.EditOffered[c.AccountID]++
			switch {
			case c.live(j.t):
				st.LiveEdits = append(st.LiveEdits, c)
			case c.Status != "done":
				st.EditFailed++
			}
			continue
		}
		st.GradeSeen[c.AccountID] = true
		st.Shown = st.Shown || !j.est[c.AccountID]
		sc := st.byScript[c.Script]
		if sc == nil {
			continue
		}
		sc.Offered[c.AccountID] = true
		switch {
		case c.live(j.t):
			sc.Live = append(sc.Live, c)
		case c.Status != "done" && j.est[c.AccountID]:
			sc.Tries++
		}
	}
	return out
}

// openRounds lists the rounds being worked on, oldest first. (There is one at a time.)
func openRounds(s *State) []*Round {
	var out []*Round
	for _, r := range s.Rounds {
		if r.Status == "open" {
			out = append(out, r)
		}
	}
	sort.Slice(out, func(x, y int) bool { return out[x].Idx < out[y].Idx })
	return out
}

// gradeView is one grade of a script, with what settling it needs to know.
type gradeView struct {
	r    *Result
	acct *Account
	est  bool
	self bool    // by the only person who wrote the script
	base bool    // the round has a baseline it was scored against
	gain float64 // its score against that
}

func (j *judge) view(sc *scriptState, r *Result) *gradeView {
	acct := j.s.Accounts[r.AccountID]
	if acct == nil {
		return nil
	}
	g := &gradeView{r: r, acct: acct, est: j.est[acct.ID], self: sc.wroteAlone(acct)}
	// Against the round's baseline, the median of those its graders measured: not this
	// grader's own, which it may not have been asked for. A grade has a score once the
	// round has a baseline (refreshScores).
	g.base = !r.Valid || r.Score != nil
	if r.Valid && r.Score != nil {
		g.gain = *r.Score
	}
	return g
}

// verdict is what the established accounts' grades of a script come to.
type verdict struct {
	counted []*gradeView // the grades that decide it
	self    bool         // which are its own writer's, because there are no others
	known   bool         // there is at least one
	// scored: and each of them has its score against the round's baseline. Until the
	// baseline somebody was asked for arrives, a grade that counts has none, and gain
	// says nothing yet: nothing is settled on it, and no two runs are compared by it.
	scored bool
	valid  bool
	gain   float64
	tie    bool // two decide it, and they disagree
}

// same reports whether two grades of a script are consistent with honest runs.
func (j *judge) same(x, y *gradeView) bool {
	if x.r.Valid != y.r.Valid {
		return false
	}
	if x.r.Valid {
		if x.r.InitSHA != "" && y.r.InitSHA != "" && x.r.InitSHA != y.r.InitSHA {
			return false
		}
		if x.r.TokensSHA != "" && y.r.TokensSHA != "" && x.r.TokensSHA != y.r.TokensSHA {
			return false
		}
	}
	return !x.r.Valid || within(x.gain-y.gain, j.a.tolerance().Grade)
}

// agrees reports whether a grade is consistent with what a script was found to be.
func (j *judge) agrees(g *gradeView, v *verdict) bool {
	if g.r.Valid != v.valid {
		return false
	}
	return !v.valid || within(g.gain-v.gain, j.a.tolerance().Grade)
}

// weigh works out what a script's grades come to so far. The established accounts' grades
// decide; a self grade counts only while it is all there is.
func (j *judge) weigh(sc *scriptState) *verdict {
	if sc.verdict != nil {
		return sc.verdict
	}
	v := &verdict{}
	var own []*gradeView
	for _, r := range sc.Results {
		rd := j.s.Rounds[r.RoundIdx]
		switch g := j.view(sc, r); {
		case g == nil || !g.est:
		case r.Valid && rd != nil && rd.InitSHA != "" && r.InitSHA != "" && r.InitSHA != rd.InitSHA:
			// Graded from different initial student weights than the round's agreed init: set aside
			// so it cannot decide the script and is voided as "other data" at settlement.
		case j.oddOut(r):
			// Its machine's baseline is the odd one out of the round's: the script wants a
			// grade as if it had none, and this one is set aside when the round settles.
		case !g.base && !j.baselineComing(r.RoundIdx):
			// It has no score, and as things stand never will: the round has no baseline,
			// and nobody at work on it was asked for one. So the script wants a grade as if
			// it had none, and whoever is handed it is asked for the baseline (asksBaseline).
			// The moment somebody is, this grade counts again, and is scored when theirs comes.
		case g.self:
			own = append(own, g)
		default:
			v.counted = append(v.counted, g)
		}
	}
	if len(v.counted) == 0 {
		v.counted, v.self = own, len(own) > 0
	}
	v.known = len(v.counted) > 0
	v.scored = v.known
	var gains []float64
	for _, g := range v.counted {
		v.scored = v.scored && g.base
		if g.r.Valid {
			gains = append(gains, g.gain)
		}
	}
	// Most of them must find it valid. As many for as against is not enough.
	if v.valid = 2*len(gains) > len(v.counted); v.valid {
		v.gain = median(gains)
	}
	v.tie = len(v.counted) == 2 && !j.same(v.counted[0], v.counted[1])
	sc.verdict = v
	return v
}

// middleOf is the part of a sorted list that its median is made of: the middle one of an
// odd number, the middle two of an even one.
func middleOf[T any](sorted []T) []T {
	if len(sorted) == 0 {
		return nil
	}
	return sorted[(len(sorted)-1)/2 : len(sorted)/2+1]
}

// standIns says which finished run stands in for each of an entry's runs that failed: the
// entry's worst finished run, when that is below zero. A failed run otherwise counts zero,
// and a curator below zero would halve its loss by failing one run in two. Failing is never
// better than finishing.
func standIns(runs []*Run) map[*Run]*Run {
	var worst *Run
	for _, run := range runs {
		if run.Valid && (worst == nil || run.Gain < worst.Gain) {
			worst = run
		}
	}
	out := map[*Run]*Run{}
	if worst == nil || worst.Gain >= 0 {
		return out
	}
	for _, run := range runs {
		if !run.Valid {
			out[run] = worst
		}
	}
	return out
}

// pillars is a script's gain on each of the exam's pillars: the mean, pillar by pillar, over
// the grades its gain is the median of. With one grade that decides it, or two, those are all
// of them; of three, it is the middle one, so that a grade the median went against moves the
// pillars no more than it moves the gain.
func (v *verdict) pillars() map[string]float64 {
	var valid []*gradeView
	for _, g := range v.counted {
		if g.r.Valid {
			valid = append(valid, g)
		}
	}
	sort.SliceStable(valid, func(x, y int) bool { return valid[x].gain < valid[y].gain })
	made := middleOf(valid)
	out := map[string]float64{}
	for _, g := range made {
		for k, x := range g.r.BySource {
			out[k] += x / float64(len(made))
		}
	}
	return out
}

// runsWanted is how many runs an entry should have: the usual number, and one more when
// two of those it has are known to differ.
func (j *judge) runsWanted(st *entryState) int {
	n := j.a.cfg.RunsPerEntry
	if n >= maxRuns {
		return n
	}
	type outcome struct {
		failed bool
		gain   float64
	}
	var known []outcome
	for _, run := range st.Runs {
		if run.Script == "" {
			known = append(known, outcome{failed: true})
		} else if v := j.weigh(st.byScript[run.Script]); v.scored {
			known = append(known, outcome{failed: !v.valid, gain: v.gain})
		}
	}
	tol := j.a.tolerance().Run
	for x := range known {
		for y := x + 1; y < len(known); y++ {
			if known[x].failed != known[y].failed || (!known[x].failed && !within(known[x].gain-known[y].gain, tol)) {
				return n + 1
			}
		}
	}
	if n == 1 && len(st.Runs) == 1 && st.Runs[0].Script == "" && !j.vouched(st, st.Runs[0]) {
		return 2 // one volunteer says the editor failed: somebody else has to find the same
	}
	return n
}

// owns reports whether an entry is this person's: entered by this account, or under the
// GitHub identity this account has, wherever that identity is linked now.
func (j *judge) owns(st *entryState, acct *Account) bool {
	if st.OwnerID == acct.ID {
		return true
	}
	owner := j.s.Accounts[st.OwnerID]
	return acct.GithubID != 0 && (st.OwnerGH == acct.GithubID || (owner != nil && owner.GithubID == acct.GithubID))
}

// own reports whether an entry is this account's own when it comes to who runs and grades
// it: its owner's, and, for an organisers' machine, any entry of the organisers'. Two of
// their machines are not two parties, so neither stands in for a volunteer to the other.
func (j *judge) own(st *entryState, acct *Account) bool {
	if j.owns(st, acct) {
		return true
	}
	owner := j.s.Accounts[st.OwnerID]
	return acct.Anchor && owner != nil && owner.Anchor
}

// mayRun reports whether a runner could be handed an entry to run.
func (j *judge) mayRun(st *entryState, q *asker) bool {
	// A run that was given up, because nobody could check it, does not use up one of the
	// entry's runs: somebody who comes later may give it another. Three is still the most.
	if st.EditFailed >= maxEditAttempts || len(st.Runs)+len(st.LiveEdits) >= j.runsWanted(st) || len(st.Entry.Runs)+len(st.LiveEdits) >= maxRuns {
		return false
	}
	return j.couldRun(st, q)
}

// spare reports whether an entry may yet need a run from this account: its runs may turn
// out to differ, and this is somebody who could give the one that settles it. Whoever has
// seen one of an entry's scripts cannot run it, so such an account is kept clear of them
// while someone else who is around can grade them.
func (j *judge) spare(st *entryState, q *asker) bool {
	if j.a.cfg.RunsPerEntry >= maxRuns || st.EditFailed >= maxEditAttempts || len(st.Entry.Runs)+len(st.LiveEdits) >= maxRuns {
		return false
	}
	return j.mayDiffer(st) && j.couldRun(st, q)
}

// mayDiffer reports whether the runs an entry has, or is getting, may yet be found to
// differ. Runs that all wrote one script, or all wrote none, cannot.
func (j *judge) mayDiffer(st *entryState) bool {
	if len(st.LiveEdits) > 0 || len(st.Runs) < j.a.cfg.RunsPerEntry {
		return true
	}
	same, known := true, true
	for _, run := range st.Runs {
		same = same && run.Script == st.Runs[0].Script
		known = known && (run.Script == "" || j.weigh(st.byScript[run.Script]).scored)
	}
	if same {
		return false
	}
	return !known || len(st.Runs) < j.runsWanted(st)
}

// unspared reports whether somebody who is around, other than this account, could give a
// script a grade that counts without being a runner the entry may yet need.
func (j *judge) unspared(st *entryState, sc *scriptState, not *Account) bool {
	for _, q := range j.active() {
		if q.acct.ID != not.ID && q.est && !sc.wroteAlone(q.acct) && j.mayGrade(st, sc, q.acct) && !j.running(st, q.acct) && !j.spare(st, q) {
			return true
		}
	}
	return false
}

// couldRun reports whether an entry is one this runner could be handed, if it wanted a run.
func (j *judge) couldRun(st *entryState, q *asker) bool {
	acct := q.acct
	if !q.canEdit || j.barred(acct, st.Round.Idx) {
		return false
	}
	// Whoever has been handed one of its scripts has seen what a run of it writes. And
	// once one has gone to an account that is not established the entry takes no more
	// runs from anybody: such an account is a key, keys are free, and whoever made it could
	// be the next runner.
	if st.Shown || j.sawScript(st, acct) {
		return false
	}
	// Nobody runs their own entry. The organisers' machines may run the organisers' own,
	// which is how the reference editors get run at all while a season starts, but only
	// when nobody else who is around could be handed it.
	if j.own(st, acct) && (!acct.Anchor || j.otherRunner(st, acct)) {
		return false
	}
	// A job is offered once. An organisers' machine gets one that came to nothing again.
	if st.EditOffered[acct.ID] > 0 && !acct.Anchor {
		return false
	}
	for _, run := range st.Entry.Runs {
		if by := j.s.Accounts[run.By]; run.By == acct.ID || (by != nil && (writer{by, run.GithubID}).is(acct)) {
			return false // one run of an entry each
		}
	}
	for _, c := range st.LiveEdits {
		if by := j.s.Accounts[c.AccountID]; c.AccountID == acct.ID || (by != nil && samePerson(by, acct)) {
			return false
		}
	}
	return true
}

// otherRunner reports whether somebody who is around, and does not own the entry, could
// be handed it to run.
func (j *judge) otherRunner(st *entryState, not *Account) bool {
	for _, q := range j.active() {
		if q.acct.ID != not.ID && !j.own(st, q.acct) && j.mayRun(st, q) {
			return true
		}
	}
	return false
}

// otherGrader reports whether somebody who is around, and does not own the entry, could
// give one of its scripts a grade that counts.
func (j *judge) otherGrader(st *entryState, sc *scriptState, not *Account) bool {
	for _, q := range j.active() {
		if q.acct.ID != not.ID && q.est && !j.own(st, q.acct) && !sc.wroteAlone(q.acct) && j.mayGrade(st, sc, q.acct) {
			return true
		}
	}
	return false
}

// sawScript reports whether a person has been handed any of an entry's scripts to grade,
// under this account or another key of the same GitHub identity.
func (j *judge) sawScript(st *entryState, acct *Account) bool {
	if st.GradeSeen[acct.ID] {
		return true
	}
	if acct.GithubID == 0 {
		return false
	}
	for id := range st.GradeSeen {
		if o := j.s.Accounts[id]; o != nil && samePerson(o, acct) {
			return true
		}
	}
	return false
}

// running reports whether a person holds the entry as an edit job now. Whoever does is
// handed none of its scripts: the next thing they report could be a copy of one.
func (j *judge) running(st *entryState, acct *Account) bool {
	for _, c := range st.LiveEdits {
		if by := j.s.Accounts[c.AccountID]; c.AccountID == acct.ID || (by != nil && samePerson(by, acct)) {
			return true
		}
	}
	return false
}

// runPending reports whether an entry may yet get another run from someone who is here:
// an edit job on it is under way, or somebody who is around could be handed one. Until
// that is over its scripts go to established accounts only, and once one has gone to
// anybody else it is over for good (Shown). A key costs nothing, and one that was shown a
// script could pass it to a runner to hand in as a run.
func (j *judge) runPending(st *entryState) bool {
	if st.runOpen == nil {
		open := len(st.LiveEdits) > 0 || j.anyone(st, nil)
		st.runOpen = &open
	}
	return *st.runOpen
}

// mayGrade reports whether an account could be handed a script at all: it is not the
// entry's owner, and has not had the script before.
func (j *judge) mayGrade(st *entryState, sc *scriptState, acct *Account) bool {
	if sc.Graded[acct.ID] || j.barred(acct, st.Round.Idx) {
		return false
	}
	// Nobody grades their own entry. The organisers' machines may grade the organisers'
	// own, but only when nobody else who is around could: see mayRun.
	if j.own(st, acct) && (!acct.Anchor || j.otherGrader(st, sc, acct)) {
		return false
	}
	if !sc.Offered[acct.ID] {
		return true
	}
	// A job is offered once. An organisers' machine gets one that came to nothing again.
	if !acct.Anchor || sc.Tries >= maxGradeAttempts {
		return false
	}
	for _, c := range sc.Live {
		if c.AccountID == acct.ID {
			return false
		}
	}
	return true
}

// someoneElse reports whether anybody who is around, other than this account, could give
// a script a grade that counts: an established account that did not write it alone.
func (j *judge) someoneElse(st *entryState, sc *scriptState, not *Account) bool {
	for _, q := range j.active() {
		if q.acct.ID != not.ID && q.est && !sc.wroteAlone(q.acct) && j.mayGrade(st, sc, q.acct) {
			return true
		}
	}
	return false
}

// offer is a job an account could be handed.
type offer struct {
	st       *entryState
	kind     string       // edit | grade
	sc       *scriptState // a grade job's script
	score    int          // lowest first
	required bool         // a score is waiting for it
}

// offers lists the jobs on an entry that an account could be handed now. With closing, only
// those a score is waiting for: the round has everything else it is going to get, and a
// job started now would be thrown away when it settles.
func (j *judge) offers(st *entryState, q *asker, closing bool) []offer {
	var out []offer
	base := st.Round.Idx * 1000
	if j.mayRun(st, q) {
		// Edits first, since they unblock everyone else, and first runs before second ones.
		score := base - 500 + 100*(len(st.Runs)+st.Dropped+len(st.LiveEdits))
		other := len(st.Runs) > 0 && q.gpuMake != ""
		for _, run := range st.Runs {
			other = other && run.Backend != "" && run.Backend != q.gpuMake
		}
		if other {
			score -= 10 // an entry run on one make of GPU goes to the other when it can
		}
		out = append(out, offer{st: st, kind: "edit", score: score, required: true})
	}
	for _, sc := range st.Scripts {
		if !j.mayGrade(st, sc, q.acct) || j.running(st, q.acct) || (!examPosted(st.Round) && !q.early) {
			continue
		}
		v := j.weigh(sc)
		under, others, newcomer := len(sc.Results)+len(sc.Live), 0, false // grades reported or under way; by accounts not yet established
		counting, any := 0, 0                                             // under way by established accounts: not its own writer's, and all
		for _, c := range sc.Live {
			acct := j.s.Accounts[c.AccountID]
			switch {
			case acct == nil:
			case !j.est[acct.ID]:
				others++
				newcomer = newcomer || acct.Agreed == 0
			default:
				any++
				if !sc.wroteAlone(acct) {
					counting++
				}
			}
		}
		for _, r := range sc.Results {
			if acct := j.s.Accounts[r.AccountID]; acct != nil && !j.est[acct.ID] {
				others++
				newcomer = newcomer || acct.Agreed == 0
			}
		}
		o := offer{st: st, kind: "grade", sc: sc, score: base + under} // oldest round, least attention
		for _, other := range st.Scripts {
			if other != sc && (other.Graded[q.acct.ID] || other.Offered[q.acct.ID]) {
				o.score += 50 // an entry's scripts go to different graders when there are any
				break
			}
		}
		switch {
		case sc.wroteAlone(q.acct):
			// Its own script. Never a volunteer's to grade; an organisers' machine's only when
			// the script has no grade, and nobody else who is around could give it one.
			if q.acct.Anchor && !v.known && any == 0 && !j.someoneElse(st, sc, q.acct) {
				o.required = true
				out = append(out, o)
			}
		case q.est:
			// A script wants one grade that counts and is not its own writer's, and a third
			// when two disagree. A script that two runs wrote wants two, from two people:
			// it is the entry's only script, and one grade of it would be one machine's
			// word for the whole score. Beyond that an established account replicates,
			// like anyone, but only before the exam: a replica taken then trains while the
			// exam is made, and one taken after it would hold the round for as long as it
			// trained.
			o.required = counting < j.short(sc)
			switch spare := j.spare(st, q); {
			case o.required && spare && j.unspared(st, sc, q.acct):
				// Somebody who cannot give the entry a run can give it this grade.
			case o.required || (!examPosted(st.Round) && !spare && under < 1+maxReplicas):
				out = append(out, o)
			}
		default:
			// At most one of those not yet established may be a newcomer (no agreed grade to
			// its name), so that fresh accounts cannot crowd the others out of the work. None
			// of them is handed a script while its entry may yet get another run from someone
			// who is here, or after handing one of the round's scripts back.
			if !closing && others < maxReplicas && (q.acct.Agreed > 0 || !newcomer) && !j.quit[q.acct.ID][st.Round.Idx] && !j.runPending(st) {
				out = append(out, o)
			}
		}
	}
	return out
}

// wanting reports whether a script's score is waiting for another grade that counts: it
// has none, or only its own writer's, or two that disagree, or one when two runs wrote it.
func (j *judge) wanting(sc *scriptState) bool { return j.short(sc) > 0 }

// short is how many more grades that count a script's score is waiting for.
func (j *judge) short(sc *scriptState) int {
	switch v := j.weigh(sc); {
	case len(sc.Writers) > 1 && len(v.counted) < 2:
		return 2 - len(v.counted)
	case !v.known || v.self || v.tie:
		return 1
	}
	return 0
}

// sitting reports whether the grades a script's score waits for are all in the hands of
// established accounts that took them before the exam was posted: trained, or nearly, and
// waiting only to sit it.
func (j *judge) sitting(st *entryState, sc *scriptState) bool {
	early := 0
	for _, c := range sc.Live {
		if !j.est[c.AccountID] || sc.wroteAlone(j.s.Accounts[c.AccountID]) {
			continue
		}
		if c.ClaimedAt >= st.Round.ExamAt {
			return false // still to train: further grades alongside it hold nothing back
		}
		early++
	}
	return early >= j.short(sc)
}

// settleable reports whether an entry could be given a score as it stands: it has a run,
// nobody is at work on another, every script its runs wrote has a grade that counts and
// no established account at work on one more, and a run that wrote no script is not one
// volunteer's word alone. A grade that counts and has no score yet is not one to settle
// on: the round's baseline, which somebody was asked for, has still to come.
func (j *judge) settleable(st *entryState) bool {
	if len(st.LiveEdits) > 0 || !examPosted(st.Round) {
		return false
	}
	for _, sc := range st.Scripts {
		if !j.weigh(sc).scored {
			return false
		}
		for _, c := range sc.Live {
			if j.est[c.AccountID] {
				return false
			}
		}
	}
	for _, run := range st.Runs {
		if run.Script == "" && !j.vouched(st, run) {
			return false
		}
	}
	return len(st.Runs) > 0
}

// closing reports whether a round is past handing out grades no score waits for: it could
// settle now, or every entry has all the grades it waits for, whatever further ones are
// still under way, or all that is left of them is sitting the exam. An established account
// is handed no such grade after the exam anyway (offers); past this, nor is anyone else,
// since the round would settle before it was done. Before the exam, a grade handed out
// has until it is posted to train.
func (j *judge) closing(states []*entryState) bool {
	if j.ready(states) {
		return true
	}
	for _, st := range states {
		if len(st.LiveEdits) > 0 || !examPosted(st.Round) || len(st.Runs) == 0 {
			return false
		}
		for _, sc := range st.Scripts {
			if j.wanting(sc) && !j.sitting(st, sc) {
				return false
			}
		}
		for _, run := range st.Runs {
			if run.Script == "" && !j.vouched(st, run) {
				return false
			}
		}
	}
	return true
}

// ready reports whether a round has everything it is going to get: every open entry could
// be scored as it stands, and nobody who is around could be handed a job that a score is
// waiting for. Someone at work on a job is around, and is asked what they could be handed
// when it is done.
func (j *judge) ready(states []*entryState) bool {
	for _, st := range states {
		if !j.settleable(st) {
			return false
		}
	}
	for _, q := range j.active() {
		for _, st := range states {
			if len(j.offers(st, q, true)) > 0 {
				return false
			}
		}
	}
	return true
}

// --- claiming -------------------------------------------------------------------------------

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
	Run       int            `json:"run,omitempty"` // a grade job: which run of the entry wrote the script
	ExpiresAt int64          `json:"expires_at"`
	// Baseline, on a grade job, says whether this machine is to measure the round's unedited
	// baseline too and send it with the grade (asksBaseline). False: it trains the script's
	// student alone, and its result carries no baseline. A client that does not know the
	// field measures it, as every grader did before there was one.
	Baseline *bool `json:"baseline,omitempty"`
}

func (a *App) jobFor(s *State, c *Claim) *Job {
	rd, ed := s.Rounds[c.RoundIdx], s.Editors[c.EditorID]
	job := &Job{
		ID: c.ID, Kind: c.Kind, Round: rd.description(a.season), Scale: rd.Scale, ExpiresAt: c.ExpiresAt,
		Editor: map[string]any{"id": ed.ID, "name": ed.Name, "version": ed.Version, "owner": s.owner(ed).Handle, "size": ed.Size},
	}
	if c.Kind == "grade" {
		asks := c.Baseline // what the job was told when it was handed out: read again, it says the same
		job.Baseline = &asks
		job.Script = &ScriptRef{SHA256: c.Script}
		for _, run := range s.entry(c.RoundIdx, c.EditorID).Runs {
			if run.Script == c.Script {
				job.Script.Size, job.Run = run.ScriptSize, run.N
				break
			}
		}
	}
	return job
}

func (a *App) checkHarness(harness, season, digest string) error {
	if season != a.season.ID || digest != a.season.Digest || versionLess(harness, a.season.Harness) {
		return fail(426, "this verifier is out of date (season %s needs harness %s or later). Upgrade: curl -fsSL %s/install.sh | sh", a.season.ID, a.season.Harness, a.cfg.BaseURL)
	}
	return nil
}

// Handed is the answer to a request for work.
type Handed struct {
	Job *Job
	// Runner says whether this account, on this machine, may be handed editors to run, and
	// if it asked to and may not, why. A machine that may not is handed grade jobs.
	Runner map[string]any
	// Note says why there is no job, when the reason is the account's own.
	Note string
	// Retry is how many seconds to wait before asking again, when there is no job and that is
	// known better than the usual minute.
	Retry int
}

// claim hands a verifier its next job, or none if there is nothing it should do right now.
func (a *App) claim(ctx context.Context, acctID int64, h *Hello) (*Handed, error) {
	if err := a.checkHarness(h.Harness, h.Season, h.SeasonDigest); err != nil {
		return nil, err
	}
	if !a.cfg.Dev && !a.season.gradesOn(deviceOf(h.Hardware)) {
		return nil, fail(400, "grading for the leaderboard needs an NVIDIA (CUDA) or AMD (ROCm) GPU, so that every grade uses the same arithmetic")
	}
	out := &Handed{}
	err := a.st.update(ctx, func(tx *Tx) error {
		s := tx.s
		acct := s.Accounts[acctID]
		if acct == nil {
			return fail(401, "no such account")
		}
		if muted(acct, now()) {
			return fail(403, "this account is given no work until %s: its grades kept differing from what established verifiers found for the same scripts. If this machine is honest, the organisers want to hear about it",
				time.Unix(acct.LastStrikeAt+muteFor, 0).UTC().Format("2 January"))
		}
		if !acct.Anchor {
			// Taking a job and neither reporting nor handing it back holds a place on that
			// entry for hours, and may hold its round. After that, a day's rest.
			for _, c := range s.Claims {
				if c.AccountID == acctID && c.Status == "lapsed" && c.ExpiresAt > now()-lapseRest {
					return fail(403, "a job this verifier took was given up on at %s: no result had come, and it had not been handed back. It is offered work again a day after that. (Ctrl-C hands a job back at once.)",
						time.Unix(c.ExpiresAt, 0).UTC().Format("15:04 UTC"))
				}
			}
		}
		canEdit, why := a.runsEditors(acct, h)
		if h.Edits {
			out.Runner = map[string]any{"eligible": canEdit, "reason": why}
		}
		hw := readHardware(h.Hardware)
		if raw := clip(string(h.Hardware), 2000); json.Valid([]byte(raw)) {
			acct.Hardware = json.RawMessage(raw)
		}
		acct.LastSeen, acct.RunnerSeen, acct.Backend = now(), 0, hw.gpuMake()
		acct.CuratorsWhy, acct.Harness = clip(a.curatorsWhy(canEdit, why, h), 300), clip(strings.ToValidUTF8(h.Harness, "?"), 40)
		if canEdit {
			acct.RunnerSeen = now()
		}
		// Said with every request, and nothing when it has not graded yet: a key moves from
		// one rented machine to the next, and the last one's GPU may have been another.
		acct.PassSeconds = 0
		if p := h.PassSeconds; finite(p) && p > 0 && p < 86400 {
			acct.PassSeconds = p
		}
		tx.touch(acct)
		if canEdit && a.established(acct) {
			a.onDemand(tx)
		}

		// A verifier that restarts picks its unfinished job back up. An editor it took to run
		// and now cannot (it asks as a machine that does not run editors) is not swapped for a
		// script: it is handed nothing until that job is reported, handed back or out of time.
		// A runner in the middle of an entry who could also ask for scripts would be shown the
		// other run's, and could hand it in as its own.
		//
		// It holds one job at a time, but for grades whose students it has trained before the
		// round's exam, which wait only to sit it once it is posted (Trained, from stet 0.4.4).
		// It is not handed those again, and while they are all it holds it may be handed
		// another grade (more). One it no longer names it lost, and is handed again to train.
		trained := map[int64]bool{}
		for _, id := range h.Trained {
			trained[id] = true
		}
		var mine *Claim
		held := 0
		for _, c := range s.Claims {
			if c.AccountID != acctID || !c.live(now()) {
				continue
			}
			if e := s.entry(c.RoundIdx, c.EditorID); e == nil || e.Status != "open" {
				continue
			}
			if c.Kind == "grade" && trained[c.ID] {
				if c.TrainedAt == 0 {
					c.TrainedAt = now()
					tx.saveClaim(c)
				}
				held++
				continue
			}
			if c.TrainedAt != 0 {
				c.TrainedAt = 0
				tx.saveClaim(c)
			}
			if mine == nil || c.ID < mine.ID {
				mine = c
			}
		}
		if mine != nil && mine.Kind == "edit" && !canEdit {
			out.Note = "This account holds a curator it took to run, and is asking now as a machine that does not run curators. It is handed nothing else until that job is reported, handed back or out of time. To finish it, start `stet verify --edits` on the machine that took it."
			return nil
		}
		if mine != nil {
			out.Job = a.jobFor(s, mine)
			return nil
		}
		var pick *offer
		best, n := math.MaxInt, 0
		j := a.newJudge(s)
		q := &asker{acct: acct, est: j.est[acctID], canEdit: canEdit, gpuMake: hw.gpuMake(), early: h.Early}
		for _, rd := range openRounds(s) {
			if held > 0 && !j.more(rd, q, held, h.PassSeconds) {
				continue
			}
			states := j.states(rd)
			closing := j.closing(states)
			early := j.tooEarly(rd, q, held, h.PassSeconds)
			var offers []offer
			for _, st := range states {
				for _, o := range j.offers(st, q, closing) {
					if (held == 0 || (o.kind == "grade" && o.required)) && !(early && o.kind == "grade") {
						offers = append(offers, o)
					}
				}
			}
			kept := j.fastFirst(rd, q, offers)
			if len(kept) < len(offers) {
				out.Retry = max(5, int(rd.ExamAt+fastFirstSeconds-j.t)) // the grades held back are anyone's then
			}
			for _, o := range kept {
				// Lowest score first; ties at random so that nobody can predict or steer
				// what they will be given.
				switch {
				case o.score < best:
					best, pick, n = o.score, &o, 1
				case o.score == best:
					if n++; rand.IntN(n) == 0 {
						pick = &o
					}
				}
			}
		}
		if pick == nil {
			for _, rd := range openRounds(s) {
				if !q.est && j.quit[acctID][rd.Idx] {
					out.Note = "This account was handed a script of the open round and handed it back, or let the job run out, so it is offered no more of that round's scripts. It is offered work again when the next round is drawn."
				}
			}
			return nil
		}
		// Far more than any GPU can do. It bounds what one account can add to the records.
		if !acct.Anchor && !a.limits.allow(fmt.Sprintf("claim:%d", acctID), 30, 3600) {
			return fail(429, "this account has been handed a great many jobs in the last hour; try again later")
		}
		e := pick.st.Entry
		c := &Claim{RoundIdx: e.RoundIdx, EditorID: e.EditorID, AccountID: acctID, Kind: pick.kind,
			ClaimedAt: now(), ExpiresAt: now() + int64(a.cfg.Lease.Seconds()), Status: "claimed"}
		if pick.sc != nil {
			c.Script = pick.sc.SHA
			c.Baseline = j.asksBaseline(pick.st.Round, acct)
		}
		tx.newClaim(c)
		if q.est {
			a.moved(tx, pick.st.Round) // the round waits its fixed while for this job, from now
		}
		out.Job = a.jobFor(s, c)
		return nil
	})
	return out, err
}

// moved notes that a round has just been moved on by someone whose word counts.
func (a *App) moved(tx *Tx, rd *Round) {
	rd.ActiveAt = now()
	tx.saveRound(rd)
}

func (a *App) release(ctx context.Context, acctID, claimID int64, reason string) error {
	return a.st.update(ctx, func(tx *Tx) error {
		if c := tx.s.Claims[claimID]; c != nil && c.AccountID == acctID && c.Status == "claimed" {
			a.handBack(tx, c, reason)
			a.settleReady(tx) // the round may have been waiting for this job alone
		}
		return nil
	})
}

// handBack ends a job its holder gives up, so that it may be handed to somebody else, and
// keeps why.
func (a *App) handBack(tx *Tx, c *Claim, reason string) {
	c.Status, c.Reason, c.EndedAt = "released", clip(strings.ToValidUTF8(reason, "?"), 300), now()
	tx.saveClaim(c)
	if rd := tx.s.Rounds[c.RoundIdx]; rd != nil && rd.Status == "open" && a.established(tx.s.Accounts[c.AccountID]) {
		a.moved(tx, rd)
	}
}

func article(kind string) string {
	if kind == "edit" {
		return "an edit"
	}
	return "a " + kind
}

// findClaim checks that a job is this account's to report on. A job ends with its lease
// and with the round: a grade that arrives after the results are public could have been
// copied from them.
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
	case !acct.Anchor && c.Status == "lapsed":
		return nil, fail(409, "this job ended before your result arrived: it was given up on at %s, with no result and without being handed back. It earns no credit, and this account is offered work again a day after that",
			time.Unix(c.ExpiresAt, 0).UTC().Format("15:04 UTC"))
	case !acct.Anchor && !c.live(now()):
		return nil, fail(409, "this job ended before your result arrived: the round settled first. It earns no credit, and it does not count against you")
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
	return a.st.blobs.Read(ctx, scriptKey(sha), a.season.maxScriptBlob())
}

type Ack struct {
	Status   string         `json:"status"` // settled | pending | edited
	Message  string         `json:"message,omitempty"`
	Verifier map[string]any `json:"verifier,omitempty"`
}

// submitScript records a run: the outcome of an edit job. With a script, the run goes on
// to be graded. Without one, the editor failed on this machine; that is a run too, and
// the entry's other run may still succeed.
func (a *App) submitScript(ctx context.Context, acctID, claimID int64, p *ScriptPayload) (*Ack, error) {
	if err := a.checkHarness(p.Harness, p.Season, p.SeasonDigest); err != nil {
		return nil, err
	}
	if !sha256Re.MatchString(p.CorpusSHA256) {
		return nil, fail(400, "the edit does not say which corpus it ran on")
	}
	// check says whether the run may be recorded: whether it was made on the round's corpus,
	// and whether whoever sends it has been handed one of the entry's scripts to grade. A run
	// on some other corpus, or from someone who has seen what another run wrote, ends the
	// job and records nothing.
	check := func(s *State) (c *Claim, other, saw bool, err error) {
		acct := s.Accounts[acctID]
		if acct == nil || !a.established(acct) {
			return nil, false, false, fail(403, "curators are run by established verifiers only")
		}
		if c, err = findClaim(s, acct, claimID, "edit"); err != nil {
			return nil, false, false, err
		}
		rd := s.Rounds[c.RoundIdx]
		if p.Round != roundID(a.season.ID, c.RoundIdx) || p.Scale != rd.Scale {
			return nil, false, false, fail(400, "this edit is for %s at %s, but the job is %s at %s", p.Round, p.Scale, roundID(a.season.ID, c.RoundIdx), rd.Scale)
		}
		e := s.entry(c.RoundIdx, c.EditorID)
		if e == nil || e.Status != "open" {
			return nil, false, false, fail(409, "this entry settled before your run arrived. It does not count against you")
		}
		for _, run := range e.Runs {
			if run.Claim == c.ID || run.By == acctID {
				return nil, false, false, fail(409, "this entry already has your run")
			}
		}
		return c, rd.CorpusSHA != "" && rd.CorpusSHA != p.CorpusSHA256, handedScript(s, acct, c.RoundIdx, c.EditorID), nil
	}
	var pre error
	other, saw := false, false
	a.st.view(func(s *State) { _, other, saw, pre = check(s) })
	if pre != nil {
		return nil, pre
	}
	p.Edit.LogTail = clip(strings.ToValidUTF8(p.Edit.LogTail, "?"), 2000)
	p.Edit.Reason = clip(strings.ToValidUTF8(p.Edit.Reason, "?"), 300)
	sha := ""
	var size int64
	if p.ScriptSHA256 != "" && !other && !saw {
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
	ack := &Ack{Status: "edited", Message: "sealed until the round settles"}
	var refused error // the job is over, and that has to be stored, so it is not the update's error
	err := a.st.update(ctx, func(tx *Tx) error {
		s := tx.s
		c, other, saw, err := check(s)
		if err != nil {
			return err
		}
		acct, rd, e := s.Accounts[acctID], s.Rounds[c.RoundIdx], s.entry(c.RoundIdx, c.EditorID)
		if saw {
			// It has been handed one of this entry's scripts to grade, so whatever it sends
			// now could be that script. The job is over, as if it had been handed back.
			c.Status, c.EndedAt = "released", now()
			tx.saveClaim(c)
			a.moved(tx, rd)
			refused = fail(409, "this account has been handed one of this entry's scripts to grade, so a run of the entry from it cannot be taken. The job has been handed back")
			a.settleReady(tx)
			return nil
		}
		if other || (sha == "" && p.ScriptSHA256 != "") {
			// The first run of a round fixes its corpus. A machine that built another has run
			// the editor on other documents, and its script would mean nothing here.
			c.Status, c.EndedAt = "refused", now()
			tx.saveClaim(c)
			a.moved(tx, rd)
			tx.event("established-differ", map[string]any{"round": p.Round, "verifier": acct.Handle, "what": "corpus"})
			if len(rd.Differs) < 50 {
				rd.Differs = append(rd.Differs, Differ{Handle: acct.Handle, What: "corpus", Est: true})
				tx.saveRound(rd)
			}
			refused = fail(409, "your corpus for %s differs from the round's. This verifier rebuilds its copy once", p.Round)
			a.settleReady(tx)
			return nil
		}
		hw := readHardware(acct.Hardware)
		edit := p.Edit
		e.Runs = append(e.Runs, &Run{N: len(e.Runs) + 1, By: acctID, GithubID: acct.GithubID, Claim: c.ID, Script: sha, ScriptSize: size, Edit: &edit,
			At: now(), GPU: hw.gpuName(), Backend: hw.gpuMake(), Owed: true})
		tx.saveEntry(e)
		if rd.CorpusSHA == "" {
			rd.CorpusSHA = p.CorpusSHA256
		}
		a.moved(tx, rd)
		c.Status, c.EndedAt = "done", now()
		tx.saveClaim(c)
		// The run is credited to its runner when the entry settles (settleEntry), as a grade
		// is: a count that moved now would say who ran what while the round is sealed.
		acct.LastSeen = now()
		tx.saveAccount(acct)
		a.settleReady(tx)
		if e.Status == "settled" {
			ack = &Ack{Status: "settled"}
		}
		return nil
	})
	if err == nil && refused != nil {
		return nil, refused
	}
	if err != nil {
		return nil, err
	}
	return ack, nil
}

// handedScript reports whether a person has been handed any script of an entry to grade,
// whatever came of the job, under this account or another key of the same GitHub identity.
func handedScript(s *State, acct *Account, round int, editor string) bool {
	is := func(id int64) bool {
		o := s.Accounts[id]
		return id == acct.ID || (o != nil && samePerson(o, acct))
	}
	for _, c := range s.Claims {
		if c.Kind == "grade" && c.RoundIdx == round && c.EditorID == editor && is(c.AccountID) {
			return true
		}
	}
	for _, r := range s.results(round, editor) {
		if is(r.AccountID) {
			return true
		}
	}
	return false
}

// editSeconds is the GPU time a run is credited with: what the runner said the edit took,
// and never more than an edit can take.
func (a *App) editSeconds(run *Run) float64 {
	if run.Edit == nil || !finite(run.Edit.Seconds) || run.Edit.Seconds < 0 {
		return 0
	}
	return math.Min(run.Edit.Seconds, 2*float64(a.season.Scales[a.cfg.Scale].EditSeconds))
}

// expireClaims ends the outstanding jobs on an entry that has settled.
func expireClaims(tx *Tx, e *Entry) {
	for _, c := range tx.s.Claims {
		if c.RoundIdx == e.RoundIdx && c.EditorID == e.EditorID && c.Status == "claimed" {
			c.Status, c.EndedAt = "expired", now()
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
// editor is not scheduled again, whether it is still qualifying or sits on the ladder; the
// verifiers' time is better spent on someone else's entry.
//
// A round in which the verifiers could not run an entry is void for it and is not held
// against it. Three of those in a row are, when the verifiers ran other entries in each:
// whatever the cause, it is this entry they cannot run. A round in which nothing ran, a
// void long ago, or a round in which nobody was there to run or check it, says nothing
// about the editor.
func failing(s *State, editorID string) bool {
	es := s.ofEditor[editorID]
	seen, invalid, void := 0, 0, 0
	unbroken := true // void rounds count only in a run, back from the latest
	for i := len(es) - 1; i >= 0; i-- {
		switch e := es[i]; {
		case e.Void:
			if unbroken && !e.Unattended && othersRan(s, e) {
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

// othersRan reports whether the verifiers ran some other entry of the same round. A round
// in which nothing ran says the verifiers were in trouble, not the entry.
func othersRan(s *State, e *Entry) bool {
	for _, o := range s.inRound[e.RoundIdx] {
		if o != e && o.Status == "settled" && !o.Void {
			return true
		}
	}
	return false
}

// roundsRun is how many rounds count for an editor: all it was drawn into, except those
// the verifiers could not run.
func roundsRun(s *State, editorID string) int {
	n := 0
	for _, e := range s.ofEditor[editorID] {
		// A round that settled with no baseline (nothing in it was graded) cannot be rated, so
		// it does not use up one of an editor's rounds either.
		if rd := s.Rounds[e.RoundIdx]; e.Void || (rd != nil && rd.Status == "settled" && rd.BaselineLoss == nil) {
			continue
		}
		n++
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

func clipJSON(raw json.RawMessage, limit int) json.RawMessage {
	if len(raw) == 0 || len(raw) > limit || !json.Valid(raw) {
		return nil
	}
	return raw
}

func scriptTokensSHA(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s struct {
		TokensSHA256 string `json:"tokens_sha256"`
	}
	if json.Unmarshal(raw, &s) != nil {
		return ""
	}
	if sha256Re.MatchString(s.TokensSHA256) {
		return s.TokensSHA256
	}
	return ""
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
	// One score for every item: how many the round has is checked against the round below.
	check := func(sc *ExamScore, what string) error {
		if sc == nil {
			return fail(400, "the result has no usable %s score", what)
		}
		return checkItems(sc, len(sc.ItemScores), what)
	}
	// A baseline is sent by a machine whose job asked for one, and may be by any that has
	// the round's on its disk. Whether this job asked is known with the job, below.
	if p.Baseline != nil {
		if err := check(p.Baseline, "baseline"); err != nil {
			return nil, err
		}
	}
	if p.Valid {
		if err := check(p.Exam, "exam"); err != nil {
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
			return fail(403, "this account is given no work for now")
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
		written := false
		for _, run := range e.Runs {
			written = written || (!run.Dropped && run.Script != "" && run.Script == c.Script)
		}
		if c.Script == "" || p.ScriptSHA256 != c.Script {
			return fail(400, "this result is for a different edit script than the one the job named")
		}
		if p.Baseline == nil && c.Baseline {
			// The job asked for the round's baseline: the round may be counting on it, and an
			// account not yet established is checked by it.
			return fail(400, "the result has no usable baseline score, and this job asked for the round's unedited baseline")
		}
		if !written {
			return fail(409, "this script was given up before your result arrived. It earns no credit, and it does not count against you")
		}
		for _, r := range s.results(c.RoundIdx, c.EditorID) {
			if r.AccountID == acctID && r.Script == c.Script {
				return fail(409, "this script already has your grade")
			}
		}
		est := a.established(acct)
		refuse := func(what string, diff float64, format string, args ...any) error {
			c.Status, c.EndedAt = "refused", now()
			tx.saveClaim(c)
			if est {
				tx.event("established-differ", map[string]any{"round": p.Round, "verifier": acct.Handle, "what": what})
				a.moved(tx, rd)
			}
			// Whoever is turned away is told it will be listed with the round, and it is,
			// established or not: if honest machines of one kind keep landing here the
			// tolerance is wrong, and this list is how anyone would see it.
			if len(rd.Differs) < 50 {
				rd.Differs = append(rd.Differs, Differ{Handle: acct.Handle, What: what, Diff: diff, Est: est})
				tx.saveRound(rd)
			}
			refused = fail(409, format, args...)
			a.settleReady(tx) // the round may have been waiting for this job alone
			return nil
		}
		// Everyone must be looking at the same corpus and exam, and whoever measures the
		// unedited baseline on them must measure the same one. The round's first run fixes
		// the corpus, and the exam is the one the organisers posted. A baseline is checked
		// against the median of the round's graders' as it arrives, once two others are in
		// (baselineDev); whoever reports before that is kept, and is checked against the
		// others each time one arrives (baselinesKept) and when the round settles.
		//
		// A grade whose job did not ask for the baseline comes without one, and there is
		// nothing of its own to check it by here. It is scored against the round's baseline
		// like every other, and what holds it to account is what holds them all: the grades
		// of a script by different people must agree (settleScript), and only an
		// established account, which earned that by agreeing, is ever not asked.
		if (rd.CorpusSHA != "" && rd.CorpusSHA != p.CorpusSHA256) || (rd.ExamSHA != "" && rd.ExamSHA != p.ExamSHA256) {
			return refuse("corpus", 0, "your corpus or exam for %s differs from the round's. This verifier rebuilds its copy once", p.Round)
		}
		initSHA := ""
		if p.Train != nil && sha256Re.MatchString(p.Train.InitSHA256) {
			initSHA = p.Train.InitSHA256
		}
		tokensSHA := scriptTokensSHA(p.Script)
		if p.Valid && len(a.season.Student) > 0 {
			if initSHA == "" {
				return fail(400, "valid grades for %s must report a 64-hex train.init_sha256", a.season.ID)
			}
			if tokensSHA == "" {
				return fail(400, "valid grades for %s must report a 64-hex script.tokens_sha256", a.season.ID)
			}
		}
		if p.Valid && initSHA != "" && rd.InitSHA != "" && rd.InitSHA != initSHA {
			return refuse("init", 0, "your initial weights hash for %s differs from the round's. Check your student model cache and harness version", p.Round)
		}
		if p.Valid && tokensSHA != "" {
			tokVotes := map[string]map[int64]bool{}
			for _, prev := range s.results(rd.Idx, e.EditorID) {
				if prev.Script == c.Script && prev.Valid && prev.TokensSHA != "" && a.established(s.Accounts[prev.AccountID]) {
					if tokVotes[prev.TokensSHA] == nil {
						tokVotes[prev.TokensSHA] = map[int64]bool{}
					}
					tokVotes[prev.TokensSHA][prev.AccountID] = true
				}
			}
			for agreedTok, voters := range tokVotes {
				if len(voters) >= 2 && agreedTok != tokensSHA {
					return refuse("tokens", 0, "your tokenized training stream for this script in %s differs from established graders'", p.Round)
				}
			}
		}
		n := len(rd.ExamTable)
		if rd.ExamSHA == "" || n == 0 {
			return fail(409, "%s has no exam yet", p.Round)
		}
		if p.Baseline != nil {
			if err := checkItems(p.Baseline, n, "baseline"); err != nil {
				return err
			}
		}
		if p.Valid {
			if err := checkItems(p.Exam, n, "exam"); err != nil {
				return err
			}
		}
		tol := a.tolerance()
		if p.Baseline != nil {
			if dev, ok := a.baselineDev(s, rd, acctID, p.Baseline.ItemScores); ok && !within(dev, tol.Baseline) {
				return refuse("baseline", places5(dev),
					"your unedited baseline for %s is further from the round's other graders' than the tolerance of %.2f bits per byte allows, so this machine's setup or arithmetic differs. That is published with the round when it settles; nothing counts against you", p.Round, tol.Baseline)
			}
		}

		// Everything is checked. From here on the change is made.
		if p.Baseline != nil {
			b := &Baseline{RoundIdx: rd.Idx, AccountID: acctID, Loss: places5(-p.Baseline.Mean), Items: p.Baseline.ItemScores,
				CorpusSHA: p.CorpusSHA256, ExamSHA: p.ExamSHA256, ReceivedAt: now(), Established: est}
			tx.saveBaseline(b)
		}
		if est {
			rd.CorpusSHA = p.CorpusSHA256
			if p.Valid && initSHA != "" && rd.InitSHA == "" {
				// Require agreement from at least two independent established accounts before
				// locking rd.InitSHA (unless the examiner already posted it authoritatively),
				// so one honest machine with a bad first result cannot poison the whole round.
				voters := map[int64]bool{acctID: true}
				for _, r := range s.Results {
					if r.RoundIdx == rd.Idx && r.Valid && r.InitSHA == initSHA && a.established(s.Accounts[r.AccountID]) {
						voters[r.AccountID] = true
					}
				}
				if len(voters) >= 2 {
					rd.InitSHA = initSHA
				}
			}
			a.moved(tx, rd)
		}
		// GPU time is what the grader says, but never more than the job was out for. It
		// counts once the grade has: when the round settles.
		seconds := math.Min(float64(p.FinishedAt-p.StartedAt), float64(now()-c.ClaimedAt))
		if !finite(seconds) || seconds < 0 {
			seconds = 0
		}
		res := &Result{ClaimID: claimID, RoundIdx: rd.Idx, EditorID: e.EditorID, AccountID: acctID, Script: c.Script, ReceivedAt: now(),
			Valid: p.Valid, CorpusSHA: p.CorpusSHA256, ExamSHA: p.ExamSHA256, InitSHA: initSHA, TokensSHA: tokensSHA, Seconds: seconds, Verdict: "pending",
			InvalidReason: clip(strings.ToValidUTF8(p.InvalidReason, "?"), 300),
			Summary:       &ResultSummary{Script: clipJSON(p.Script, 16<<10), Hardware: clipJSON(p.Hardware, 2000), Train: p.Train}}
		if p.Valid {
			res.Loss, res.Items = places5(-p.Exam.Mean), p.Exam.ItemScores
		}
		tx.newResult(res)
		a.refreshScores(tx, rd) // the round's baseline, and every grade's score against it
		c.Status, c.EndedAt = "done", now()
		tx.saveClaim(c)
		acct.LastSeen = now()
		tx.saveAccount(acct)

		a.settleReady(tx)
		ack = &Ack{Status: "pending", Message: "sealed until the round settles",
			Verifier: map[string]any{"validated": acct.Agreed, "streak": acct.Streak, "trusted": acct.Anchor, "established": a.established(acct)}}
		if e.Status == "settled" {
			ack.Status, ack.Message = "settled", ""
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

// settleReady publishes every round that is ready. It runs after anything that could make
// one ready, and on the clock, because a round also becomes ready by time passing: someone
// stops asking for work, or the wait for them runs out.
func (a *App) settleReady(tx *Tx) {
	for _, rd := range openRounds(tx.s) {
		a.settleRound(tx, rd)
	}
	a.settleRounds(tx)
}

// settleRound settles a round's open entries, all at once, if the round is ready.
func (a *App) settleRound(tx *Tx, rd *Round) {
	s, t := tx.s, now()
	wait := int64(a.cfg.ReplicaWait.Seconds())
	j := a.newJudge(s)
	states := j.states(rd)
	if !examPosted(rd) {
		return // the round waits for its exam, which is made after the draw: nothing is given up meanwhile
	}
	// What nobody is there to do is given up, once the round has waited its while.
	if t >= max(rd.ActiveAt, rd.DrawnAt)+wait {
		// An entry nobody has run waits for somebody to come, unless other entries' scores
		// are waiting behind it, or somebody who runs editors is here and it is not theirs to
		// run: then it is the round that would wait, for ever, and no later one be drawn.
		held := false
		for _, st := range states {
			held = held || j.settleable(st)
		}
		for _, q := range j.active() {
			held = held || q.canEdit
		}
		gave := false
		for _, st := range states {
			gave = a.giveUp(tx, j, st, held) || gave
		}
		if gave {
			j = a.newJudge(s)
			states = j.states(rd)
		}
	}
	if len(states) == 0 || !j.ready(states) {
		return
	}
	// The round has everything it is going to get, and no established account is at work
	// on it. It waits a while for whoever else is, but only for those who have finished a
	// job before: an account that has never reported a grade that agreed could otherwise
	// hold every round back by taking a job. The while is counted from the last thing that
	// happened in the round, or from when the job was taken if that was later: whoever is
	// handed a job is given the time to do it.
	var held []*Claim
	for _, c := range s.Claims {
		if c.RoundIdx == rd.Idx && c.live(t) && j.standing[c.AccountID] {
			if t < max(rd.ActiveAt, rd.DrawnAt, c.ClaimedAt)+wait {
				return
			}
			held = append(held, c)
		}
	}
	// Whoever the round waited for to the end, and who never reported, has let a job run
	// out as surely as one whose lease ended.
	for _, c := range held {
		c.Status, c.ExpiresAt = "lapsed", t
		tx.saveClaim(c)
	}
	for _, st := range states {
		a.settleEntry(tx, j, st)
	}
}

// anyone reports whether somebody who is around could be handed a job on an entry that
// would give it what a score needs: with a script, a grade of it that counts; without
// one, a run.
func (j *judge) anyone(st *entryState, sc *scriptState) bool {
	for _, q := range j.active() {
		if sc == nil {
			if j.mayRun(st, q) {
				return true
			}
		} else if q.est && j.mayGrade(st, sc, q.acct) && (q.acct.Anchor || !sc.wroteAlone(q.acct)) {
			return true
		}
	}
	return false
}

// vouched reports whether a run that wrote no script can be taken at its word. From an
// organisers' machine it can. From a volunteer's it cannot, alone: one account could
// otherwise say of any entry that it failed. A second person's run of the entry, whatever
// it found, makes the score something two people answer for.
func (j *judge) vouched(st *entryState, run *Run) bool {
	by := j.s.Accounts[run.By]
	return len(st.Runs) > 1 || (by != nil && by.Anchor)
}

// giveUp drops what nobody is there to check: a script that no established account has
// graded, is grading, or could be handed; a failed run with nobody to run the entry a
// second time; and, when the rest of the round is being held up by it, an entry with no
// run and nobody to give it one. An entry left with nothing is void for the round, and
// that is not held against it. Reports whether it changed anything.
func (a *App) giveUp(tx *Tx, j *judge, st *entryState, held bool) bool {
	e := st.Entry
	if len(st.Runs) == 0 {
		if held && len(st.LiveEdits) == 0 && !j.anyone(st, nil) {
			why := "no verifier was there to run this entry; the round does not count for it"
			runners := false
			for _, q := range j.active() {
				runners = runners || q.canEdit
			}
			switch {
			case st.EditFailed > 0:
				why = fmt.Sprintf("the verifiers who were there took this entry to run %s and did not finish, and nobody else was there to run it; the round does not count for it", times(st.EditFailed))
			case runners:
				why = "the verifiers who were there may not run this entry, and nobody else came to; the round does not count for it"
			}
			a.voidEntry(tx, e, why, true)
			return true
		}
		return false
	}
	gave := false
	for _, sc := range st.Scripts {
		counting := false
		for _, c := range sc.Live {
			counting = counting || j.est[c.AccountID]
		}
		if !j.weigh(sc).known && !counting && !j.anyone(st, sc) {
			why := "nobody else was there to grade it"
			for _, r := range sc.Results {
				if j.est[r.AccountID] && r.Valid && r.Score == nil {
					// Graded, by a machine that was not asked for the round's baseline; those
					// that were never sent one, and nobody was left to be asked.
					why = "nobody was there to measure the round's unedited baseline, which its grade is scored against"
				}
			}
			a.dropScript(tx, j, st, sc, why)
			gave = true
		}
	}
	for _, run := range st.Runs {
		if run.Script == "" && !j.vouched(st, run) && len(st.LiveEdits) == 0 && !j.anyone(st, nil) {
			run.Dropped, run.Reason = true, "nobody else was there to run the entry a second time"
			tx.saveEntry(e)
			gave = true
		}
	}
	if gave && !counts(e) {
		a.voidEntry(tx, e, "no second verifier was there to check this entry's run; the round does not count for it", true)
	}
	return gave
}

func times(n int) string {
	switch n {
	case 1:
		return "once"
	case 2:
		return "twice"
	}
	return fmt.Sprintf("%d times", n)
}

// counts reports whether an entry has a run that has not been dropped.
func counts(e *Entry) bool {
	for _, run := range e.Runs {
		if !run.Dropped {
			return true
		}
	}
	return false
}

// dropScript gives up every run that wrote a script nobody established could grade, or
// whose established grades the round never had a baseline to score against. What was made
// of the script is published with the runs, with nothing to compare it with.
func (a *App) dropScript(tx *Tx, j *judge, st *entryState, sc *scriptState, why string) {
	var grades []Grade
	for _, r := range sc.Results {
		g := j.view(sc, r)
		if g == nil || r.Verdict != "pending" {
			continue
		}
		shown := "no established grade"
		if g.est && !g.base {
			shown = "no baseline" // an established grade, and nothing to score it against
		}
		grades = append(grades, j.publish(g, shown, false))
		r.Verdict, r.Summary = "void", nil
		tx.saveResult(r)
	}
	for _, c := range sc.Live {
		c.Status, c.EndedAt = "expired", now() // nothing is held against whoever was at work on it
		tx.saveClaim(c)
	}
	for _, run := range st.Entry.Runs {
		if !run.Dropped && run.Script == sc.SHA {
			run.Dropped, run.Reason, run.Grades = true, why, grades
		}
	}
	tx.saveEntry(st.Entry)
	a.moved(tx, st.Round)
	tx.event("dropped", map[string]any{"round": roundID(a.season.ID, st.Round.Idx), "editor": st.Entry.EditorID, "run": sc.Run})
}

// voidEntry settles an entry as not run: the round does not count for it. unattended says
// that nobody was there to do it, which says nothing about the entry.
func (a *App) voidEntry(tx *Tx, e *Entry, why string, unattended bool) {
	for _, r := range tx.s.results(e.RoundIdx, e.EditorID) {
		if r.Verdict == "pending" {
			r.Verdict, r.Summary = "void", nil
			tx.saveResult(r)
		}
	}
	e.Void, e.Unattended, e.NRuns = true, unattended, 0
	a.settleInvalid(tx, e, why, &EntryDetail{})
	a.moved(tx, tx.s.Rounds[e.RoundIdx])
	tx.event("abandoned", map[string]any{"round": roundID(a.season.ID, e.RoundIdx), "editor": e.EditorID, "at": now()})
}

// abandonStuck gives up what the verifiers have tried and failed to do: an entry that
// maxEditAttempts edit jobs could not run, a script that maxGradeAttempts grade jobs by
// established accounts could not grade. Without this one bundle or one script that upsets
// the harness would hold its round open for ever, and no later round could be drawn.
func (a *App) abandonStuck(tx *Tx) {
	for _, rd := range openRounds(tx.s) {
		j := a.newJudge(tx.s)
		for _, st := range j.states(rd) {
			e := st.Entry
			if len(e.Runs) == 0 {
				if len(st.LiveEdits) == 0 && st.EditFailed >= maxEditAttempts {
					a.voidEntry(tx, e, fmt.Sprintf("the verifiers could not run this entry in %d attempts; the round does not count for it", maxEditAttempts), false)
				}
				continue
			}
			gave := false
			for _, sc := range st.Scripts {
				counting := false
				for _, c := range sc.Live {
					counting = counting || j.est[c.AccountID]
				}
				if !j.weigh(sc).known && !counting && sc.Tries >= maxGradeAttempts {
					a.dropScript(tx, j, st, sc, fmt.Sprintf("the verifiers could not grade this script in %d attempts", maxGradeAttempts))
					gave = true
				}
			}
			if gave && !counts(e) {
				a.voidEntry(tx, e, fmt.Sprintf("the verifiers could not grade this entry's script in %d attempts; the round does not count for it", maxGradeAttempts), false)
			}
		}
	}
}

// publish is a grade as the round's page shows it.
func (j *judge) publish(g *gradeView, verdict string, counted bool) Grade {
	row := Grade{Handle: g.acct.Handle, Verdict: verdict, Valid: g.r.Valid, Self: g.self, Counted: counted, Established: g.est,
		First: !j.standing[g.acct.ID] && !g.acct.Anchor}
	if g.r.Valid {
		loss := g.r.Loss
		row.Loss = &loss
		if g.base {
			gain := places5(g.gain)
			row.Gain = &gain
		}
	}
	return row
}

// graded is what settling a script found.
type graded struct {
	v        *verdict
	grades   []Grade        // every grade of it, as published
	best     *Result        // of the grades that decided it, the one nearest their median
	summary  *ResultSummary // and what that one said about its training
	reason   string         // why it is invalid, if it is
	outliers int
	names    [2][]string // whose grades decided it; who else agreed
}

// settleScript compares every grade of a script with what the established accounts'
// grades of it come to, and gives each grader what that earns.
func (a *App) settleScript(tx *Tx, j *judge, st *entryState, sc *scriptState) *graded {
	rd := st.Round
	v := j.weigh(sc)
	out := &graded{v: v}
	decides := map[*Result]bool{}
	for _, g := range v.counted {
		decides[g.r] = true
		if g.r.Valid && v.valid && (out.best == nil || math.Abs(g.gain-v.gain) < math.Abs(j.view(sc, out.best).gain-v.gain)-1e-12) {
			out.best, out.summary = g.r, g.r.Summary
		}
		if !g.r.Valid && !v.valid && out.reason == "" {
			out.reason = g.r.InvalidReason
		}
	}
	if !v.valid && out.reason == "" {
		out.reason = "the script is invalid"
	}
	// A script that is the unedited corpus itself scores the round's baseline, which every
	// grader not yet established reports with every grade. Agreeing about it shows nothing
	// that was not already shown, so it earns no standing. Nor does agreeing that a script
	// is invalid: that is decided before anything is trained.
	plain := !v.valid || math.Abs(v.gain) < 1e-9
	// A strike says that a grade is wrong, and one volunteer's grade against another's
	// does not show which of them is. It takes a gain that two established grades agree
	// on, or one of an organisers' machine.
	solid := !v.tie && len(v.counted) > 1
	for _, g := range v.counted {
		solid = solid || (!v.tie && g.acct.Anchor)
	}
	bestTokens := ""
	if out.best != nil {
		bestTokens = out.best.TokensSHA
	}
	ref := map[string]any{"round": roundID(a.season.ID, rd.Idx), "editor": st.Entry.EditorID}
	for _, r := range sc.Results {
		g := j.view(sc, r)
		if g == nil || r.Verdict != "pending" {
			continue
		}
		shown := "agreed"
		b := j.s.Baselines[baselineKey(r.RoundIdx, r.AccountID)]
		switch {
		case decides[r] && len(v.counted) == 1:
			// It is the script's gain, with nothing to compare it with: shown as agreed, and
			// not counted among the grades that did agree with somebody's.
			r.Verdict = "counted"
			g.acct.GPUSeconds += r.Seconds
		case decides[r] && v.tie:
			// Two established accounts, two answers, and nobody to say which is right.
			r.Verdict, shown = "disputed", "disputed"
			g.acct.GPUSeconds += r.Seconds
		case !decides[r] && (r.CorpusSHA != rd.CorpusSHA || r.ExamSHA != rd.ExamSHA ||
			(r.InitSHA != "" && rd.InitSHA != "" && r.InitSHA != rd.InitSHA) ||
			(r.TokensSHA != "" && bestTokens != "" && r.TokensSHA != bestTokens)):
			// It graded different data. A setup problem, and no evidence about this script.
			r.Verdict, shown = "void", "other data"
		case !decides[r] && b != nil && (j.oddOut(r) || !a.baselineAgrees(rd, b)):
			// Its arithmetic differs on the unedited corpus too. (A grader that was not asked
			// for the baseline, and sent none, has only its grade to be judged by.)
			r.Verdict, shown = "void", "baseline differs"
		case j.agrees(g, v):
			r.Verdict = "agreed"
			if plain || v.tie {
				r.Verdict = "same" // agreed, and counted for nothing: with a tie, agreed with neither
			} else if !g.acct.Anchor {
				g.acct.Agreed++
				g.acct.Streak++
			}
			g.acct.GPUSeconds += r.Seconds
		case !solid:
			// A real difference, and nothing to say which side of it is wrong.
			r.Verdict, shown = "unresolved", "unresolved"
			out.outliers++
		default:
			// Same script, same corpus, same code, same baseline: a real difference.
			r.Verdict, shown = "outlier", "outlier"
			out.outliers++
			if !g.acct.Anchor {
				g.acct.Streak = 0
				g.acct.Strikes++
				g.acct.LastStrikeAt = now()
			}
			tx.event("outlier", map[string]any{"round": ref["round"], "editor": ref["editor"], "verifier": g.acct.Handle})
		}
		tx.saveAccount(g.acct)
		switch {
		case decides[r]:
			out.names[0] = append(out.names[0], g.acct.Handle)
		case shown == "agreed":
			out.names[1] = append(out.names[1], g.acct.Handle)
		}
		out.grades = append(out.grades, j.publish(g, shown, decides[r]))
		r.Summary = nil // it has done its job; the entry keeps the one that counts
		tx.saveResult(r)
	}
	// The grades that decide it first, then everyone else's, each in the order they came.
	sort.SliceStable(out.grades, func(x, y int) bool { return out.grades[x].Counted && !out.grades[y].Counted })
	if v.tie {
		tx.event("disputed", ref)
	}
	return out
}

// settleEntry makes an entry's score public: the median of its runs, each run the median of
// the established grades of the script it wrote.
func (a *App) settleEntry(tx *Tx, j *judge, st *entryState) {
	e, rd, tol := st.Entry, st.Round, a.tolerance()
	if e.Status != "open" {
		return
	}
	scripts := map[string]*graded{}
	for _, sc := range st.Scripts {
		scripts[sc.SHA] = a.settleScript(tx, j, st, sc)
	}
	detail := &EntryDetail{Verifiers: []string{}, Audited: len(st.Scripts) > 0}
	var runs []*Run
	var gains []float64
	valid, self := 0, false
	for _, run := range e.Runs {
		if run.Dropped {
			continue
		}
		run.Valid, run.Gain, run.Reason = false, 0, ""
		if g := scripts[run.Script]; g == nil {
			if run.Edit != nil {
				run.Reason = run.Edit.Reason
			}
			if run.Reason == "" {
				status := "no report"
				if run.Edit != nil {
					status = run.Edit.Status
				}
				run.Reason = fmt.Sprintf("the curator produced no script (%s)", status)
			}
		} else {
			run.Grades, run.Self = g.grades, g.v.self
			self = self || g.v.self
			if g.v.valid {
				run.Valid, run.Gain = true, places5(g.v.gain)
				valid++
			} else {
				run.Reason = clip(g.reason, 300)
			}
		}
		runs = append(runs, run)
		gains = append(gains, run.Gain)
	}
	// Each run that counts is credited to whoever made it, now that it is public. (A run
	// recorded by an older server was credited when it arrived, and is not owed.)
	for _, run := range runs {
		if by := tx.s.Accounts[run.By]; by != nil && run.Owed {
			by.Runs++
			by.GPUSeconds += a.editSeconds(run)
			tx.saveAccount(by)
		}
		run.Owed = false
	}
	if len(runs) == 0 { // not reached: a round is settled only when every entry has a run
		a.voidEntry(tx, e, "no verifier ran this entry; the round does not count for it", true)
		return
	}
	// The score. With three runs of which two failed, the editor fails more often than
	// not, and its score is not a measurement.
	stand := standIns(runs)
	count := map[*Run]float64{}
	for i, run := range runs {
		if stand[run] != nil {
			gains[i] = stand[run].Gain
		}
		count[run] = gains[i]
	}
	e.Gain, e.Valid = places5(median(gains)), valid > 0 && 2*valid >= len(runs)
	if !e.Valid {
		e.Gain = 0
	}
	// The run the score is nearest to stands for the entry wherever one script is wanted.
	var rep *Run
	for _, run := range runs {
		if run.Valid == e.Valid && (rep == nil || math.Abs(run.Gain-e.Gain) < math.Abs(rep.Gain-e.Gain)-1e-12) {
			rep = run
		}
	}
	if rep == nil {
		rep = runs[0]
	}
	// Confirmed: two people's runs agree, the score is theirs, and no run rests on a grade
	// by its own runner. The organisers' machines are one person's, however many there
	// are: two of them agreeing says that the machines agree, not that two people do.
	person := func(id int64) int64 {
		if acct := tx.s.Accounts[id]; acct != nil && acct.Anchor {
			return 0
		}
		return id
	}
	sorted := append([]*Run(nil), runs...)
	sort.SliceStable(sorted, func(x, y int) bool { return count[sorted[x]] < count[sorted[y]] })
	middle := sorted[len(sorted)/2] // of three; with two, the score is between them
	pair, two := false, false
	for x, p := range runs {
		for _, q := range runs[x+1:] {
			same := p.Valid == q.Valid && (!p.Valid || within(p.Gain-q.Gain, tol.Run))
			agree := same && (len(runs) == 2 || p == middle || q == middle)
			pair = pair || agree
			two = two || (agree && person(p.By) != person(q.By))
		}
	}
	// And two people answer for it: the grades that decide its runs are not all one
	// person's. (A run that wrote no script is its runner's own word.)
	deciders := map[int64]bool{}
	for _, run := range runs {
		if g := scripts[run.Script]; g != nil {
			for _, gv := range g.v.counted {
				deciders[person(gv.r.AccountID)] = true
			}
		} else {
			deciders[person(run.By)] = true
		}
	}
	e.NRuns, e.RunsDiffer = len(runs), len(runs) > 1 && !pair
	e.Confirmed = two && !self && len(deciders) > 1
	e.OneGrader = pair && !self && len(deciders) < 2
	e.OneRunner = pair && !self && !two && len(deciders) > 1
	e.Script, e.ScriptSize, e.Edit, e.EditedAt, e.EditedBy = rep.Script, rep.ScriptSize, rep.Edit, rep.At, rep.By
	e.Loss, e.BySource, e.Reason = nil, nil, ""
	if e.Valid {
		if g := scripts[rep.Script]; g != nil && g.best != nil && g.summary != nil {
			detail.Script, detail.Hardware, detail.Train = g.summary.Script, g.summary.Hardware, g.summary.Train
		}
		// Its gain on each pillar is made the way its gain is: the mean, pillar by pillar,
		// over the runs the gain is the median of, of each run's script's. A run that failed
		// counts on every pillar as it counts in the gain: as the run that stands in for it,
		// or zero (standIns). A grade's score is a fixed weighted sum of its pillars, so these
		// add up to the entry's gain in the same proportions; one grade's pillars, which this
		// used to be, did not.
		per, by := map[*Run]map[string]float64{}, map[string]float64{}
		for _, run := range runs {
			of := run
			if stand[run] != nil {
				of = stand[run]
			}
			if g := scripts[of.Script]; of.Valid && g != nil {
				per[run] = g.v.pillars()
				for k := range per[run] {
					by[k] = 0
				}
			}
		}
		made := middleOf(sorted)
		for k := range by {
			for _, run := range made {
				by[k] += per[run][k]
			}
			by[k] = places5(by[k] / float64(len(made)))
		}
		if len(by) > 0 {
			e.BySource = by
		}
		if rd.BaselineLoss != nil {
			loss := places5(*rd.BaselineLoss - e.Gain)
			e.Loss = &loss
		}
	} else {
		for _, run := range runs {
			if !run.Valid {
				e.Reason = run.Reason
				break
			}
		}
	}
	// Who graded it: those whose grades are the score, then everyone else who agreed.
	named := map[string]bool{}
	for _, k := range []int{0, 1} {
		for _, sc := range st.Scripts {
			for _, h := range scripts[sc.SHA].names[k] {
				if !named[h] {
					named[h] = true
					detail.Verifiers = append(detail.Verifiers, h)
				}
			}
		}
	}
	for _, sc := range st.Scripts {
		detail.Disputed += scripts[sc.SHA].outliers
		detail.DisputedTogether = detail.DisputedTogether || scripts[sc.SHA].v.tie
	}
	e.Status, e.Detail, e.NResults, e.SettledAt = "settled", detail, len(detail.Verifiers), now()
	tx.saveEntry(e)
	expireClaims(tx, e)
	if !e.Valid {
		a.afterInvalid(tx, e.EditorID)
	}
}

// regain recomputes every settled entry's loss in a round from the round's current
// baseline. An entry's gain is what was measured (its grades' scores against the round's
// baseline); the loss shown beside it is the unedited student's bits per byte less that.
func (a *App) regain(tx *Tx, rd *Round) {
	for _, e := range tx.s.inRound[rd.Idx] {
		if e.Status != "settled" || !e.Valid || rd.BaselineLoss == nil {
			continue
		}
		if loss := places5(*rd.BaselineLoss - e.Gain); e.Loss == nil || *e.Loss != loss {
			e.Loss = &loss
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
