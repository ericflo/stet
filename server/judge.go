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
//   - A verifier never chooses its job and never grades its own editor. (Anchors are the
//     exception: they are trusted by definition, and grade the reference editors the
//     organisers enter.)
//   - Results are sealed: nothing about an entry's grades is visible until it settles.
//   - An anchor's grade settles an entry alone. A trusted verifier's grade settles an entry
//     alone unless the entry was secretly marked for a spot check, in which case a second,
//     agreeing grade is needed. An untrusted verifier's grade never settles anything; it is
//     compared with a trusted one, and that is how trust is earned.
//   - Two grades agree if both say the script is invalid, or both are valid with exam
//     losses within the season's tolerance (measured, and a few times the spread that
//     floating-point differences between GPUs produce).
//   - A grade that disagrees with the settled one is a strike: same script, same corpus,
//     same code, so one of the two machines is broken or lying. Three strikes and the
//     account stops being given work.

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

	"github.com/klauspost/compress/zstd"
)

const (
	trustAfterAgreements = 5
	banAfterStrikes      = 3
	maxResultsPerEntry   = 5
	retireAfterInvalid   = 2
)

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

// agree reports whether two grades of the same entry are consistent with honest runs.
func (a *App) agree(x, y *Result) bool {
	if x.CorpusSHA != y.CorpusSHA || x.ExamSHA != y.ExamSHA || x.Valid != y.Valid {
		return false
	}
	return !x.Valid || math.Abs(x.Loss-y.Loss) <= a.tolerance().Grade
}

// --- claiming -------------------------------------------------------------------------------

type entryState struct {
	Entry     *Entry
	OwnerID   int64
	Editing   bool // an anchor holds a live edit claim
	Results   int
	Trusted   int // trusted results and live claims
	Untrusted int
	Claims    int
	HasAnchor bool // an anchor grade exists or an anchor is grading it
	Mine      bool // this verifier already has a grade or a live claim
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
	tally := func(st *entryState, accountID int64, isResult bool) {
		acct := s.Accounts[accountID]
		if acct == nil {
			return
		}
		if isResult {
			st.Results++
		} else {
			st.Claims++
		}
		switch {
		case acct.Anchor:
			st.HasAnchor = true
			st.Trusted++
		case acct.Trusted:
			st.Trusted++
		default:
			st.Untrusted++
		}
		if accountID == acctID {
			st.Mine = true
		}
	}
	for k, st := range index {
		for _, r := range s.resultsOf[k] {
			if r.AccountID == acctID {
				st.Mine = true // one grade per verifier per entry, whatever became of it
			}
			if r.Verdict != "void" {
				tally(st, r.AccountID, true)
			}
		}
	}
	for _, c := range s.Claims {
		st := index[entryKey(c.RoundIdx, c.EditorID)]
		if st == nil || !c.live(t) {
			continue
		}
		if c.Kind == "edit" {
			st.Editing = true
			continue
		}
		tally(st, c.AccountID, false)
	}
	return states
}

// wantsGrade decides whether this verifier should grade this entry now.
func wantsGrade(st *entryState, acct *Account) bool {
	if st.Entry.Script == "" || st.Mine || st.HasAnchor {
		return false
	}
	total := st.Results + st.Claims
	if acct.Anchor {
		return true // including the reference editors the organisers enter themselves
	}
	if st.OwnerID == acct.ID {
		return false
	}
	if !acct.Trusted {
		return st.Untrusted == 0 && total < 3 // one apprentice per entry, checked against a trusted grade
	}
	need := 1
	if st.Entry.Spot {
		need = 2
	}
	if st.Claims == 0 && st.Results >= need && st.Trusted >= 1 {
		// Enough grades are in and it has not settled: they disagree. Ask for another opinion.
		return st.Results < maxResultsPerEntry-1
	}
	return st.Trusted < 1 || total < need
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
		return fail(426, "this verifier is out of date (season %s, harness %s is current). Upgrade: curl -fsSL %s/install.sh | sh", a.season.ID, a.season.Harness, a.cfg.BaseURL)
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
		if acct.Banned {
			return fail(403, "this account can no longer grade: its results repeatedly disagreed with everyone else's")
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
		for _, st := range openEntryStates(s, acctID) {
			if canEdit && st.Entry.EditedAt == 0 && !st.Editing {
				consider(st, "edit", st.Entry.RoundIdx*1000-500) // edits first: they unblock everyone else
			}
			if wantsGrade(st, acct) {
				consider(st, "grade", st.Entry.RoundIdx*1000+st.Results+st.Claims) // oldest round, least attention
			}
		}
		if pick == nil {
			return nil
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

// findClaim checks that a job is this account's to report on.
func findClaim(s *State, acctID, claimID int64, kind string) (*Claim, error) {
	c := s.Claims[claimID]
	if c == nil || c.AccountID != acctID {
		return nil, fail(404, "no such job for this account (it may have been finished long ago)")
	}
	if c.Kind != kind {
		return nil, fail(400, "this is %s job, not %s job", article(c.Kind), article(kind))
	}
	if c.Status == "done" || c.Status == "released" {
		return nil, fail(409, "this job was already %s", c.Status)
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
		c, err := findClaim(s, acctID, claimID, "edit")
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
	if !finite(seconds) || seconds < 0 || seconds > 86400 {
		seconds = 0
	}
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
			a.settleInvalid(tx, e, reason, &EntryDetail{Verifiers: []string{acct.Handle}, Audited: true})
			a.settleRounds(tx)
			ack = &Ack{Status: "settled", Message: "no script: the entry scores as the unedited corpus"}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return ack, nil
}

// expireClaims ends the outstanding jobs on an entry that no longer needs them.
func expireClaims(tx *Tx, e *Entry, keepAnchors bool) {
	for _, c := range tx.s.Claims {
		if c.RoundIdx != e.RoundIdx || c.EditorID != e.EditorID || c.Status != "claimed" {
			continue
		}
		if acct := tx.s.Accounts[c.AccountID]; keepAnchors && acct != nil && acct.Anchor {
			continue // an anchor's grade can still overrule the settlement
		}
		c.Status = "expired"
		tx.saveClaim(c)
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
	expireClaims(tx, e, false)
	a.afterInvalid(tx, e.EditorID)
}

// afterInvalid stops scheduling an editor that keeps failing: its rounds are better spent
// on someone else's entry.
func (a *App) afterInvalid(tx *Tx, editorID string) {
	n := 0
	for _, e := range tx.s.ofEditor[editorID] {
		if e.Status == "settled" && !e.Valid {
			n++
		}
	}
	if ed := tx.s.Editors[editorID]; n >= retireAfterInvalid && (ed.Status == "queued" || ed.Status == "running") {
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
	if p.Train != nil && len(p.Train.Curve) > 200 {
		p.Train.Curve = nil
	}
	seconds := float64(p.FinishedAt - p.StartedAt)
	if seconds < 0 || seconds > 86400 {
		seconds = 0
	}
	var ack *Ack
	err := a.st.update(ctx, func(tx *Tx) error {
		s := tx.s
		acct := s.Accounts[acctID]
		if acct == nil {
			return fail(401, "no such account")
		}
		c, err := findClaim(s, acctID, claimID, "grade")
		if err != nil {
			return err
		}
		rd, e := s.Rounds[c.RoundIdx], s.entry(c.RoundIdx, c.EditorID)
		if p.Round != roundID(a.season.ID, c.RoundIdx) || p.Scale != rd.Scale {
			return fail(400, "this result is for %s at %s, but the job is %s at %s", p.Round, p.Scale, roundID(a.season.ID, c.RoundIdx), rd.Scale)
		}
		if e.Script == "" || p.ScriptSHA256 != e.Script {
			return fail(400, "this result is for a different edit script than the one the job named")
		}
		for _, r := range s.results(c.RoundIdx, c.EditorID) {
			if r.AccountID == acctID {
				return fail(409, "you have already reported a result for this entry")
			}
		}
		// Everyone must be looking at the same corpus and exam. The first reliable verifier to
		// report fixes them for the round; anchors can overrule.
		if !acct.Anchor && ((rd.CorpusSHA != "" && rd.CorpusSHA != p.CorpusSHA256) || (rd.ExamSHA != "" && rd.ExamSHA != p.ExamSHA256)) {
			return fail(409, "your corpus or exam for %s differs from the round's. Delete ~/.cache/stet/rounds/%s and the pool cache, then try again", p.Round, p.Round)
		}
		tol := a.tolerance()
		if rd.BaselineLoss != nil && !acct.Anchor && math.Abs(*rd.BaselineLoss-p.Baseline.Loss) > tol.Baseline {
			return fail(409, "your baseline for %s is %.4f; the round's is %.4f. That is further apart than floating-point noise allows, so something about this machine's setup differs", p.Round, p.Baseline.Loss, *rd.BaselineLoss)
		}

		// Everything is checked. From here on the change is made.
		wasReliable := acct.reliable()
		tx.saveBaseline(&Baseline{RoundIdx: rd.Idx, AccountID: acctID, Loss: p.Baseline.Loss, BySource: p.Baseline.BySource,
			CorpusSHA: p.CorpusSHA256, ExamSHA: p.ExamSHA256, ReceivedAt: now()})
		if wasReliable {
			if rd.CorpusSHA == "" || rd.ExamSHA == "" || acct.Anchor {
				rd.CorpusSHA, rd.ExamSHA = p.CorpusSHA256, p.ExamSHA256
				tx.saveRound(rd)
			}
			a.refreshBaseline(tx, rd)
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
		acct.GPUSeconds += seconds
		acct.LastSeen = now()
		tx.saveAccount(acct)

		settled := a.settleEntry(tx, e)
		a.settleRounds(tx)
		ack = &Ack{Status: "pending", Verifier: map[string]any{"validated": acct.Agreed, "streak": acct.Streak, "trusted": acct.reliable()}}
		switch {
		case settled:
			ack.Status = "settled"
		case !wasReliable:
			ack.Message = "held until a trusted verifier grades the same script; agreeing with them is how trust is earned"
		default:
			ack.Message = "waiting for a second opinion"
		}
		return nil
	})
	return ack, err
}

// refreshBaseline sets the round's baseline to the median of what reliable verifiers measured.
func (a *App) refreshBaseline(tx *Tx, rd *Round) {
	var losses []float64
	per := map[string][]float64{}
	for _, b := range tx.s.Baselines {
		acct := tx.s.Accounts[b.AccountID]
		if b.RoundIdx != rd.Idx || acct == nil || !acct.reliable() || b.CorpusSHA != rd.CorpusSHA || b.ExamSHA != rd.ExamSHA {
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
	rd.BaselineLoss, rd.Baseline, rd.BaselineN = &m, by, len(losses)
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

// decide picks the grades that define an entry's score, or nil if it cannot be settled yet.
func (a *App) decide(s *State, results []*Result, spot bool) []*Result {
	var anchors, trusted []*Result
	for _, r := range results {
		acct := s.Accounts[r.AccountID]
		if r.Verdict == "void" || acct == nil {
			continue
		}
		if acct.Anchor {
			anchors = append(anchors, r)
		} else if acct.Trusted {
			trusted = append(trusted, r)
		}
	}
	cluster := func(center *Result) []*Result {
		out := []*Result{center}
		for _, r := range results {
			if r != center && r.Verdict != "void" && a.agree(center, r) {
				out = append(out, r)
			}
		}
		return out
	}
	if len(anchors) > 0 {
		return cluster(anchors[0])
	}
	for _, t := range trusted {
		c := cluster(t)
		if !spot || len(c) >= 2 {
			return c
		}
	}
	return nil
}

// settleEntry settles an entry if its grades allow, and hands out verdicts. It is also
// called for late grades on an entry that has already settled (an anchor's audit can
// overrule an earlier settlement).
func (a *App) settleEntry(tx *Tx, e *Entry) bool {
	s := tx.s
	results := s.results(e.RoundIdx, e.EditorID)
	final := a.decide(s, results, e.Spot)
	if final == nil {
		return false
	}
	in := map[int64]bool{}
	audited := false
	for _, r := range final {
		in[r.ID] = true
		audited = audited || s.Accounts[r.AccountID].Anchor
	}
	rep := final[0]
	for _, r := range results {
		if r.Verdict != "pending" {
			continue
		}
		acct := s.Accounts[r.AccountID]
		switch {
		case in[r.ID]:
			r.Verdict = "agreed"
			// Agreement only builds trust when someone else reliable was part of it.
			vouched := acct.reliable()
			for _, o := range final {
				if o.AccountID != r.AccountID && s.Accounts[o.AccountID].reliable() {
					vouched = true
				}
			}
			if vouched {
				acct.Agreed++
				acct.Streak++
				if !acct.Banned && acct.Strikes == 0 && acct.Agreed >= trustAfterAgreements {
					acct.Trusted = true
				}
				tx.saveAccount(acct)
			}
		case r.CorpusSHA != rep.CorpusSHA || r.ExamSHA != rep.ExamSHA:
			r.Verdict = "void" // looked at different data: useless, but a setup problem rather than a lie
		default:
			r.Verdict = "outlier" // same script, same corpus, same code: a real difference is a fault
			if !acct.Anchor {
				acct.Streak, acct.Trusted = 0, false
				acct.Strikes++
				acct.Banned = acct.Banned || acct.Strikes >= banAfterStrikes
				tx.saveAccount(acct)
			}
			tx.event("outlier", map[string]any{"round": roundID(a.season.ID, e.RoundIdx), "editor": e.EditorID, "verifier": acct.Handle})
		}
		tx.saveResult(r)
	}

	detail := &EntryDetail{}
	if e.Detail != nil {
		*detail = *e.Detail
	}
	detail.Audited, detail.Verifiers = audited, nil
	for _, r := range final {
		detail.Verifiers = append(detail.Verifiers, s.Accounts[r.AccountID].Handle)
	}
	if rep.Summary != nil {
		detail.Script, detail.Hardware, detail.Train = rep.Summary.Script, rep.Summary.Hardware, rep.Summary.Train
	}
	e.Status, e.Valid, e.Detail, e.NResults = "settled", rep.Valid, detail, len(final)
	e.Loss, e.BySource, e.Reason = nil, nil, ""
	if rep.Valid {
		sum := 0.0
		by := map[string]float64{}
		for _, r := range final {
			sum += r.Loss
			for k, v := range r.BySource {
				by[k] += v / float64(len(final))
			}
		}
		loss := sum / float64(len(final))
		e.Loss, e.BySource = &loss, by
	} else {
		e.Reason = rep.InvalidReason
	}
	if e.SettledAt == 0 {
		e.SettledAt = now()
	}
	tx.saveEntry(e)
	// The summaries have done their job; the entry keeps the one that counts.
	for _, r := range results {
		if r.Summary != nil {
			r.Summary = nil
			tx.saveResult(r)
		}
	}
	expireClaims(tx, e, true)
	a.regain(tx, s.Rounds[e.RoundIdx])
	if !e.Valid {
		a.afterInvalid(tx, e.EditorID)
	}
	return true
}

// regain recomputes every settled entry's gain in a round from the round's current baseline.
func (a *App) regain(tx *Tx, rd *Round) {
	for _, e := range tx.s.inRound[rd.Idx] {
		if e.Status != "settled" {
			continue
		}
		gain := 0.0
		if e.Valid && e.Loss != nil && rd.BaselineLoss != nil {
			gain = *rd.BaselineLoss - *e.Loss
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
