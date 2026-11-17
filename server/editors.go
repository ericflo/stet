package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"sort"
	"strconv"
	"strings"
)

func isHex(s string) bool {
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return s != ""
}

// findEditor resolves an id, an id prefix (8+ hex characters) or "handle/name[@version]"
// (the latest version unless one is named).
//
// Only editors the viewer may see are considered (public ones, and the viewer's own; pass 0
// for a stranger), so a submission that has not run cannot hide a public editor by taking
// its name or the start of its id. A short id means the oldest editor it fits: ids are
// hashes, and someone who grinds out a later one with the same first characters gets
// nothing for it.
func findEditor(s *State, ref string, viewer int64) *Editor {
	ref = strings.ToLower(strings.TrimSpace(ref))
	visible := func(e *Editor) bool { return e != nil && (e.Public || (viewer != 0 && e.AccountID == viewer)) }
	switch {
	case strings.Contains(ref, "/"):
		handle, name, _ := strings.Cut(ref, "/")
		name, ver, _ := strings.Cut(name, "@")
		want, _ := strconv.Atoi(strings.TrimPrefix(ver, "v"))
		acct := s.byHandle[handle]
		if acct == nil {
			return nil
		}
		var best *Editor
		for _, e := range s.ownedBy[acct.ID] {
			if e.Name != name || (want > 0 && e.Version != want) || !visible(e) {
				continue
			}
			if best == nil || e.Version > best.Version {
				best = e
			}
		}
		return best
	case sha256Re.MatchString(ref):
		if e := s.Editors[ref]; visible(e) {
			return e
		}
	case len(ref) >= 8 && len(ref) < 64 && isHex(ref):
		var best *Editor
		for id, e := range s.Editors {
			if strings.HasPrefix(id, ref) && visible(e) && (best == nil || e.CreatedAt < best.CreatedAt || (e.CreatedAt == best.CreatedAt && e.ID < best.ID)) {
				best = e
			}
		}
		return best
	}
	return nil
}

type Submission struct {
	Editor   *Editor
	Replaced string
	InPlay   int // how many entries this person now has waiting or qualifying
	Round    *Round
	Message  string
}

// mayEnter says whether an account may submit, and if not, what it should do about it.
// Running an entry costs volunteers' GPU time, which is the scarcest thing the competition
// has, so entries need an identity that is not free to make. The same test stands between
// an account and being established: whoever is handed entrants' code, and whose grades
// move scores, is one person too.
func (a *App) mayEnter(acct *Account) error {
	if acct.Anchor || !a.cfg.RequireGithub {
		return nil
	}
	if acct.Github == "" {
		return fail(403, "entering needs a linked GitHub account, so that one person is one entrant: stet login --github")
	}
	if age := now() - acct.GithubCreatedAt; a.cfg.GithubMinAgeDays > 0 && age < int64(a.cfg.GithubMinAgeDays)*86400 {
		return fail(403, "the linked GitHub account is younger than %d days; entries are open to accounts older than that", a.cfg.GithubMinAgeDays)
	}
	return nil
}

// submit stores a new entry. An account has one entry in play at a time: submitting again
// replaces whatever of theirs is still waiting or qualifying. Editors that have finished
// qualifying keep their results and their place on the ladder.
func (a *App) submit(ctx context.Context, acctID int64, blob []byte) (*Submission, error) {
	sum := sha256.Sum256(blob)
	id := hex.EncodeToString(sum[:])

	// Checked before the upload, and again inside the change that records it.
	check := func(s *State) error {
		acct := s.Accounts[acctID]
		if acct == nil {
			return fail(401, "no such account")
		}
		if err := a.mayEnter(acct); err != nil {
			return err
		}
		if existing := s.Editors[id]; existing != nil {
			if existing.AccountID == acctID {
				return fail(409, "you have already submitted exactly these files (%s v%d). Change something first", existing.Name, existing.Version)
			}
			if !existing.Public { // not drawn yet: its code is nobody's to fork
				return fail(409, "these exact files have already been submitted by someone else. Change something first")
			}
			return fail(409, "these exact files have already been submitted by %s. Fork it and make it your own: stet fork %s", s.owner(existing).Handle, id[:12])
		}
		// Counted per person, not per key: a GitHub identity that moves to a new account
		// takes its day's submissions with it.
		recent := 0
		for _, e := range s.Editors {
			if e.CreatedAt > now()-86400 && (e.AccountID == acctID || (acct.GithubID != 0 && e.GithubID == acct.GithubID)) {
				recent++
			}
		}
		if recent >= a.cfg.SubmitsPerDay && !acct.Anchor {
			return fail(429, "that is %d entries in a day, which is the most one person may submit. Each is run and graded on volunteers' machines for at least %d rounds. `stet run` has no limit and uses the same harness: try things there, and submit again tomorrow", recent, a.cfg.QualifyRounds)
		}
		return nil
	}
	var pre error
	a.st.view(func(s *State) { pre = check(s) })
	if pre != nil {
		return nil, pre
	}
	// Only now is the bundle unpacked: that costs memory, and the account has earned it.
	m, files, err := parseBundle(blob, a.season.Assets)
	if err != nil {
		return nil, fail(400, "%s", err.Error())
	}
	// Volunteers will download what the entry names. The host is asked now whether each
	// file is there for anyone, and is the file the entry says it is.
	if err := checkAssets(ctx, m.Files); errors.Is(err, errHubAway) {
		return nil, fail(503, "the entry names files on Hugging Face, and Hugging Face could not be asked about them just now. Nothing was submitted; try again in a minute")
	} else if err != nil {
		return nil, fail(400, "%s", err.Error())
	}
	var assetBytes int64
	for _, as := range m.Files {
		assetBytes += as.Size
	}
	// The files go first. If the record that follows is never written, an unreferenced
	// file is all that is left behind.
	if err := a.st.blobs.Put(ctx, bundleKey(id), blob, "application/zstd"); err != nil {
		return nil, err
	}
	sub := &Submission{}
	err = a.st.update(ctx, func(tx *Tx) error {
		s := tx.s
		if err := check(s); err != nil {
			return err
		}
		acct := s.Accounts[acctID]
		parent := m.Parent
		if parent != "" && s.Editors[parent] == nil {
			parent = "" // a parent we have never seen is not lineage
		}
		version := 1
		for _, e := range s.ownedBy[acctID] {
			if e.Name == m.Name && e.Version >= version {
				version = e.Version + 1
			}
		}
		if !acct.Anchor { // the organisers seed the board with reference editors, as many as it takes
			// One person may have a few entries waiting or qualifying at once: ideas to compare
			// on the same rounds. A new version of an entry takes the old version's place, and
			// past the limit the oldest makes way. Open entries of an editor that leaves still
			// settle: verifiers may already be at work on them.
			var mine []*Editor
			for _, e := range s.Editors {
				if (e.Status == "queued" || e.Status == "running") && (e.AccountID == acctID || (acct.GithubID != 0 && e.GithubID == acct.GithubID)) {
					mine = append(mine, e)
				}
			}
			sort.Slice(mine, func(i, j int) bool {
				return mine[i].CreatedAt < mine[j].CreatedAt || (mine[i].CreatedAt == mine[j].CreatedAt && mine[i].ID < mine[j].ID)
			})
			leave := func(e *Editor) {
				e.Status = "withdrawn"
				tx.saveEditor(e)
				sub.Replaced = e.ID
			}
			var others []*Editor
			for _, e := range mine {
				if e.Name == m.Name {
					leave(e)
				} else {
					others = append(others, e)
				}
			}
			for len(others) >= a.cfg.EntriesInPlay {
				leave(others[0])
				others = others[1:]
			}
			sub.InPlay = len(others) + 1
		}
		ed := &Editor{
			ID: id, AccountID: acctID, Name: m.Name, Version: version, Description: m.Description, Technique: m.Technique,
			Parent: parent, Entry: m.Entry, Files: files, Size: int64(len(blob)), CreatedAt: now(), Status: "queued",
			GithubID: acct.GithubID, Assets: m.Files, AssetBytes: assetBytes,
		}
		tx.saveEditor(ed)
		tx.event("submit", map[string]any{"editor": id, "name": m.Name, "version": version, "owner": acct.Handle})
		a.onDemand(tx) // somebody is waiting now: the reply should name the draw they will be in
		sub.Editor, sub.Round = ed, s.roundWithStatus("scheduled")
		ahead := 0
		for _, e := range s.Editors {
			if e.Status == "queued" && e.CreatedAt < ed.CreatedAt {
				ahead++
			}
		}
		if ahead >= a.cfg.MaxNewPerRound {
			sub.Message = "The queue is longer than one round can take, so it may wait a round or two. Entries from people who grade for others go first: `stet verify`."
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return sub, nil
}

func (a *App) readBundle(ctx context.Context, id string) ([]byte, error) {
	if !sha256Re.MatchString(id) {
		return nil, errNoBlob
	}
	return a.st.blobs.Read(ctx, bundleKey(id), maxBundleBytes)
}

// editorsOf lists an account's editors, newest first.
func editorsOf(s *State, acctID int64, limit int) []*Editor {
	eds := append([]*Editor(nil), s.ownedBy[acctID]...)
	sort.SliceStable(eds, func(i, j int) bool { return eds[i].CreatedAt > eds[j].CreatedAt })
	if limit > 0 && len(eds) > limit {
		eds = eds[:limit]
	}
	return eds
}
