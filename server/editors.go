package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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
func findEditor(s *State, ref string) *Editor {
	ref = strings.ToLower(strings.TrimSpace(ref))
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
			if e.Name != name || (want > 0 && e.Version != want) {
				continue
			}
			if best == nil || e.Version > best.Version {
				best = e
			}
		}
		return best
	case sha256Re.MatchString(ref):
		return s.Editors[ref]
	case len(ref) >= 8 && len(ref) < 64 && isHex(ref):
		var best *Editor
		for id, e := range s.Editors {
			if strings.HasPrefix(id, ref) && (best == nil || e.CreatedAt > best.CreatedAt) {
				best = e
			}
		}
		return best
	}
	return nil
}

const maxSubmissionsPerDay = 6

type Submission struct {
	Editor   *Editor
	Replaced string
	Round    *Round
	Message  string
}

// mayEnter says whether an account may submit, and if not, what it should do about it.
// Running an entry costs an anchor's time, which is the scarcest thing the competition
// has, so entries need an identity that is not free to make.
func (a *App) mayEnter(acct *Account) error {
	if acct.Banned {
		return fail(403, "this account is suspended")
	}
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
	m, files, err := parseBundle(blob)
	if err != nil {
		return nil, fail(400, "%s", err.Error())
	}
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
			return fail(409, "these exact files have already been submitted by %s. Fork it and make it your own: stet fork %s", s.owner(existing).Handle, id[:12])
		}
		recent := 0
		for _, e := range s.ownedBy[acctID] {
			if e.CreatedAt > now()-86400 {
				recent++
			}
		}
		if recent >= maxSubmissionsPerDay && !acct.Anchor {
			return fail(429, "that is %d entries in a day. Grading is done by volunteers; use `stet run` to iterate and come back tomorrow", recent)
		}
		return nil
	}
	var pre error
	a.st.view(func(s *State) { pre = check(s) })
	if pre != nil {
		return nil, pre
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
		if !acct.Anchor { // anchors seed the board with reference editors and may run several at once
			for _, e := range s.ownedBy[acctID] {
				if e.Status == "queued" || e.Status == "running" {
					// Open entries of a replaced editor still settle: verifiers may already be at work on them.
					e.Status = "withdrawn"
					tx.saveEditor(e)
					sub.Replaced = e.ID
				}
			}
		}
		ed := &Editor{
			ID: id, AccountID: acctID, Name: m.Name, Version: version, Description: m.Description, Technique: m.Technique,
			Parent: parent, Entry: m.Entry, Files: files, Size: int64(len(blob)), CreatedAt: now(), Status: "queued",
		}
		tx.saveEditor(ed)
		tx.event("submit", map[string]any{"editor": id, "name": m.Name, "version": version, "owner": acct.Handle})
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
