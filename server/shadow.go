package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"regexp"
	"sort"
)

// Shadow is what the next exam says about one settled round: season 0's shadow score. It is
// measured by the organisers' machine (stet shadow --upload), shown beside the season's
// scores, and ranks nothing. See docs/research/exam-proposal.md.
type Shadow struct {
	RoundIdx  int                `json:"round"`
	At        int64              `json:"at"`
	By        string             `json:"by"`
	Items     int                `json:"items"`
	Corners   int                `json:"corners"`
	Generated int                `json:"generated"`
	Fresh     int                `json:"fresh"`
	Digest    string             `json:"items_digest"`
	Version   int                `json:"generator_version"`
	Languages map[string]float64 `json:"languages,omitempty"`
	Entries   []ShadowEntry      `json:"entries"`
}

// ShadowEntry is one entry's score on the next exam: its paired gain over the unedited corpus,
// in bits per byte, and the parts it is made of.
type ShadowEntry struct {
	EditorID  string             `json:"editor"`
	Score     float64            `json:"score"`
	Generated *float64           `json:"generated,omitempty"`
	Fresh     *float64           `json:"fresh,omitempty"`
	Families  map[string]float64 `json:"families,omitempty"`
}

func shadowKey(idx int) string { return fmt.Sprintf("shadow/%08d", idx) }

func (tx *Tx) saveShadow(sh *Shadow) {
	tx.s.Shadows[sh.RoundIdx] = sh
	tx.put(shadowKey(sh.RoundIdx), sh)
}

var familyRe = regexp.MustCompile(`^[a-z0-9-]{1,32}$`)

// bounded is a number a shadow may hold: finite, and not absurd.
func bounded(v float64) bool { return finite(v) && math.Abs(v) < 1e6 }

// check says what is wrong with a shadow a client sent for a round, if anything.
func (sh *Shadow) check(s *State) error {
	rd := s.Rounds[sh.RoundIdx]
	if rd == nil || rd.Status != "settled" {
		return fail(409, "round %d has not settled", sh.RoundIdx)
	}
	if sh.Items <= 0 || sh.Items > 100_000 || sh.Corners <= 0 || sh.Corners > sh.Items || sh.Generated < 0 || sh.Fresh < 0 ||
		sh.Generated+sh.Fresh != sh.Items || !sha256Re.MatchString(sh.Digest) || len(sh.Entries) == 0 || len(sh.Entries) > 500 ||
		len(sh.Languages) > 400 {
		return fail(400, "the shadow does not add up")
	}
	for lang, v := range sh.Languages {
		if !familyRe.MatchString(lang) || !bounded(v) {
			return fail(400, "a language share is not one")
		}
	}
	seen := map[string]bool{}
	for _, e := range sh.Entries {
		if seen[e.EditorID] || s.entry(sh.RoundIdx, e.EditorID) == nil {
			return fail(400, "%.12s is not an entry of round %d, or is given twice", e.EditorID, sh.RoundIdx)
		}
		seen[e.EditorID] = true
		if !bounded(e.Score) || (e.Generated != nil && !bounded(*e.Generated)) || (e.Fresh != nil && !bounded(*e.Fresh)) || len(e.Families) > 64 {
			return fail(400, "a score is not a number")
		}
		for f, v := range e.Families {
			if !familyRe.MatchString(f) || !bounded(v) {
				return fail(400, "a family's score is not one")
			}
		}
	}
	return nil
}

// apiAdminShadow keeps the next exam's verdict on a settled round. Organisers only: it is their
// machine that measures it, while it ranks nothing.
func (a *App) apiAdminShadow(w http.ResponseWriter, r *http.Request) {
	acct, _, body, err := a.signed(w, r, 1<<20, true)
	if err != nil {
		writeErr(w, err)
		return
	}
	if !acct.Anchor {
		writeErr(w, fail(403, "organisers only"))
		return
	}
	var sh Shadow
	if err := json.Unmarshal(body, &sh); err != nil {
		writeErr(w, fail(400, "not a shadow: %v", err))
		return
	}
	sh.At, sh.By = now(), acct.Handle
	err = a.st.update(r.Context(), func(tx *Tx) error {
		if err := sh.check(tx.s); err != nil {
			return err
		}
		sort.SliceStable(sh.Entries, func(i, j int) bool { return sh.Entries[i].Score > sh.Entries[j].Score })
		tx.saveShadow(&sh)
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, 200, map[string]any{"round": roundID(a.season.ID, sh.RoundIdx), "entries": len(sh.Entries)})
}

// shadowStanding is one curator's average on the next exam, over the rounds it was measured in.
type shadowStanding struct {
	EditorID string
	Name     string
	Version  int
	Owner    string
	Mean     float64
	Rounds   int
	Families map[string]float64
	Season   *BoardRow // its row on the season's board, if it has one
}

// shadowStandings averages the next exam's scores per curator, best first.
func (a *App) shadowStandings(s *State, b *Board) []shadowStanding {
	sum := map[string]float64{}
	n := map[string]int{}
	fam := map[string]map[string][2]float64{}
	for _, sh := range s.Shadows {
		for _, e := range sh.Entries {
			sum[e.EditorID] += e.Score
			n[e.EditorID]++
			if fam[e.EditorID] == nil {
				fam[e.EditorID] = map[string][2]float64{}
			}
			for f, v := range e.Families {
				x := fam[e.EditorID][f]
				fam[e.EditorID][f] = [2]float64{x[0] + v, x[1] + 1}
			}
		}
	}
	out := []shadowStanding{}
	for id, total := range sum {
		ed := s.Editors[id]
		if ed == nil {
			continue
		}
		row := shadowStanding{EditorID: id, Name: ed.Name, Version: ed.Version, Owner: s.owner(ed).Handle, Mean: total / float64(n[id]),
			Rounds: n[id], Families: map[string]float64{}, Season: b.ByID[id]}
		for f, x := range fam[id] {
			row.Families[f] = x[0] / x[1]
		}
		out = append(out, row)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Mean != out[j].Mean {
			return out[i].Mean > out[j].Mean
		}
		return out[i].EditorID < out[j].EditorID
	})
	return out
}
