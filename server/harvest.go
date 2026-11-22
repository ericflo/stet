package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
)

// --- the harvests a round's exam is made from --------------------------------------------------
//
// Three quarters of an exam is writing people published. In season 1 it was the hour after the
// draw, and every round waited two hours for its exam. Now the examiner harvests every hour of
// the clock as it settles, and tells the server each harvest's hash at once (a Harvest). At a
// draw the server gives the round a few of the day's hours, chosen by the round's seed, and the
// examiner makes the exam from exactly those. What keeps anyone from aiming at it:
//
//   - the hours follow from the seed, which nobody has before the draw;
//   - each harvest's hash was here before the draw, so a harvest cannot be made again to suit
//     a round;
//   - a curator runs with no network and may add a few thousand tokens of its own, and a
//     script whose own text holds a passage of the exam is refused by the grader.
//
// An entry may be submitted after the writing its exam will hold was published. That is let
// be: it would have to carry the text in, and the last point is what stops that.

const (
	harvestHour  = 3600            // a harvest is an hour of the clock
	harvestsKept = 24 * 21         // the oldest are forgotten past three weeks of them
	harvestSkew  = 6 * harvestHour // an hour is not taken this long before it could have settled
)

// Harvest is one hour's harvest, as the examiner reported it.
type Harvest struct {
	Hour    int64          `json:"hour"`   // unix seconds, a multiple of an hour: the harvest is of [hour, hour+3600)
	SHA     string         `json:"sha256"` // of the snapshot file the examiner keeps
	Docs    int            `json:"documents"`
	Sources map[string]int `json:"sources,omitempty"` // documents from each source
	At      int64          `json:"at"`                // when the server was told
}

// RoundHour is one of the harvests a round's exam is made from.
type RoundHour struct {
	Hour int64  `json:"hour"`
	SHA  string `json:"sha256"`
}

func harvestKey(hour int64) string { return fmt.Sprintf("harvest/%012d", hour) }

// chooseHours gives the k of the hours whose sha256 with the round's seed is smallest, in the
// hours' order. stet.exam.fresh.harvest.choose_hours is the same rule, for anyone to check with.
func chooseHours(seedHex string, hours []int64, k int) []int64 {
	seed, _ := hex.DecodeString(seedHex)
	type ranked struct {
		hour int64
		sum  [32]byte
	}
	rows := make([]ranked, 0, len(hours))
	seen := map[int64]bool{}
	for _, h := range hours {
		if !seen[h] {
			seen[h] = true
			rows = append(rows, ranked{h, sha256.Sum256(append(append([]byte{}, seed...), []byte("|fresh/hour|"+strconv.FormatInt(h, 10))...))})
		}
	}
	sort.Slice(rows, func(i, j int) bool { return bytes.Compare(rows[i].sum[:], rows[j].sum[:]) < 0 })
	if k < len(rows) {
		rows = rows[:max(k, 0)]
	}
	out := make([]int64, len(rows))
	for i, r := range rows {
		out[i] = r.hour
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// roundHours chooses a round's hours at its draw: from the harvests the server had been told
// of before the draw whose hour began no longer than the season's reach before it.
func (a *App) roundHours(s *State, rd *Round) []RoundHour {
	f := a.season.Exam.Fresh
	var could []int64
	for _, h := range s.Harvests {
		if h.Hour >= rd.DrawAt-f.Reach && h.At <= rd.DrawAt {
			could = append(could, h.Hour)
		}
	}
	var out []RoundHour
	for _, hour := range chooseHours(rd.Seed, could, f.Hours) {
		out = append(out, RoundHour{hour, s.Harvests[hour].SHA})
	}
	return out
}

// snapshotOf is the one name for what an exam's writing was made from: its hours' hashes, in
// the hours' order, hashed.
func snapshotOf(hours []RoundHour) string {
	var b []byte
	for _, h := range hours {
		b = append(b, h.SHA...)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// checkHours says whether an exam was made from the hours it should have been. A round that
// was given hours at its draw takes an exam made from those and no others. One that was given
// none, because the server had been told of no harvest, takes an exam made from any harvests
// it has been told of since, of hours within the season's reach of the draw; late then says so.
func (a *App) checkHours(s *State, rd *Round, p *ExamPost) (late bool, err error) {
	if len(rd.Hours) > 0 {
		if len(p.Hours) != len(rd.Hours) {
			return false, fail(400, "this round's exam is made from %d hours' harvests, named at its draw", len(rd.Hours))
		}
		for i, h := range rd.Hours {
			if p.Hours[i] != h {
				return false, fail(400, "this round's exam is made from the harvests named at its draw, and this one was not")
			}
		}
	} else {
		for i, h := range p.Hours {
			got := s.Harvests[h.Hour]
			if got == nil || got.SHA != h.SHA || h.Hour < rd.DrawAt-a.season.Exam.Fresh.Reach || (i > 0 && p.Hours[i-1].Hour >= h.Hour) {
				return false, fail(400, "an exam is made from harvests the server has been told of, of hours in the day before the draw")
			}
		}
		late = len(p.Hours) > 0
	}
	if len(p.Hours) > 0 && p.Snapshot != snapshotOf(p.Hours) {
		return false, fail(400, "the exam's snapshot is not its hours' harvests")
	}
	return late, nil
}

// apiHarvestPost: POST /api/v1/admin/harvests, from the organisers' examiner. Telling the
// server of the same harvest again is harmless; another harvest of the same hour is refused,
// because a round may already rest on the first.
func (a *App) apiHarvestPost(w http.ResponseWriter, r *http.Request) {
	_, _, body, err := a.signedBy(w, r, 1<<16, true, organisers)
	if err != nil {
		writeErr(w, err)
		return
	}
	var h Harvest
	if err := json.Unmarshal(body, &h); err != nil {
		writeErr(w, fail(400, "not a harvest: %v", err))
		return
	}
	t := now()
	settle := a.season.Exam.Fresh.Settle
	switch {
	case h.Hour <= 0 || h.Hour%harvestHour != 0 || !sha256Re.MatchString(h.SHA) || h.Docs < 0 || len(h.Sources) > 64:
		writeErr(w, fail(400, "the harvest does not add up"))
		return
	case h.Hour+harvestHour > t:
		writeErr(w, fail(400, "that hour is not over"))
		return
	case h.Hour+harvestHour+settle > t+harvestSkew:
		writeErr(w, fail(400, "that hour has not settled"))
		return
	}
	h.At = t
	created := false
	err = a.st.update(r.Context(), func(tx *Tx) error {
		s := tx.s
		if old := s.Harvests[h.Hour]; old != nil {
			if old.SHA != h.SHA {
				return fail(409, "that hour has been harvested already, and this is another harvest of it")
			}
			h = *old
			return nil
		}
		created = true
		keep := h
		s.Harvests[h.Hour] = &keep
		tx.put(harvestKey(h.Hour), &keep)
		if len(s.Harvests) > harvestsKept { // the oldest go; a round's own hours are kept on the round
			hours := make([]int64, 0, len(s.Harvests))
			for hour := range s.Harvests {
				hours = append(hours, hour)
			}
			sort.Slice(hours, func(i, j int) bool { return hours[i] < hours[j] })
			for _, hour := range hours[:len(hours)-harvestsKept] {
				delete(s.Harvests, hour)
				tx.put(harvestKey(hour), nil)
			}
		}
		return nil
	})
	if err != nil {
		writeErr(w, err)
		return
	}
	code := 200
	if created {
		code = 201
	}
	writeJSON(w, code, h)
}

// apiHarvests: GET /api/v1/harvests, the harvests the server has been told of, newest first.
// Public: with a round's seed, anyone can work out which hours its exam had to be made from.
func (a *App) apiHarvests(w http.ResponseWriter, r *http.Request) {
	limit := 72
	if n, err := strconv.Atoi(r.URL.Query().Get("limit")); err == nil && n > 0 && n <= harvestsKept {
		limit = n
	}
	var out []*Harvest
	a.st.view(func(s *State) {
		for _, h := range s.Harvests {
			out = append(out, h)
		}
	})
	sort.Slice(out, func(i, j int) bool { return out[i].Hour > out[j].Hour })
	if len(out) > limit {
		out = out[:limit]
	}
	if out == nil {
		out = []*Harvest{}
	}
	f := a.season.Exam.Fresh
	writeJSON(w, 200, map[string]any{"harvests": out, "hours": f.Hours, "reach_seconds": f.Reach, "settle_seconds": f.Settle})
}
