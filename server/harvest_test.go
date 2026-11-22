package main

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// The rule that gives a round its hours is the harness's too (stet.exam.fresh.harvest.choose_hours,
// tests/test_season.py has this vector), so that anyone can check which hours an exam had to be
// made from.
func TestARoundsHoursFollowFromItsSeed(t *testing.T) {
	var day []int64
	for i := int64(1); i <= 24; i++ {
		day = append(day, 3600*i)
	}
	zero := strings.Repeat("00", 32)
	if got := chooseHours(zero, day, 3); !reflect.DeepEqual(got, []int64{3600, 43200, 75600}) {
		t.Fatalf("the hours for the all-zero seed: %v", got)
	}
	if got := chooseHours(zero, append(append([]int64{}, day...), day...), 3); len(got) != 3 {
		t.Fatalf("an hour named twice is one hour: %v", got)
	}
	if got := chooseHours(zero, day[:2], 3); !reflect.DeepEqual(got, []int64{3600, 7200}) {
		t.Fatalf("never more hours than there are: %v", got)
	}
	if got := chooseHours(strings.Repeat("11", 32), day, 3); reflect.DeepEqual(got, []int64{3600, 43200, 75600}) {
		t.Fatal("another seed chose the same hours")
	}
	if len(chooseHours(zero, nil, 3)) != 0 {
		t.Fatal("hours from none")
	}
}

func harvestBody(hour int64, sha string) map[string]any {
	return map[string]any{"hour": hour, "sha256": strings.Repeat(sha, 32), "documents": 700, "sources": map[string]int{"wikipedia": 300, "ccnews": 150}}
}

// The examiner tells the server of each hour's harvest as it is made. A round drawn afterwards
// is given three of the day's hours by its seed, and takes an exam made from those and no others.
func TestARoundIsGivenItsHoursAtTheDrawAndItsExamIsMadeFromThem(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	stranger := w.join("stranger", nil)
	w.submit(anchor, "ref-a", "print(1)")
	hour := now() - now()%3600
	var out map[string]any
	if code := stranger.do("POST", "/api/v1/admin/harvests", harvestBody(hour-4*3600, "aa"), &out); code != 403 {
		t.Fatalf("a harvest from someone who is not an organiser: HTTP %d", code)
	}
	for _, bad := range []map[string]any{harvestBody(hour+3600, "aa"), harvestBody(hour-4*3600+60, "aa"), harvestBody(hour-4*3600, "zz"), harvestBody(hour+12*3600, "aa")} {
		if code := anchor.do("POST", "/api/v1/admin/harvests", bad, &out); code != 400 {
			t.Fatalf("a harvest that does not add up (%v): HTTP %d", bad["hour"], code)
		}
	}
	// Ten of the day's hours, one from before it, and one harvested twice.
	for i := int64(2); i <= 11; i++ {
		if code := anchor.do("POST", "/api/v1/admin/harvests", harvestBody(hour-i*3600, "aa"), &out); code != 201 {
			t.Fatalf("telling the server of a harvest: HTTP %d %v", code, out)
		}
	}
	if code := anchor.do("POST", "/api/v1/admin/harvests", harvestBody(hour-40*3600, "bb"), &out); code != 201 {
		t.Fatalf("an older harvest: HTTP %d", code)
	}
	if code := anchor.do("POST", "/api/v1/admin/harvests", harvestBody(hour-5*3600, "aa"), &out); code != 200 {
		t.Fatalf("the same harvest again: HTTP %d", code)
	}
	if code := anchor.do("POST", "/api/v1/admin/harvests", harvestBody(hour-5*3600, "cc"), &out); code != 409 {
		t.Fatalf("another harvest of an hour already harvested: HTTP %d", code)
	}
	var list struct {
		Harvests []Harvest `json:"harvests"`
		Hours    int       `json:"hours"`
		Reach    int64     `json:"reach_seconds"`
	}
	if code := stranger.do("GET", "/api/v1/harvests", nil, &list); code != 200 || len(list.Harvests) != 11 || list.Harvests[0].Hour != hour-2*3600 || list.Hours != 3 || list.Reach != 93600 {
		t.Fatalf("the public list of harvests: HTTP %d %+v", code, list)
	}

	idx := w.drawWithoutExam()
	var rd Round
	w.app.st.view(func(s *State) { rd = *s.Rounds[idx] })
	var could []int64
	for i := int64(2); i <= 11; i++ {
		could = append(could, hour-i*3600)
	}
	want := chooseHours(rd.Seed, could, 3)
	if len(rd.Hours) != 3 || rd.Hours[0].Hour != want[0] || rd.Hours[1].Hour != want[1] || rd.Hours[2].Hour != want[2] || rd.Hours[0].SHA != strings.Repeat("aa", 32) || rd.HoursLate {
		t.Fatalf("the round's hours: %+v, the seed chooses %v", rd.Hours, want)
	}
	var pubRound struct {
		Round struct {
			Hours []RoundHour `json:"exam_hours"`
		} `json:"round"`
	}
	if code := stranger.do("GET", "/api/v1/rounds/"+roundID(w.app.season.ID, idx), nil, &pubRound); code != 200 || !reflect.DeepEqual(pubRound.Round.Hours, rd.Hours) {
		t.Fatalf("the round says which hours its exam is made from: HTTP %d %+v", code, pubRound.Round.Hours)
	}
	// A harvest the server is told of after the draw is no part of this round.
	if code := anchor.do("POST", "/api/v1/admin/harvests", harvestBody(hour-12*3600, "dd"), &out); code != 201 {
		t.Fatalf("a harvest after the draw: HTTP %d", code)
	}
	other := []RoundHour{rd.Hours[0], rd.Hours[1], {hour - 12*3600, strings.Repeat("dd", 32)}}
	for name, hours := range map[string][]RoundHour{"none": nil, "two of the three": rd.Hours[:2], "another hour": other} {
		if code, msg := w.postExamFrom(anchor, idx, testTable(), hours, snapshotOf(hours)); code != 400 {
			t.Fatalf("an exam made from %s: HTTP %d %s", name, code, msg)
		}
	}
	if code, msg := w.postExamFrom(anchor, idx, testTable(), rd.Hours, strings.Repeat("ee", 32)); code != 400 {
		t.Fatalf("an exam whose snapshot is not its hours': HTTP %d %s", code, msg)
	}
	if code, msg := w.postExamFrom(anchor, idx, testTable(), rd.Hours, snapshotOf(rd.Hours)); code != 200 {
		t.Fatalf("the exam made from the round's hours: HTTP %d %s", code, msg)
	}
	w.check()
}

// The server had been told of no harvest when the round was drawn (the examiner was off). The
// exam is then made from harvests it is told of afterwards, and the round says so.
func TestARoundDrawnWithNoHarvestTakesHoursChosenLate(t *testing.T) {
	pub, priv := newKey()
	w := newWorld(t, pub)
	anchor := w.join("anchor-one", priv)
	w.submit(anchor, "ref-a", "print(1)")
	idx := w.drawWithoutExam()
	hour := now() - now()%3600
	var out map[string]any
	for _, h := range []int64{hour - 3*3600, hour - 2*3600, hour - 60*3600} {
		if code := anchor.do("POST", "/api/v1/admin/harvests", harvestBody(h, "aa"), &out); code != 201 {
			t.Fatalf("telling the server of a harvest: HTTP %d %v", code, out)
		}
	}
	sha := strings.Repeat("aa", 32)
	old := []RoundHour{{hour - 60*3600, sha}}
	if code, msg := w.postExamFrom(anchor, idx, testTable(), old, snapshotOf(old)); code != 400 {
		t.Fatalf("an exam from an hour before the round's day: HTTP %d %s", code, msg)
	}
	unknown := []RoundHour{{hour - 5*3600, sha}}
	if code, msg := w.postExamFrom(anchor, idx, testTable(), unknown, snapshotOf(unknown)); code != 400 {
		t.Fatalf("an exam from a harvest the server was never told of: HTTP %d %s", code, msg)
	}
	hours := []RoundHour{{hour - 3*3600, sha}, {hour - 2*3600, sha}}
	if code, msg := w.postExamFrom(anchor, idx, testTable(), hours, snapshotOf(hours)); code != 200 {
		t.Fatalf("the exam: HTTP %d %s", code, msg)
	}
	var rd Round
	w.app.st.view(func(s *State) { rd = *s.Rounds[idx] })
	if !rd.HoursLate || !reflect.DeepEqual(rd.Hours, hours) {
		t.Fatalf("the round's hours, chosen late: %+v late=%v", rd.Hours, rd.HoursLate)
	}
	w.check()
}

// postExamFrom posts a round's exam as made from the given hours' harvests.
func (w *world) postExamFrom(by *actor, idx int, table []ItemInfo, hours []RoundHour, snapshot string) (int, string) {
	w.t.Helper()
	var out struct {
		Error string `json:"error"`
	}
	body := testExam(table)
	body["hours"], body["snapshot_sha256"] = hours, snapshot
	code := by.do("POST", fmt.Sprintf("/api/v1/admin/rounds/%s/exam", roundID(w.app.season.ID, idx)), body, &out)
	return code, out.Error
}
