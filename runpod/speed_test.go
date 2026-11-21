package main

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
)

// A kind of GPU is known by the longest name its ID holds as whole words.
func TestTheSpeedOfEachKindOfGPU(t *testing.T) {
	for id, want := range map[string]float64{
		"NVIDIA RTX A4000":                             0.64, // not an A40
		"NVIDIA A40":                                   0.95,
		"NVIDIA L40S":                                  1.9, // not an L40
		"NVIDIA L40":                                   1.6,
		"NVIDIA L4":                                    0.45,
		"NVIDIA GeForce RTX 3080 Ti":                   0.95, // not a 3080
		"NVIDIA GeForce RTX 3080":                      0.85,
		"NVIDIA GeForce RTX 4090":                      1.8,
		"NVIDIA A100-SXM4-80GB":                        2,
		"NVIDIA RTX 6000 Ada Generation":               1.8,
		"NVIDIA RTX PRO 6000 Blackwell Server Edition": 2.6,
		"AMD Instinct MI300X OAM":                      0,
		"NVIDIA RTX 4000 SFF Ada Generation":           0,
	} {
		if got := speed(id); got != want {
			t.Errorf("%s: %v, want %v", id, got, want)
		}
	}
}

// For the bigger students, what a student costs comes first, the quicker card where that is
// about the same, and kinds of unknown speed last.
func TestTheBiggerStudentsAreRentedByWhatAStudentCosts(t *testing.T) {
	kind := func(id string, price float64) Offer { return Offer{Card: id, Price: price} }
	cs := []Offer{ // RunPod's this morning, cheapest first
		kind("NVIDIA GeForce RTX 3080 Ti", 0.18), kind("NVIDIA RTX 4000 SFF Ada Generation", 0.20),
		kind("NVIDIA RTX 4000 Ada Generation", 0.28), kind("NVIDIA GeForce RTX 3090", 0.50),
		kind("NVIDIA A40", 0.59), kind("NVIDIA L4", 0.69), kind("NVIDIA RTX A6000", 0.69),
		kind("NVIDIA L40", 0.82), kind("NVIDIA GeForce RTX 4090", 0.89),
	}
	byStudent(cs, speed)
	var got []string
	for _, c := range cs {
		got = append(got, shortGPU(c.Card))
	}
	want := "RTX 3080 Ti, RTX 4090, L40, RTX 3090, RTX 4000 Ada Generation, A40, RTX A6000, L4, RTX 4000 SFF Ada Generation"
	if s := strings.Join(got, ", "); s != want {
		t.Fatalf("tried in the order %s", s)
	}
}

// By default the fastest card comes first, the cheaper first where two are as fast, and a
// kind of unknown speed last.
func TestMachinesAreRentedFastestFirst(t *testing.T) {
	kind := func(id string, price float64) Offer { return Offer{Card: id, Price: price} }
	cs := []Offer{ // RunPod's on Nov 20, cheapest first
		kind("NVIDIA GeForce RTX 3080 Ti", 0.18), kind("NVIDIA RTX A4000", 0.25), kind("NVIDIA RTX A4500", 0.25),
		kind("NVIDIA RTX 4000 Ada Generation", 0.28), kind("NVIDIA Mystery 1", 0.30), kind("NVIDIA RTX PRO 4000 Blackwell", 0.57),
		kind("NVIDIA A40", 0.59), kind("NVIDIA L4", 0.69), kind("NVIDIA RTX A6000", 0.69), kind("NVIDIA GeForce RTX 4090", 0.89),
		kind("NVIDIA L40S", 1.09), kind("NVIDIA RTX 6000 Ada Generation", 1.09),
	}
	byPace(cs, speed)
	var got []string
	for _, c := range cs {
		got = append(got, shortGPU(c.Card))
	}
	want := "L40S, RTX 4090, RTX 6000 Ada Generation, RTX A6000, RTX 3080 Ti, A40, RTX PRO 4000 Blackwell, RTX A4500, RTX A4000, RTX 4000 Ada Generation, L4, Mystery 1"
	if s := strings.Join(got, ", "); s != want {
		t.Fatalf("tried in the order %s", s)
	}
}

// With --cheapest, what a grade costs comes first: its price over its speed, so a quick card
// dearer by the hour can come before a slow cheap one, and a kind of unknown speed counts as
// half a 3090.
func TestMachinesAreRentedByWhatAGradeCosts(t *testing.T) {
	kind := func(id string, price float64) Offer { return Offer{Card: id, Price: price} }
	cs := []Offer{ // RunPod's on Nov 20, cheapest first
		kind("NVIDIA GeForce RTX 3080 Ti", 0.18), kind("NVIDIA RTX A4000", 0.25), kind("NVIDIA RTX A4500", 0.25),
		kind("NVIDIA RTX 4000 Ada Generation", 0.28), kind("NVIDIA Mystery 1", 0.30), kind("NVIDIA RTX PRO 4000 Blackwell", 0.57),
		kind("NVIDIA A40", 0.59), kind("NVIDIA L4", 0.69), kind("NVIDIA RTX A6000", 0.69), kind("NVIDIA GeForce RTX 4090", 0.89),
		kind("NVIDIA L40S", 1.09),
	}
	byGrade(cs, speed)
	var got []string
	for _, c := range cs {
		got = append(got, shortGPU(c.Card))
	}
	want := "RTX 3080 Ti, RTX A4500, RTX A4000, RTX 4000 Ada Generation, RTX 4090, L40S, Mystery 1, A40, RTX A6000, RTX PRO 4000 Blackwell, L4"
	if s := strings.Join(got, ", "); s != want {
		t.Fatalf("tried in the order %s", s)
	}
}

// For a grade taken before an exam, each kind is rented just in time to report between a
// look before the exam and 15 minutes after: none before its time, none while a kind still to
// come would cost less a round, and once no kind can be in time, the quickest first.
func TestGradersAreRentedJustInTime(t *testing.T) {
	kind := func(id string, price float64) Offer { return Offer{Card: id, Price: price} }
	cs := []Offer{
		kind("NVIDIA RTX A4000", 0.17), kind("NVIDIA GeForce RTX 3080 Ti", 0.18), kind("NVIDIA GeForce RTX 3090", 0.22),
		kind("NVIDIA RTX 4000 Ada Generation", 0.28), kind("NVIDIA GeForce RTX 5080", 0.39), kind("NVIDIA GeForce RTX 3090", 0.50),
		kind("NVIDIA GeForce RTX 4090", 0.89), kind("NVIDIA L40S", 1.09), kind("NVIDIA L4", 0.69),
	}
	// Leads of 11 minutes and two passes of 18 over the speed: L4 91 minutes, the 4000 Ada 71,
	// the A4000 67, the 3080 Ti 49, the 3090s 47, the 5080 37, the 4090 31, the L40S 30. A
	// round's grade then costs $0.15 on the 3080 Ti, $0.17 on the cheaper 3090, $0.19 on the
	// A4000, $0.24 on the 5080, $0.33 on the 4000 Ada, $0.39 on the dearer 3090, $0.46 on the
	// 4090, $0.54 on the L40S and $1.05 on the L4.
	lead := func(c Offer) time.Duration {
		return 11*time.Minute + time.Duration(2*float64(18*time.Minute)/speed(c.Card))
	}
	at := func(left time.Duration) string {
		var got []string
		for _, c := range forExam(cs, left, 15*time.Minute, time.Minute, examPlan{lead: lead}) {
			got = append(got, fmt.Sprintf("%s $%.2f", shortGPU(c.Card), c.Price))
		}
		return strings.Join(got, ", ")
	}
	for _, c := range []struct {
		left time.Duration
		want string
	}{
		{90 * time.Minute, ""}, // the L4's time, but cheaper kinds are to come
		{72 * time.Minute, ""}, // the 4000 Ada's, the same
		{50 * time.Minute, ""}, // the A4000's has passed, and the 3080 Ti's has not come
		{49*time.Minute + 30*time.Second, "RTX 3080 Ti $0.18"},
		{47*time.Minute + 30*time.Second, "RTX 3080 Ti $0.18, RTX 3090 $0.22"},
		{37 * time.Minute, "RTX 3080 Ti $0.18, RTX 3090 $0.22, RTX 5080 $0.39, RTX 3090 $0.50"},
		{20 * time.Minute, "RTX 4090 $0.89, L40S $1.09"},
		{10 * time.Minute, "L40S $1.09, RTX 4090 $0.89, RTX 5080 $0.39, RTX 3090 $0.22, RTX 3090 $0.50, RTX 3080 Ti $0.18, RTX A4000 $0.17, RTX 4000 Ada Generation $0.28, L4 $0.69"},
	} {
		if got := at(c.left); got != c.want {
			t.Errorf("%s before the exam: %q, want %q", c.left, got, c.want)
		}
	}
	if next, ok := waitingFor(cs, 72*time.Minute, time.Minute, examPlan{lead: lead}); !ok || next.Price != 0.18 {
		t.Errorf("waiting for %+v", next)
	}
}

// When the server lets a machine hold several grades before the exam, a kind is best rented
// just in time to take the most it may: its first grade's lead, then a pass for each further
// one, finished two minutes before the exam. What a grade costs is the machine's hours over
// the grades it takes.
func TestAMachineTakesSeveralGradesBeforeTheExam(t *testing.T) {
	lead := func(c Offer) time.Duration {
		return 11*time.Minute + time.Duration(2*float64(18*time.Minute)/speed(c.Card))
	}
	pass := func(c Offer) time.Duration { return time.Duration(float64(18*time.Minute) / speed(c.Card)) }
	pl := examPlan{lead: lead, pass: pass, most: 3}
	g4090, g3080ti := Offer{Card: "RTX 4090", Price: 0.89}, Offer{Card: "RTX 3080 Ti", Price: 0.18}
	// A 4090 passes in 10 minutes and leads by 31: its first grade is trained 30 minutes after
	// it is rented, and each further one 10 later, by two minutes before the exam.
	for left, want := range map[time.Duration]int{52 * time.Minute: 3, 70 * time.Minute: 3, 42 * time.Minute: 2, 35 * time.Minute: 1, 20 * time.Minute: 1} {
		if got := pl.grades(g4090, left); got != want {
			t.Errorf("a 4090 rented %s before the exam takes %d grades, want %d", left, got, want)
		}
	}
	if full := pl.full(g4090); full != 52*time.Minute {
		t.Errorf("a 4090 is best rented %s before the exam", full)
	}
	if c := pl.cost(g4090, 52*time.Minute); math.Abs(c-0.89*53.0/60/3) > 1e-9 {
		t.Errorf("a grade on a 4090 rented in its time: $%.4f", c)
	}
	// Rented 90 minutes before, a 4090 would wait 38 of them: a grade would cost more.
	if pl.cost(g4090, 90*time.Minute) <= pl.cost(g4090, 52*time.Minute) {
		t.Error("a 4090 rented sooner than its time costs no more a grade")
	}
	cs := []Offer{g4090, g3080ti, {Card: "RTX A4000", Price: 0.17}}
	// The 3080 Ti's time (a lead of 49 minutes and two passes of 19, 88 minutes) has not come
	// 90 minutes before the exam, and it is the cheapest a grade: nothing is rented yet. In its
	// time it is, to take three.
	if got := forExam(cs, 90*time.Minute, 15*time.Minute, time.Minute, pl); len(got) != 0 {
		t.Errorf("90 minutes before: %+v", got)
	}
	if next, ok := waitingFor(cs, 90*time.Minute, time.Minute, pl); !ok || next.Card != "RTX 3080 Ti" || pl.grades(next, pl.full(next)) != 3 {
		t.Errorf("waiting for %+v", next)
	}
	// (The A4000, which would take one, costs less a grade than the 4090 to come, and is tried
	// if the 3080 Ti is not to be had.)
	if got := forExam(cs, 88*time.Minute, 15*time.Minute, time.Minute, pl); len(got) != 2 || got[0].Card != "RTX 3080 Ti" || pl.grades(got[0], 88*time.Minute) != 3 || got[1].Card != "RTX A4000" {
		t.Errorf("88 minutes before: %+v", got)
	}
	// With one grade a machine, as before.
	one := examPlan{lead: lead, pass: pass, most: 1}
	if one.grades(g4090, 90*time.Minute) != 1 || one.full(g4090) != lead(g4090) || one.cost(g4090, 90*time.Minute) != 0.89*lead(g4090).Hours() {
		t.Error("one grade a machine is not as it was")
	}
}

// With the server's window, a machine rented as early as it likes still has its first grade
// trained no sooner than the window and the grades after it allow: rented in its time, it
// takes the most grades, as without the window.
func TestTheWindowForEarlyGrades(t *testing.T) {
	lead := func(c Offer) time.Duration { return 24 * time.Minute } // a 4090: 5 to start, 2 to fetch, 2×8, 1 to sit
	pass := func(c Offer) time.Duration { return 8 * time.Minute }
	pl := examPlan{lead: lead, pass: pass, most: 3, window: 10 * time.Minute}
	g := Offer{Card: "RTX 4090", Price: 0.40}
	if f := pl.first(g, 3*time.Hour); f != 26*time.Minute {
		t.Errorf("rented three hours before, its first grade is trained %s before the exam, not 26m", f)
	}
	if full := pl.full(g); full != 41*time.Minute || pl.grades(g, full) != 3 {
		t.Errorf("rented in its time (%s), it takes %d grades", full, pl.grades(g, full))
	}
	if pl.grades(g, 3*time.Hour) != 3 {
		t.Errorf("rented three hours before, it takes %d grades", pl.grades(g, 3*time.Hour))
	}
}
