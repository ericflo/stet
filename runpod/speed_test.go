package main

import (
	"strings"
	"testing"
)

// A kind of GPU is known by the longest name its ID holds as whole words.
func TestTheSpeedOfEachKindOfGPU(t *testing.T) {
	for id, want := range map[string]float64{
		"NVIDIA RTX A4000":               0.64, // not an A40
		"NVIDIA A40":                     0.95,
		"NVIDIA L40S":                    1.9, // not an L40
		"NVIDIA L40":                     1.6,
		"NVIDIA L4":                      0.45,
		"NVIDIA GeForce RTX 3080 Ti":     0.95, // not a 3080
		"NVIDIA GeForce RTX 3080":        0.85,
		"NVIDIA GeForce RTX 4090":        1.8,
		"NVIDIA A100-SXM4-80GB":          2,
		"NVIDIA RTX 6000 Ada Generation": 1.8,
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
	kind := func(id string, price float64) Choice { return Choice{GPU: GPUType{ID: id}, Price: price} }
	cs := []Choice{ // RunPod's this morning, cheapest first
		kind("NVIDIA GeForce RTX 3080 Ti", 0.18), kind("NVIDIA RTX 4000 SFF Ada Generation", 0.20),
		kind("NVIDIA RTX 4000 Ada Generation", 0.28), kind("NVIDIA GeForce RTX 3090", 0.50),
		kind("NVIDIA A40", 0.59), kind("NVIDIA L4", 0.69), kind("NVIDIA RTX A6000", 0.69),
		kind("NVIDIA L40", 0.82), kind("NVIDIA GeForce RTX 4090", 0.89),
	}
	byStudent(cs)
	var got []string
	for _, c := range cs {
		got = append(got, shortGPU(c.GPU.ID))
	}
	want := "RTX 3080 Ti, RTX 4090, L40, RTX 3090, RTX 4000 Ada Generation, A40, RTX A6000, L4, RTX 4000 SFF Ada Generation"
	if s := strings.Join(got, ", "); s != want {
		t.Fatalf("tried in the order %s", s)
	}
}
