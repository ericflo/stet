package main

import (
	"math"
	"testing"
)

// Season 2's curves are written as minus the gain of each unedited student over the one
// trained on the budget itself: 0 at 1x, positive below it, negative above (`stet curve`,
// src/stet/exam/measure.py). The multiplier reads only differences from the point at 1x, so
// such a curve needs nothing new here. These are the numbers tests/test_measure.py reads off
// the same curve with the client's score.multiplier.
func TestACurveWrittenAsGainsReadsAsAMultiplier(t *testing.T) {
	s := &Season{Curves: map[string][][2]float64{"official/standard": {{0.5, 0.04}, {1, 0}, {2, -0.035}, {4, -0.06}}}}
	for _, c := range []struct {
		gain, want float64
		capped     bool
	}{
		{0, 1, false},
		{0.035, 2, false},
		{0.0175, math.Sqrt2, false},
		{0.0475, math.Pow(2, 1.5), false},
		{0.06, 4, false},
		{0.1, 4, true}, // past the last measured point: "more than 4x", never extrapolated
		{-0.02, math.Pow(2, -0.5), false},
		{-0.04, 0.5, false},
		{-0.09, 0.5, true},
	} {
		m, capped, ok := s.multiplier("official/standard", c.gain)
		if !ok || capped != c.capped || math.Abs(m-c.want) > 1e-9 {
			t.Fatalf("a gain of %v: got %v (capped %v, ok %v), want %v (capped %v)", c.gain, m, capped, ok, c.want, c.capped)
		}
	}
	// A curve whose points are in any order, as a season file might hold it, reads the same.
	s.Curves["official/large"] = [][2]float64{{2, -0.035}, {1, 0}, {0.5, 0.04}}
	if m, capped, ok := s.multiplier("official/large", 0.0175); !ok || capped || math.Abs(m-math.Sqrt2) > 1e-9 {
		t.Fatalf("got %v %v %v", m, capped, ok)
	}
}
