package main

// Temporary stand-in for NextScore until the port of newexam/aggregate.py lands (it replaces this file).
type ItemInfo struct {
	Pillar  string `json:"pillar"`
	Part    string `json:"part"`
	Lang    string `json:"lang"`
	Stratum string `json:"stratum"`
}

func NextScore(items []ItemInfo, gains []float64, shares map[string]float64) (float64, map[string]float64, error) {
	s := 0.0
	for _, g := range gains {
		s += g
	}
	return s / float64(len(gains)), map[string]float64{}, nil
}
