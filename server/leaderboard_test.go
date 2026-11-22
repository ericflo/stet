package main

import (
	"path/filepath"
	"testing"
)

// Each row of the leaderboard says whether its curator was entered by the organisers. The
// finals' tooling tells the yardsticks from the entrants by it: the list of verifiers is cut
// at 200 rows, most work first, and the organisers' accounts count as having done none.
func TestTheLeaderboardSaysWhichCuratorsAreTheOrganisers(t *testing.T) {
	pub, priv := newKey()
	cfg := testConfig(t, pub)
	w := startWorld(t, cfg, newMemKV(), dirBlobs{filepath.Join(cfg.DataDir, "blobs")})
	anchor := w.join("anchor-one", priv)
	w.submit(anchor, "reference", "print(1)")
	w.submit(w.join("alice", nil), "tidy", "print(2)")
	w.check()
	w.play(anchor, 5, func(name string, round int) float64 {
		if name == "reference" {
			return 5.7 + 0.003*float64(round%2)
		}
		return 5.4 + 0.003*float64(round%3)
	})
	var lb struct {
		Editors     []map[string]any `json:"editors"`
		Provisional []map[string]any `json:"provisional"`
	}
	(&actor{t: t, srv: w.srv}).do("GET", "/api/v1/leaderboard", nil, &lb)
	got := map[string]any{}
	for _, row := range append(lb.Editors, lb.Provisional...) {
		got[row["name"].(string)] = row["organisers"]
	}
	if len(lb.Editors) != 2 || got["reference"] != true || got["tidy"] != false {
		t.Fatalf("whose each curator is, as the leaderboard says it: %v (ranked: %d)", got, len(lb.Editors))
	}
}
