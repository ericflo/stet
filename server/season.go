package main

import (
	"crypto/sha256"
	_ "embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
)

// season.json is written by tools/export_season.py. The server never derives a corpus or
// trains anything; it only needs the season's public facts and the seed formula.
//
//go:embed season.json
var seasonJSON []byte

type Scale struct {
	CorpusUnits        int     `json:"corpus_units"`
	TrainTokens        int     `json:"train_tokens"`
	NLayer             int     `json:"n_layer"`
	NHead              int     `json:"n_head"`
	DModel             int     `json:"d_model"`
	Ctx                int     `json:"ctx"`
	BatchSize          int     `json:"batch_size"`
	MicroBatch         int     `json:"micro_batch,omitempty"`
	LR                 float64 `json:"lr"`
	ExamItemsPerSource int     `json:"exam_items_per_source"`
	EditSeconds        int     `json:"edit_seconds"`
	SFTSteps           int     `json:"sft_steps,omitempty"`
}

type Beacon struct {
	Name      string   `json:"name"`
	ChainHash string   `json:"chain_hash"`
	Genesis   int64    `json:"genesis"`
	Period    int64    `json:"period"`
	Relays    []string `json:"relays"`
}

type Pool struct {
	Kind          string `json:"kind"`
	BaseURL       string `json:"base_url"`
	Paths         string `json:"paths,omitempty"`
	ShardsPerUnit int    `json:"shards_per_unit"`
	Crawl         string `json:"crawl,omitempty"`
	About         string `json:"about"`
}

// Tolerance is how far apart two honest measurements of the same thing may be.
type Tolerance struct {
	Grade    float64 `json:"grade"`    // two grades of the same edit script
	Baseline float64 `json:"baseline"` // a verifier's baseline against the round's
	Run      float64 `json:"run"`      // two runs of the same editor: the gains of the scripts they wrote
}

type ScriptLimits struct {
	MaxBytes int64 `json:"max_bytes"`
}

// AssetLimits bounds the files an editor may name by reference (model weights).
type AssetLimits struct {
	MaxFiles int   `json:"max_files"`
	MaxBytes int64 `json:"max_bytes"`
}

// Machine is what every editor is given, whatever the machine that runs it has.
type Machine struct {
	Cores       int `json:"cores"`
	MemoryGB    int `json:"memory_gb"`
	GPUMemoryGB int `json:"gpu_memory_gb"`
	ScratchGB   int `json:"scratch_gb"`
}

// RunnerFloor is the slowest machine that may run editors: see src/stet/bench.py.
type RunnerFloor struct {
	GPUTokensPerSecond float64 `json:"gpu_tokens_per_second"`
	CPUMBPerSecond     float64 `json:"cpu_mb_per_second"`
	GPUMemoryGB        float64 `json:"gpu_memory_gb"`
	MemoryGB           float64 `json:"memory_gb"`
	Cores              int     `json:"cores"`
	DiskGB             float64 `json:"disk_gb"`
}

// Environment is the Python and the libraries an editor may count on, at exact versions.
type Environment struct {
	Python     string            `json:"python"`
	TorchIndex map[string]string `json:"torch_index"` // where PyTorch's build for each GPU maker is
	Packages   map[string]string `json:"packages"`
	Also       map[string]string `json:"also"` // what those bring with them, pinned too
	Linux      []string          `json:"linux_x86_64_only"`
	CUDALinux  []string          `json:"cuda_linux_only,omitempty"`
}

// BaselineRule is how many machines measure a round's unedited baseline. See judge.asksBaseline.
type BaselineRule struct {
	PerRound int `json:"per_round"` // 0: every grader measures it
}

// ReferenceRule is which of the organisers' reference editors run in every round, by name,
// and how often each of the others takes a turn. See App.resting.
type ReferenceRule struct {
	EveryRound []string `json:"every_round"`
	OneIn      int      `json:"one_in"` // under 2: they all run in every round
}

// RecordRule is how the record changes hands. See checkRecord.
type RecordRule struct {
	Looks  []int              `json:"looks"`
	Alpha  float64            `json:"alpha"`
	Margin map[string]float64 `json:"margin"`
}

type Season struct {
	ID            string                  `json:"id"`
	Title         string                  `json:"title"`
	Digest        string                  `json:"digest"`
	Harness       string                  `json:"harness"`
	RulesVersion  int                     `json:"rules_version"`
	OfficialScale string                  `json:"official_scale"`
	Scales        map[string]Scale        `json:"scales"`
	MinCopyBytes  int                     `json:"min_copy_bytes"`
	MaxDocBytes   int                     `json:"max_doc_bytes"`
	Script        ScriptLimits            `json:"script"`
	ExamSources   []string                `json:"exam_sources"` // the exam's pillars, in the order they are shown
	Exam          ExamPlan                `json:"exam"`         // what the score is made of: see aggregate.go
	Beacon        Beacon                  `json:"beacon"`
	DrawGrid      int64                   `json:"draw_grid_seconds"`
	Pools         map[string]Pool         `json:"pools"`
	GradeDevices  []string                `json:"grade_devices"`
	Reference     map[string]any          `json:"reference"`
	Curves        map[string][][2]float64 `json:"reference_curves"`
	Tolerance     map[string]Tolerance    `json:"tolerance"`
	Noise         map[string]any          `json:"noise"`
	Record        RecordRule              `json:"record"`
	Baselines     BaselineRule            `json:"baselines"`
	References    ReferenceRule           `json:"references"`
	Parity        json.RawMessage         `json:"parity"`
	Train         map[string]float64      `json:"train"`
	Student       map[string]any          `json:"student,omitempty"`
	SFT           map[string]any          `json:"sft,omitempty"`
	Tokenizer     map[string]int          `json:"tokenizer"`
	Assets        AssetLimits             `json:"assets"`
	Machine       Machine                 `json:"machine"`
	Runner        RunnerFloor             `json:"runner"`
	Environment   Environment             `json:"environment"`
}

func loadSeason() *Season {
	var s Season
	if err := json.Unmarshal(seasonJSON, &s); err != nil {
		panic("season.json: " + err.Error())
	}
	return &s
}

// gradesOn reports whether grades from this kind of device count for the leaderboard.
func (s *Season) gradesOn(device string) bool {
	for _, d := range s.GradeDevices {
		if d == device {
			return true
		}
	}
	return false
}

// Crawl identifies the published Common Crawl crawl an official corpus is drawn from.
type Crawl struct {
	Crawl       string `json:"crawl"`
	PathsSHA256 string `json:"paths_sha256"`
	NShards     int    `json:"n_shards"`
}

// officialSeed must match stet.spec.official_seed byte for byte.
func officialSeed(seasonID string, index int, drandRound int64, randomness string, c Crawl) []byte {
	s := fmt.Sprintf("stet|%s|official|%d|%d|%s|%s|%s", seasonID, index, drandRound, randomness, c.Crawl, c.PathsSHA256)
	h := sha256.Sum256([]byte(s))
	return h[:]
}

// prfU64 must match stet.spec.prf_u64.
func prfU64(seed []byte, label string, i int) uint64 {
	h := sha256.New()
	h.Write(seed)
	h.Write([]byte("|" + label + "|" + strconv.Itoa(i)))
	return binary.BigEndian.Uint64(h.Sum(nil)[:8])
}

// sampleWithoutReplacement must match stet.spec.sample_without_replacement.
func sampleWithoutReplacement(seed []byte, label string, n, k int) []int {
	seen := map[int]bool{}
	out := make([]int, 0, k)
	for i := 0; len(out) < k; i++ {
		j := int(prfU64(seed, label, i) % uint64(n))
		if !seen[j] {
			seen[j] = true
			out = append(out, j)
		}
	}
	return out
}

// docOrder must match stet.spec.Round.doc_order.
func docOrder(seed []byte, n int) []int {
	keys := make([]uint64, n)
	out := make([]int, n)
	for i := range out {
		keys[i], out[i] = prfU64(seed, "order", i), i
	}
	sort.SliceStable(out, func(a, b int) bool { return keys[out[a]] < keys[out[b]] })
	return out
}

func roundID(seasonID string, index int) string { return fmt.Sprintf("%s-o%05d", seasonID, index) }

func hexOf(b []byte) string { return hex.EncodeToString(b) }

// drandRoundAt is the first beacon round emitted at or after unix time t.
func (b Beacon) drandRoundAt(t int64) int64 {
	if t <= b.Genesis {
		return 1
	}
	return (t-b.Genesis+b.Period-1)/b.Period + 1
}

func (b Beacon) timeOf(round int64) int64 { return b.Genesis + (round-1)*b.Period }

// drawSlot is the draw time a beacon round belongs to. Official rounds are drawn only at
// whole multiples of the season's grid, from the first beacon round at or after that
// moment, so which beacon rounds can seed a round is public and fixed in advance. Mirrors
// stet.spec.Season.draw_slot, which every verifier checks.
func (s *Season) drawSlot(round int64) (int64, bool) {
	slot := s.Beacon.timeOf(round) / s.DrawGrid * s.DrawGrid
	return slot, s.Beacon.drandRoundAt(slot) == round
}

// maxScriptBlob bounds a script as it is uploaded and stored: compressed. Compression can
// make incompressible data slightly larger, so this is more than the raw limit.
func (s *Season) maxScriptBlob() int64 { return s.Script.MaxBytes*4/3 + (1 << 20) }

// nextGrid is the first grid moment strictly after t.
func (s *Season) nextGrid(t int64) int64 { return (t/s.DrawGrid + 1) * s.DrawGrid }

// unit is what every score is measured in: gains, ratings, tolerances and the record.
const unit = "bits per byte"

// multiplier reads a gain off the season's reference curve: how many times more unedited
// data the baseline would have needed to do as well on the exam. It never extrapolates:
// past either end of the measured curve it returns the end point and capped = true, to be
// shown as "more than". With no curve for the scale (none is measured yet) ok is false, and
// there is no multiplier: the API says null and the pages leave it out. Mirrors
// stet.score.multiplier.
func (s *Season) multiplier(curveKey string, gain float64) (m float64, capped, ok bool) {
	pts := append([][2]float64(nil), s.Curves[curveKey]...)
	if len(pts) < 2 {
		return 0, false, false
	}
	sort.Slice(pts, func(i, j int) bool { return pts[i][0] < pts[j][0] })
	base := math.NaN()
	for _, p := range pts {
		if p[0] == 1 {
			base = p[1]
		}
	}
	if math.IsNaN(base) {
		return 0, false, false
	}
	target := base - gain
	last := len(pts) - 1
	if target >= pts[0][1] {
		return pts[0][0], target > pts[0][1], true
	}
	if target <= pts[last][1] {
		return pts[last][0], target < pts[last][1], true
	}
	for i := 0; i < last; i++ {
		hi, lo := pts[i][1], pts[i+1][1]
		if lo <= target && target <= hi && hi != lo {
			x0, x1 := math.Log2(pts[i][0]), math.Log2(pts[i+1][0])
			return math.Pow(2, x0+(hi-target)/(hi-lo)*(x1-x0)), false, true
		}
	}
	return 0, false, false
}
