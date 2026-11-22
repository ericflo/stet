// stet-server: the API and website for Stet.
//
// The process keeps nothing of its own. Records live in a durable-db database, one for each
// season, and files (entries, edit scripts, the rounds' exams) in a bucket. Either can be
// swapped for a local stand-in for development: memory for the records, a directory for the
// files.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// revision is the commit the binary was built from. The image build sets it.
var revision = "dev"

type Config struct {
	Addr          string
	BaseURL       string          // the website, e.g. https://www.stetrun.com
	APIURL        string          // the API, e.g. https://api.stetrun.com (defaults to BaseURL)
	RedirectHosts map[string]bool // hosts that only redirect to the website
	DataDir       string          // where files go when there is no bucket
	FilesDir      string          // downloads served from disk (the CLI wheel)
	DocsDir       string
	AgentsFile    string // AGENTS.md, which lives with the CLI's templates so that `stet agents` prints the same text

	DBURL  string // durable-db; empty keeps records in memory
	DBName string // the season's own database: "stet-<season>" unless STET_DB_NAME says
	DBKey  []byte // the durable-db deployment's signing key
	S3     S3Config

	AnchorKeys     map[string]bool
	Scale          string
	RoundInterval  time.Duration
	RoundMinGap    time.Duration // a round is drawn this long after the last when an entry is waiting; 0 keeps to the interval alone
	MaxNewPerRound int
	EntriesInPlay  int // entries one person may have waiting or qualifying at once
	SubmitsPerDay  int // submissions one person may make in a day
	LadderSize     int
	QualifyRounds  int // rounds every entry is given
	RatedRounds    int // rounds before an editor is ranked
	Lease          time.Duration
	ReplicaWait    time.Duration // how long a round waits for people who have started, and before giving up what nobody is there to do
	EstablishAfter int           // grades that must have agreed before an account's grades count and it may run editors
	RunsPerEntry   int           // times each entry is run, by different people, before a third is wanted
	ActiveWindow   time.Duration // how recently a verifier must have asked for work to be counted as around
	// EarlyWindow: a machine the organisers rent is handed a grade before the exam only when
	// it would have trained it, and the further grades it could take after it, no sooner
	// than this before the exam is due (pace.go). 0 hands them out as soon as there are any.
	EarlyWindow      time.Duration
	FixedCrawl       *Crawl
	RequireGithub    bool
	GithubMinAgeDays int
	GithubToken      string // optional: lifts GitHub's limit on how often the server may look up a gist
	Dev              bool
	// ArchiveSeed is a copy of a finished season's records (JSON Lines of key and value), to
	// serve its pages from memory for tools/archive_season.py: nothing is scheduled or kept.
	ArchiveSeed string
}

type App struct {
	cfg    Config
	st     *Store
	season *Season
	boards boardCache
	site   *Site
	limits *limiter
	// One token per large request body in memory: entrants' bundles, the runners' scripts and
	// the organisers' uploads (the rounds' exams), each apart from the others.
	uploads, runnerUploads, organiserUploads chan struct{}
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(key)); err == nil {
		return v
	}
	return def
}

func envDuration(key string, def time.Duration) time.Duration {
	if v, err := time.ParseDuration(os.Getenv(key)); err == nil && v > 0 {
		return v
	}
	return def
}

func envBool(key string, def bool) bool {
	switch strings.ToLower(os.Getenv(key)) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	}
	return def
}

func loadConfig(season *Season) (Config, error) {
	dev := os.Getenv("STET_DEV") == "1"
	cfg := Config{
		Addr:          env("STET_ADDR", ":8080"),
		BaseURL:       strings.TrimRight(env("STET_BASE_URL", "http://localhost:8080"), "/"),
		RedirectHosts: map[string]bool{},
		DataDir:       env("STET_DATA_DIR", "./data"),
		FilesDir:      env("STET_FILES_DIR", "./files"),
		DocsDir:       env("STET_DOCS_DIR", "../docs"),
		AgentsFile:    env("STET_AGENTS_FILE", "../src/stet/templates/AGENTS.md"),
		DBURL:         strings.TrimRight(os.Getenv("STET_DB_URL"), "/"),
		DBName:        env("STET_DB_NAME", "stet-"+season.ID), // a new season starts on a database of its own
		AnchorKeys:    map[string]bool{},
		Scale:         env("STET_SCALE", season.OfficialScale),
		RoundInterval: envDuration("STET_ROUND_INTERVAL", 3*time.Hour),
		RoundMinGap:   envDuration("STET_ROUND_MIN_GAP", 0),
		// Running an entry and grading its script take about 12 minutes each at the
		// official scale. A full round is the ladder plus three rounds' worth of newcomers:
		// 17 entries, a little over three hours of one machine doing everything once, inside
		// the six between draws. With two machines every entry is run twice in the same time.
		MaxNewPerRound:   envInt("STET_MAX_NEW_PER_ROUND", 3),
		EntriesInPlay:    envInt("STET_ENTRIES_IN_PLAY", 3),
		SubmitsPerDay:    envInt("STET_SUBMISSIONS_PER_DAY", 6),
		LadderSize:       envInt("STET_LADDER_SIZE", 8),
		QualifyRounds:    envInt("STET_QUALIFY_ROUNDS", 3),
		RatedRounds:      envInt("STET_RATED_ROUNDS", 3),
		Lease:            envDuration("STET_LEASE", 3*time.Hour),
		EarlyWindow:      envDuration("STET_EARLY_WINDOW", 10*time.Minute),
		ReplicaWait:      envDuration("STET_REPLICA_WAIT", 45*time.Minute),
		EstablishAfter:   envInt("STET_ESTABLISH_AFTER", 5),
		RunsPerEntry:     envInt("STET_RUNS_PER_ENTRY", 2),
		ActiveWindow:     envDuration("STET_ACTIVE_WINDOW", 10*time.Minute),
		RequireGithub:    envBool("STET_REQUIRE_GITHUB", !dev),
		GithubMinAgeDays: envInt("STET_GITHUB_MIN_AGE_DAYS", 30),
		GithubToken:      os.Getenv("STET_GITHUB_TOKEN"),
		Dev:              dev,
		ArchiveSeed:      os.Getenv("STET_ARCHIVE_SEED"),
	}
	cfg.APIURL = strings.TrimRight(env("STET_API_URL", cfg.BaseURL), "/")
	for _, h := range strings.Split(os.Getenv("STET_REDIRECT_HOSTS"), ",") {
		if h = strings.ToLower(strings.TrimSpace(h)); h != "" {
			cfg.RedirectHosts[h] = true
		}
	}
	for _, k := range strings.Split(os.Getenv("STET_ANCHOR_KEYS"), ",") {
		if k = strings.ToLower(strings.TrimSpace(k)); k != "" {
			cfg.AnchorKeys[k] = true
		}
	}
	if v := os.Getenv("STET_CRAWL"); v != "" { // "CC-MAIN-2026-43:<paths sha256>:<files>": pin the crawl instead of discovering it
		parts := strings.Split(v, ":")
		n := 0
		if len(parts) == 3 {
			n, _ = strconv.Atoi(parts[2])
		}
		if n <= 0 {
			return cfg, errors.New("STET_CRAWL must be <crawl id>:<paths sha256>:<number of files>")
		}
		cfg.FixedCrawl = &Crawl{Crawl: parts[0], PathsSHA256: parts[1], NShards: n}
	}
	if cfg.DBURL != "" {
		text := os.Getenv("STET_DB_KEY")
		if file := os.Getenv("STET_DB_KEY_FILE"); file != "" {
			raw, err := os.ReadFile(file)
			if err != nil {
				return cfg, fmt.Errorf("STET_DB_KEY_FILE: %w", err)
			}
			text = string(raw)
		}
		key, err := parseDBKey(text)
		if err != nil {
			return cfg, err
		}
		cfg.DBKey = key
	}
	cfg.S3 = S3Config{
		Endpoint: os.Getenv("STET_S3_ENDPOINT"), Region: env("STET_S3_REGION", "us-west-002"), Bucket: os.Getenv("STET_S3_BUCKET"),
		Prefix: os.Getenv("STET_S3_PREFIX"), KeyID: os.Getenv("STET_S3_ACCESS_KEY_ID"), Secret: os.Getenv("STET_S3_SECRET_ACCESS_KEY"),
	}
	if cfg.S3.Bucket != "" && (cfg.S3.Endpoint == "" || cfg.S3.KeyID == "" || cfg.S3.Secret == "") {
		return cfg, errors.New("STET_S3_BUCKET is set, so STET_S3_ENDPOINT, STET_S3_ACCESS_KEY_ID and STET_S3_SECRET_ACCESS_KEY must be too")
	}
	return cfg, cfg.check(season)
}

// check refuses settings that would make rounds no verifier accepts.
func (cfg *Config) check(season *Season) error {
	if _, ok := season.Scales[cfg.Scale]; !ok {
		return errors.New("unknown scale " + cfg.Scale)
	}
	if cfg.ArchiveSeed != "" {
		if cfg.DBURL != "" || cfg.S3.Bucket != "" {
			return errors.New("STET_ARCHIVE_SEED serves a copy: unset STET_DB_URL and STET_S3_BUCKET, so that nothing real is written")
		}
		return nil
	}
	if cfg.Dev {
		return nil
	}
	if iv := int64(cfg.RoundInterval.Seconds()); season.DrawGrid <= 0 || iv < season.DrawGrid || iv%season.DrawGrid != 0 {
		return fmt.Errorf("the round interval must be a multiple of the season's draw grid (%d seconds)", season.DrawGrid)
	}
	if cfg.DBURL == "" {
		return errors.New("STET_DB_URL is not set: outside development the records must live in durable-db")
	}
	if cfg.S3.Bucket == "" {
		return errors.New("STET_S3_BUCKET is not set: outside development the files must live in a bucket, not in the container")
	}
	return nil
}

// storage opens the two places things are kept.
func (cfg *Config) storage(ctx context.Context) (KV, Blobs, error) {
	var kv KV
	if cfg.ArchiveSeed != "" {
		m, err := seededMemKV(cfg.ArchiveSeed)
		if err != nil {
			return nil, nil, fmt.Errorf("STET_ARCHIVE_SEED: %w", err)
		}
		frozen = true
		kv = m
	} else if cfg.DBURL == "" {
		slog.Warn("STET_DB_URL is not set: records are kept in memory and lost when the server stops")
		kv = newMemKV()
	} else {
		db := newDurableDB(cfg.DBURL, cfg.DBName, cfg.DBKey)
		if err := db.ensure(ctx); err != nil {
			return nil, nil, fmt.Errorf("durable-db: %w", err)
		}
		kv = db
	}
	var blobs Blobs
	if cfg.S3.Bucket == "" {
		blobs = dirBlobs{filepath.Join(cfg.DataDir, "blobs")}
	} else {
		blobs = newS3Blobs(cfg.S3)
	}
	return kv, blobs, nil
}

func newApp(ctx context.Context, cfg Config, season *Season, kv KV, blobs Blobs) (*App, error) {
	if _, ok := season.Scales[cfg.Scale]; !ok {
		return nil, errors.New("unknown scale " + cfg.Scale)
	}
	if cfg.APIURL == "" {
		cfg.APIURL = cfg.BaseURL
	}
	if cfg.RunsPerEntry < 1 {
		cfg.RunsPerEntry = 2
	}
	if cfg.EntriesInPlay < 1 {
		cfg.EntriesInPlay = 1
	}
	if cfg.SubmitsPerDay < 1 {
		cfg.SubmitsPerDay = 6
	}
	if cfg.ActiveWindow <= 0 {
		cfg.ActiveWindow = 10 * time.Minute
	}
	if cfg.EstablishAfter < 1 { // standing is always earned
		cfg.EstablishAfter = 5
	}
	st, err := openStore(ctx, kv, blobs, cfg.AnchorKeys)
	if err != nil {
		return nil, err
	}
	a := &App{cfg: cfg, st: st, season: season, limits: newLimiter(), uploads: make(chan struct{}, 4), runnerUploads: make(chan struct{}, 4), organiserUploads: make(chan struct{}, 4)}
	githubToken = cfg.GithubToken
	a.site, err = newSite(a)
	return a, err
}

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))
	season := loadSeason()
	cfg, err := loadConfig(season)
	if err != nil {
		slog.Error("configuration", "err", err)
		os.Exit(1)
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	kv, blobs, err := cfg.storage(ctx)
	if err != nil {
		slog.Error("storage", "err", err)
		os.Exit(1)
	}
	app, err := newApp(ctx, cfg, season, kv, blobs)
	if err != nil {
		slog.Error("startup", "err", err)
		os.Exit(1)
	}
	go app.runScheduler(ctx)

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           app.routes(),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       5 * time.Minute, // bundles can be tens of megabytes on a slow link
		WriteTimeout:      30 * time.Minute,
		IdleTimeout:       2 * time.Minute,
	}
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		<-ctx.Done()
		if !cfg.Dev {
			// The proxy is told this server is going at the same moment the server is. Keep
			// answering for as long as it may still be sending requests here.
			time.Sleep(5 * time.Second)
		}
		shut, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		srv.Shutdown(shut)
		app.st.flush(shut)
	}()
	slog.Info("listening", "revision", revision, "addr", cfg.Addr, "season", season.ID, "scale", cfg.Scale, "site", cfg.BaseURL, "api", cfg.APIURL,
		"records", kv.Describe(), "files", blobs.Describe(), "organisers", len(cfg.AnchorKeys), "dev", cfg.Dev)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		slog.Error("server", "err", err)
		os.Exit(1)
	}
	<-stopped // ListenAndServe returns when shutdown starts; requests in flight are still finishing
}
