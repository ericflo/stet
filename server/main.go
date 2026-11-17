// stet-server: the API and website for Stet.
//
// The process keeps nothing of its own. Records live in a durable-db database, files
// (entries, edit scripts, the exam pack) in a bucket. Either can be swapped for a local
// stand-in for development: memory for the records, a directory for the files.
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
	DBName string
	DBKey  []byte // the durable-db deployment's signing key
	S3     S3Config

	AnchorKeys       map[string]bool
	Scale            string
	RoundInterval    time.Duration
	MaxNewPerRound   int
	LadderSize       int
	QualifyRounds    int // rounds every entry is given
	RatedRounds      int // rounds before an editor is ranked
	SpotRate         float64
	Lease            time.Duration
	AnchorPatience   time.Duration // how long an anchor leaves a grade to volunteers who are around
	FixedCrawl       *Crawl
	RequireGithub    bool
	GithubMinAgeDays int
	Dev              bool
}

type App struct {
	cfg    Config
	st     *Store
	season *Season
	boards boardCache
	site   *Site
	limits *limiter
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
		Addr:             env("STET_ADDR", ":8080"),
		BaseURL:          strings.TrimRight(env("STET_BASE_URL", "http://localhost:8080"), "/"),
		RedirectHosts:    map[string]bool{},
		DataDir:          env("STET_DATA_DIR", "./data"),
		FilesDir:         env("STET_FILES_DIR", "./files"),
		DocsDir:          env("STET_DOCS_DIR", "../docs"),
		AgentsFile:       env("STET_AGENTS_FILE", "../src/stet/templates/AGENTS.md"),
		DBURL:            strings.TrimRight(os.Getenv("STET_DB_URL"), "/"),
		DBName:           env("STET_DB_NAME", "stet"),
		AnchorKeys:       map[string]bool{},
		Scale:            env("STET_SCALE", season.OfficialScale),
		RoundInterval:    envDuration("STET_ROUND_INTERVAL", 6*time.Hour),
		MaxNewPerRound:   envInt("STET_MAX_NEW_PER_ROUND", 8),
		LadderSize:       envInt("STET_LADDER_SIZE", 10),
		QualifyRounds:    envInt("STET_QUALIFY_ROUNDS", 5),
		RatedRounds:      envInt("STET_RATED_ROUNDS", 3),
		SpotRate:         0.25,
		Lease:            envDuration("STET_LEASE", 3*time.Hour),
		AnchorPatience:   envDuration("STET_ANCHOR_PATIENCE", 45*time.Minute),
		RequireGithub:    envBool("STET_REQUIRE_GITHUB", !dev),
		GithubMinAgeDays: envInt("STET_GITHUB_MIN_AGE_DAYS", 30),
		Dev:              dev,
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
	if cfg.Dev {
		return nil
	}
	if iv := int64(cfg.RoundInterval.Seconds()); season.DrawGrid <= 0 || iv < season.DrawGrid || iv%season.DrawGrid != 0 {
		return fmt.Errorf("the round interval must be a multiple of the season's draw grid (%d seconds)", season.DrawGrid)
	}
	if cfg.DBURL == "" {
		return errors.New("STET_DB_URL is not set: outside development the records must live in durable-db")
	}
	return nil
}

// storage opens the two places things are kept.
func (cfg *Config) storage(ctx context.Context) (KV, Blobs, error) {
	var kv KV
	if cfg.DBURL == "" {
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
	st, err := openStore(ctx, kv, blobs, cfg.AnchorKeys)
	if err != nil {
		return nil, err
	}
	a := &App{cfg: cfg, st: st, season: season, limits: newLimiter()}
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
	go func() {
		<-ctx.Done()
		shut, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		srv.Shutdown(shut)
		app.st.flush(shut)
	}()
	slog.Info("listening", "revision", revision, "addr", cfg.Addr, "season", season.ID, "scale", cfg.Scale, "site", cfg.BaseURL, "api", cfg.APIURL,
		"records", kv.Describe(), "files", blobs.Describe(), "anchors", len(cfg.AnchorKeys), "dev", cfg.Dev)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		slog.Error("server", "err", err)
		os.Exit(1)
	}
}
