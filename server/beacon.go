package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

var httpClient = &http.Client{Timeout: 60 * time.Second}

const userAgent = "stet-server/0.1"

// errBusy is an upstream service saying the server has asked too often.
var errBusy = errors.New("rate limited")

func httpGet(ctx context.Context, url string, limit int64) ([]byte, error) {
	return httpGetAs(ctx, url, "", limit)
}

func httpGetAs(ctx context.Context, url, bearer string, limit int64) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == 429 || (resp.StatusCode == 403 && resp.Header.Get("X-RateLimit-Remaining") == "0") {
		return nil, errBusy
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("%s: HTTP %d", url, resp.StatusCode)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, fmt.Errorf("%s: response too large", url)
	}
	return b, nil
}

// fetchBeacon gets one drand round, checking that the randomness is the hash of the
// signature. The server does not verify the BLS signature itself; clients check the same
// value against drand independently, so a wrong value here would be caught by every
// verifier before any work was done.
func fetchBeacon(ctx context.Context, b Beacon, round int64) (string, error) {
	var last error
	for _, relay := range b.Relays {
		raw, err := httpGet(ctx, fmt.Sprintf("%s/%s/public/%d", relay, b.ChainHash, round), 1<<16)
		if err != nil {
			last = err
			continue
		}
		var got struct {
			Round      int64  `json:"round"`
			Randomness string `json:"randomness"`
			Signature  string `json:"signature"`
		}
		if err := json.Unmarshal(raw, &got); err != nil {
			last = err
			continue
		}
		sig, err := hex.DecodeString(got.Signature)
		if err != nil {
			last = err
			continue
		}
		sum := sha256.Sum256(sig)
		if got.Round != round || hex.EncodeToString(sum[:]) != strings.ToLower(got.Randomness) {
			last = errors.New("inconsistent beacon from " + relay)
			continue
		}
		return strings.ToLower(got.Randomness), nil
	}
	if last == nil {
		last = errors.New("no drand relays configured")
	}
	return "", last
}

// discoverCrawl finds the newest published Common Crawl crawl and fingerprints its list of
// WET files. Official rounds draw their corpus from it.
func discoverCrawl(ctx context.Context, pool Pool) (*Crawl, error) {
	raw, err := httpGet(ctx, "https://index.commoncrawl.org/collinfo.json", 4<<20)
	if err != nil {
		return nil, err
	}
	var infos []struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(raw, &infos); err != nil || len(infos) == 0 {
		return nil, errors.New("could not read Common Crawl's crawl list")
	}
	return fingerprintCrawl(ctx, pool, infos[0].ID)
}

var crawlIDRe = regexp.MustCompile(`^CC-MAIN-\d{4}-\d{2}$`)

func fingerprintCrawl(ctx context.Context, pool Pool, id string) (*Crawl, error) {
	if !crawlIDRe.MatchString(id) { // the same pattern every verifier holds a round to
		return nil, fmt.Errorf("unexpected crawl id %q", id)
	}
	listing, err := httpGet(ctx, pool.BaseURL+strings.ReplaceAll(pool.Paths, "{crawl}", id), 64<<20)
	if err != nil {
		return nil, err
	}
	zr, err := gzip.NewReader(bytes.NewReader(listing))
	if err != nil {
		return nil, err
	}
	plain, err := io.ReadAll(io.LimitReader(zr, 256<<20))
	if err != nil {
		return nil, err
	}
	n := len(strings.Fields(string(plain)))
	if n < 1000 {
		return nil, fmt.Errorf("crawl %s lists only %d files; not adopting it", id, n)
	}
	sum := sha256.Sum256(listing)
	return &Crawl{Crawl: id, PathsSHA256: hex.EncodeToString(sum[:]), NShards: n}, nil
}
