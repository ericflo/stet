package main

// Checking an entry's assets when it is submitted.
//
// An editor may name files it needs by reference (model weights): public files on Hugging
// Face, pinned to a commit and named by their sha256. Volunteers download them, so a
// wrong address or a wrong hash would cost every volunteer a download before anyone found
// out. The host says how long each file is and, for a large file, what its sha256 is,
// without sending it. That is asked here, once, before the entry is accepted. A small
// file is kept in git and named by a hash of git's own; those are fetched and hashed.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	maxSmallAsset     = 64 << 20 // the largest file fetched here to learn its hash
	assetCheckTime    = 90 * time.Second
	assetCheckAtATime = 6
)

// hubTransport is how the server reaches the host of the assets. Tests replace it.
var hubTransport http.RoundTripper = http.DefaultTransport

func hubClient(follow bool) *http.Client {
	c := &http.Client{Transport: hubTransport, Timeout: 60 * time.Second}
	if !follow {
		c.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	}
	return c
}

// errHubAway means the host could not be asked: the entry is neither good nor bad.
var errHubAway = errors.New("the host of the assets could not be reached")

func checkAsset(ctx context.Context, as Asset) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, as.URL, nil)
	if err != nil {
		return fmt.Errorf("asset %s: the url is not usable", as.Path)
	}
	req.Header.Set("User-Agent", "stet-server")
	resp, err := hubClient(false).Do(req)
	if err != nil {
		return errHubAway
	}
	resp.Body.Close()
	switch {
	case resp.StatusCode == 401 || resp.StatusCode == 403:
		return fmt.Errorf("asset %s: that file needs an account to download. Assets must be public: volunteers have no login to give", as.Path)
	case resp.StatusCode == 404:
		return fmt.Errorf("asset %s: no such file at that commit", as.Path)
	case resp.StatusCode >= 500 || resp.StatusCode == 429:
		return errHubAway
	case resp.StatusCode >= 400:
		return fmt.Errorf("asset %s: the host answered %d", as.Path, resp.StatusCode)
	}
	size := resp.Header.Get("X-Linked-Size")
	if size == "" && resp.StatusCode == 200 {
		size = resp.Header.Get("Content-Length")
	}
	if n, err := strconv.ParseInt(size, 10, 64); err != nil {
		return errHubAway // it did not say; this is not the entrant's doing
	} else if n != as.Size {
		return fmt.Errorf("asset %s: the file is %d bytes; the entry says %d", as.Path, n, as.Size)
	}
	etag := strings.Trim(strings.TrimPrefix(resp.Header.Get("X-Linked-Etag"), "W/"), `"`)
	if sha256Re.MatchString(etag) {
		if etag != as.SHA256 {
			return fmt.Errorf("asset %s: the file's sha256 is %s; the entry says %s", as.Path, etag, as.SHA256)
		}
		return nil
	}
	// A file kept in git: the host names it by another hash. Small by construction.
	if as.Size > maxSmallAsset {
		return fmt.Errorf("asset %s: the host does not say what this file's sha256 is, and it is too large to fetch and see", as.Path)
	}
	get, err := http.NewRequestWithContext(ctx, http.MethodGet, as.URL, nil)
	if err != nil {
		return errHubAway
	}
	get.Header.Set("User-Agent", "stet-server")
	body, err := hubClient(true).Do(get)
	if err != nil {
		return errHubAway
	}
	defer body.Body.Close()
	if body.StatusCode != 200 {
		return errHubAway
	}
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(body.Body, as.Size+1))
	if err != nil {
		return errHubAway
	}
	if n != as.Size {
		return fmt.Errorf("asset %s: the file is not the %d bytes the entry says", as.Path, as.Size)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != as.SHA256 {
		return fmt.Errorf("asset %s: the file's sha256 is %s; the entry says %s", as.Path, got, as.SHA256)
	}
	return nil
}

// checkAssets asks the host about every asset of an entry. It returns the first thing
// wrong with the entry, or errHubAway if nothing is known to be wrong but not everything
// could be asked.
func checkAssets(ctx context.Context, assets []Asset) error {
	if len(assets) == 0 {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, assetCheckTime)
	defer cancel()
	errs := make([]error, len(assets))
	sem := make(chan struct{}, assetCheckAtATime)
	var wg sync.WaitGroup
	for i, as := range assets {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			errs[i] = checkAsset(ctx, as)
		}()
	}
	wg.Wait()
	away := false
	for _, err := range errs {
		switch {
		case err == nil:
		case errors.Is(err, errHubAway):
			away = true
		default:
			return err
		}
	}
	if away {
		return errHubAway
	}
	return nil
}
