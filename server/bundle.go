package main

import (
	"archive/tar"
	"bytes"
	"errors"
	"fmt"
	"io"
	"path"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/BurntSushi/toml"
	"github.com/klauspost/compress/zstd"
)

const (
	maxBundleBytes   = 50 << 20
	maxUnpackedBytes = 400 << 20
	maxBundleFiles   = 2000
	maxBundlePath    = 255
	maxBundlePaths   = 64 << 10 // all the paths of a bundle together; they are kept with its record
	maxManifestBytes = 64 << 10
	maxBundleWindow  = 32 << 20 // `stet submit` compresses with a window of 4 MB
	maxViewBytes     = 256 << 10
)

var (
	editorNameRe = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$`)
	sha256Re     = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

type Manifest struct {
	Name        string           `toml:"name"`
	Entry       string           `toml:"entry"`
	Description string           `toml:"description"`
	Technique   string           `toml:"technique"`
	Parent      string           `toml:"parent"`
	License     string           `toml:"license"`
	Assets      []map[string]any `toml:"assets"`
}

type BundleFile struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
}

// untar opens a bundle. rest is the decompressed stream the tar is read from, cut off one
// byte past the limit, so that a caller can tell a frame that ends from one that does not.
func untar(blob []byte) (tr *tar.Reader, rest *io.LimitedReader, done func(), err error) {
	dec, err := zstd.NewReader(bytes.NewReader(blob), zstd.WithDecoderMaxMemory(maxUnpackedBytes),
		zstd.WithDecoderMaxWindow(maxBundleWindow), zstd.WithDecoderConcurrency(1))
	if err != nil {
		return nil, nil, nil, err
	}
	rest = &io.LimitedReader{R: dec, N: maxUnpackedBytes + 1}
	return tar.NewReader(rest), rest, dec.Close, nil
}

func cleanBundlePath(name string) (string, bool) {
	p := path.Clean(name)
	if p == "." || strings.HasPrefix(p, "/") || strings.HasPrefix(p, "../") || p == ".." || strings.ContainsRune(p, 0) {
		return "", false
	}
	return p, true
}

// leftOut says why a path never belongs in a bundle, or "" if it may. It mirrors
// skip_reason in src/stet/project.py.
func leftOut(p string) string {
	parts := strings.Split(p, "/")
	for i, part := range parts {
		if strings.HasPrefix(part, ".") {
			return "is a hidden file"
		}
		if i < len(parts)-1 && (part == "venv" || part == "__pycache__" || part == "node_modules" || part == "runs") {
			return "is under " + part + "/"
		}
	}
	if strings.HasSuffix(p, ".pyc") || strings.HasSuffix(p, ".pyo") {
		return "is compiled Python"
	}
	return ""
}

// parseBundle validates an uploaded bundle and returns its manifest and file list.
func parseBundle(blob []byte) (*Manifest, []BundleFile, error) {
	if len(blob) > maxBundleBytes {
		return nil, nil, fmt.Errorf("the bundle is %d MB; the limit is %d MB", len(blob)>>20, maxBundleBytes>>20)
	}
	tr, rest, done, err := untar(blob)
	if err != nil {
		return nil, nil, errors.New("the bundle is not a zstd-compressed tar; submit with `stet submit`")
	}
	defer done()
	var files []BundleFile
	var manifestRaw []byte
	var total int64
	names, seen := 0, map[string]bool{}
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, errors.New("the bundle is damaged or too large when unpacked")
		}
		p, ok := cleanBundlePath(h.Name)
		if !ok || h.Typeflag != tar.TypeReg || len(p) > maxBundlePath {
			return nil, nil, fmt.Errorf("the bundle contains an entry that is not a plain file with a path of at most %d bytes: %q", maxBundlePath, clip(h.Name, 80))
		}
		// Each file once, under the one name it has. Otherwise the copy shown on the site
		// and the copy an anchor runs could be different files.
		if p != h.Name || seen[p] {
			return nil, nil, fmt.Errorf("the bundle names %q twice, or by a roundabout path; submit with `stet submit`", clip(h.Name, 80))
		}
		seen[p] = true
		if names += len(p); names > maxBundlePaths {
			return nil, nil, fmt.Errorf("the bundle's file names add up to more than %d KB", maxBundlePaths>>10)
		}
		// A sparse file says it is larger than the bytes that follow it. Whoever unpacks the
		// bundle would write the difference as zeros.
		for k := range h.PAXRecords {
			if strings.HasPrefix(k, "GNU.sparse.") {
				return nil, nil, fmt.Errorf("the bundle contains a sparse file: %q", clip(h.Name, 80))
			}
		}
		// `stet submit` leaves these at home, and an anchor will not unpack them. A bundle
		// that holds one was not made by it, and would not run as its author expects.
		if why := leftOut(p); why != "" {
			return nil, nil, fmt.Errorf("the bundle contains %q, which %s; submit with `stet submit`", clip(p, 80), why)
		}
		if total += h.Size; h.Size < 0 || total > maxUnpackedBytes {
			return nil, nil, fmt.Errorf("the bundle is over %d MB when unpacked", maxUnpackedBytes>>20)
		}
		if len(files) >= maxBundleFiles {
			return nil, nil, fmt.Errorf("the bundle has more than %d files", maxBundleFiles)
		}
		files = append(files, BundleFile{Path: p, Size: h.Size})
		if p == "stet.toml" {
			if h.Size > maxManifestBytes {
				return nil, nil, fmt.Errorf("stet.toml is over %d KB; it holds a few short fields", maxManifestBytes>>10)
			}
			manifestRaw, err = io.ReadAll(io.LimitReader(tr, maxManifestBytes))
			if err != nil {
				return nil, nil, errors.New("could not read stet.toml")
			}
		}
	}
	// The tar ends before the compressed frame has to. Whatever follows would still be
	// decompressed by anyone unpacking the bundle, so it counts.
	if io.Copy(io.Discard, rest); rest.N <= 0 {
		return nil, nil, fmt.Errorf("the bundle is over %d MB when unpacked", maxUnpackedBytes>>20)
	}
	if manifestRaw == nil {
		return nil, nil, errors.New("the bundle has no stet.toml")
	}
	// The decoder matches field names without regard to case, so `Name` and `name` in one
	// file would both be "name", and which won would depend on map order. Field names are
	// lowercase, and anything that only looks like one is refused.
	var keys map[string]any
	if _, err := toml.Decode(string(manifestRaw), &keys); err != nil {
		return nil, nil, fmt.Errorf("stet.toml is not valid TOML: %v", err)
	}
	for k := range keys {
		for _, field := range []string{"name", "entry", "description", "technique", "parent", "license", "assets"} {
			if k != field && strings.EqualFold(k, field) {
				return nil, nil, fmt.Errorf("stet.toml: the key %q must be written %s", clip(k, 40), field)
			}
		}
	}
	var m Manifest
	if _, err := toml.Decode(string(manifestRaw), &m); err != nil {
		return nil, nil, fmt.Errorf("stet.toml is not valid TOML: %v", err)
	}
	m.Name = strings.TrimSpace(m.Name)
	if !editorNameRe.MatchString(m.Name) {
		return nil, nil, errors.New("stet.toml: name must be 3-40 characters of lowercase letters, digits and hyphens")
	}
	if m.Entry = strings.TrimSpace(m.Entry); m.Entry == "" || len(m.Entry) > 200 || strings.ContainsAny(m.Entry, "\x00\n\r") {
		return nil, nil, errors.New("stet.toml: entry is required, and is one command line of at most 200 characters")
	}
	m.Description = clip(strings.TrimSpace(m.Description), 300)
	m.Technique = clip(strings.TrimSpace(m.Technique), 120)
	// These two are shown on the site and printed in terminals, in other people's.
	for _, text := range []string{m.Description, m.Technique} {
		for _, r := range text {
			if !unicode.IsPrint(r) {
				return nil, nil, errors.New("stet.toml: description and technique are one line of ordinary text each")
			}
		}
	}
	m.Parent = strings.ToLower(strings.TrimSpace(m.Parent))
	if m.Parent != "" && !sha256Re.MatchString(m.Parent) {
		m.Parent = ""
	}
	if len(m.Assets) > 0 {
		return nil, nil, errors.New("stet.toml: [[assets]] are not supported this season. Put what the editor needs inside the bundle (up to 50 MB)")
	}
	return &m, files, nil
}

// bundleFile returns one file's text from a bundle, for the public source viewer.
func bundleFile(blob []byte, want string) (text string, binary, truncated bool, err error) {
	tr, _, done, err := untar(blob)
	if err != nil {
		return "", false, false, err
	}
	defer done()
	for {
		h, err := tr.Next()
		if err != nil {
			return "", false, false, errors.New("no such file in this editor")
		}
		if p, ok := cleanBundlePath(h.Name); !ok || p != want {
			continue
		}
		b, _ := io.ReadAll(io.LimitReader(tr, maxViewBytes+1))
		if len(b) > maxViewBytes {
			b, truncated = b[:maxViewBytes], true
		}
		if !utf8.Valid(b) && !truncated || bytes.IndexByte(b, 0) >= 0 {
			return "", true, false, nil
		}
		return strings.ToValidUTF8(string(b), "�"), false, truncated, nil
	}
}

func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}
