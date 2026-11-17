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
	"unicode/utf8"

	"github.com/BurntSushi/toml"
	"github.com/klauspost/compress/zstd"
)

const (
	maxBundleBytes   = 50 << 20
	maxUnpackedBytes = 400 << 20
	maxBundleFiles   = 2000
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

func untar(blob []byte) (*tar.Reader, func(), error) {
	dec, err := zstd.NewReader(bytes.NewReader(blob), zstd.WithDecoderMaxMemory(maxUnpackedBytes), zstd.WithDecoderConcurrency(1))
	if err != nil {
		return nil, nil, err
	}
	return tar.NewReader(io.LimitReader(dec, maxUnpackedBytes)), dec.Close, nil
}

func cleanBundlePath(name string) (string, bool) {
	p := path.Clean(name)
	if p == "." || strings.HasPrefix(p, "/") || strings.HasPrefix(p, "../") || p == ".." || strings.ContainsRune(p, 0) {
		return "", false
	}
	return p, true
}

// parseBundle validates an uploaded bundle and returns its manifest and file list.
func parseBundle(blob []byte) (*Manifest, []BundleFile, error) {
	if len(blob) > maxBundleBytes {
		return nil, nil, fmt.Errorf("the bundle is %d MB; the limit is %d MB", len(blob)>>20, maxBundleBytes>>20)
	}
	tr, done, err := untar(blob)
	if err != nil {
		return nil, nil, errors.New("the bundle is not a zstd-compressed tar; submit with `stet submit`")
	}
	defer done()
	var files []BundleFile
	var manifestRaw []byte
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, errors.New("the bundle is damaged or too large when unpacked")
		}
		p, ok := cleanBundlePath(h.Name)
		if !ok || h.Typeflag != tar.TypeReg {
			return nil, nil, fmt.Errorf("the bundle contains an entry that is not a plain file: %q", h.Name)
		}
		if len(files) >= maxBundleFiles {
			return nil, nil, fmt.Errorf("the bundle has more than %d files", maxBundleFiles)
		}
		files = append(files, BundleFile{Path: p, Size: h.Size})
		if p == "stet.toml" {
			manifestRaw, err = io.ReadAll(io.LimitReader(tr, 1<<20))
			if err != nil {
				return nil, nil, errors.New("could not read stet.toml")
			}
		}
	}
	if manifestRaw == nil {
		return nil, nil, errors.New("the bundle has no stet.toml")
	}
	var m Manifest
	if _, err := toml.Decode(string(manifestRaw), &m); err != nil {
		return nil, nil, fmt.Errorf("stet.toml is not valid TOML: %v", err)
	}
	m.Name = strings.TrimSpace(m.Name)
	if !editorNameRe.MatchString(m.Name) {
		return nil, nil, errors.New("stet.toml: name must be 3-40 characters of lowercase letters, digits and hyphens")
	}
	if m.Entry = strings.TrimSpace(m.Entry); m.Entry == "" {
		return nil, nil, errors.New("stet.toml: entry is required")
	}
	m.Description = clip(strings.TrimSpace(m.Description), 300)
	m.Technique = clip(strings.TrimSpace(m.Technique), 120)
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
	tr, done, err := untar(blob)
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
