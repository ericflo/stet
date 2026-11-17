package main

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"strings"
)

const maxClockSkew = 300 // seconds

var handleRe = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{1,22}[a-z0-9]$`)

var reservedHandles = map[string]bool{
	"stet": true, "admin": true, "anchor": true, "baseline": true, "identity": true, "api": true,
	"www": true, "root": true, "system": true, "official": true, "maintainer": true, "anonymous": true,
}

// claimedKey reads who a request says it is from, out of its headers alone: the key (hex,
// lower case) once the three signing headers are well formed and the timestamp is fresh.
// It proves nothing yet. It is what lets the server refuse a stranger before reading
// whatever they are uploading.
func claimedKey(r *http.Request) (string, error) {
	keyHex, sigHex, ts := r.Header.Get("X-Stet-Key"), r.Header.Get("X-Stet-Sig"), r.Header.Get("X-Stet-Time")
	if keyHex == "" || sigHex == "" || ts == "" {
		return "", errors.New("this request must be signed; run `stet login`")
	}
	if key, err := hex.DecodeString(keyHex); err != nil || len(key) != ed25519.PublicKeySize {
		return "", errors.New("malformed key")
	}
	if sig, err := hex.DecodeString(sigHex); err != nil || len(sig) != ed25519.SignatureSize {
		return "", errors.New("malformed signature")
	}
	t, err := strconv.ParseInt(ts, 10, 64)
	if err != nil {
		return "", errors.New("malformed timestamp")
	}
	if d := now() - t; d > maxClockSkew || d < -maxClockSkew {
		return "", fmt.Errorf("the request's timestamp is %d seconds from the server's clock; check your clock", d)
	}
	return strings.ToLower(keyHex), nil
}

// verifySignature checks the request's Ed25519 signature and returns the signing key (hex).
// The signed message is "stet-v1\nMETHOD\nPATH?QUERY\nTIME\nSHA256(body)".
func verifySignature(r *http.Request, body []byte) (string, error) {
	return verifySigned(r, sha256Hex(body))
}

// verifySigned is verifySignature for a body known only by its hash. A client sends that
// hash in X-Stet-Body, so that an upload's signature can be checked before the server
// reads any of it; whoever does this must then check that the body has that hash.
func verifySigned(r *http.Request, bodySum string) (string, error) {
	keyHex, err := claimedKey(r)
	if err != nil {
		return "", err
	}
	key, _ := hex.DecodeString(keyHex)
	sig, _ := hex.DecodeString(r.Header.Get("X-Stet-Sig"))
	ts := r.Header.Get("X-Stet-Time")
	msg := strings.Join([]string{"stet-v1", r.Method, r.URL.RequestURI(), ts, bodySum}, "\n")
	if !ed25519.Verify(key, []byte(msg), sig) {
		return "", errors.New("bad signature")
	}
	return keyHex, nil
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
