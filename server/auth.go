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

// verifySignature checks the request's Ed25519 signature and returns the signing key (hex).
// The signed message is "stet-v1\nMETHOD\nPATH?QUERY\nTIME\nSHA256(body)".
func verifySignature(r *http.Request, body []byte) (string, error) {
	keyHex, sigHex, ts := r.Header.Get("X-Stet-Key"), r.Header.Get("X-Stet-Sig"), r.Header.Get("X-Stet-Time")
	if keyHex == "" || sigHex == "" || ts == "" {
		return "", errors.New("this request must be signed; run `stet login`")
	}
	key, err := hex.DecodeString(keyHex)
	if err != nil || len(key) != ed25519.PublicKeySize {
		return "", errors.New("malformed key")
	}
	sig, err := hex.DecodeString(sigHex)
	if err != nil || len(sig) != ed25519.SignatureSize {
		return "", errors.New("malformed signature")
	}
	t, err := strconv.ParseInt(ts, 10, 64)
	if err != nil {
		return "", errors.New("malformed timestamp")
	}
	if d := now() - t; d > maxClockSkew || d < -maxClockSkew {
		return "", fmt.Errorf("the request's timestamp is %d seconds from the server's clock; check your clock", d)
	}
	sum := sha256.Sum256(body)
	msg := strings.Join([]string{"stet-v1", r.Method, r.URL.RequestURI(), ts, hex.EncodeToString(sum[:])}, "\n")
	if !ed25519.Verify(key, []byte(msg), sig) {
		return "", errors.New("bad signature")
	}
	return strings.ToLower(keyHex), nil
}
