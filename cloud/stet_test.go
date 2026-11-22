package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// A key signs exactly as the stet command line does: these are what src/stet/keys.py signs
// for a key whose seed is the bytes 0 to 31, at one moment.
func TestAKeySignsAsTheStetCommandLineDoes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "key.json")
	seed := make([]byte, 32)
	for i := range seed {
		seed[i] = byte(i)
	}
	// As `stet login` writes it.
	raw := `{"private": "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", "public": "03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8", "handle": "stet-rp-01"}`
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	k, err := loadKey(path)
	if err != nil {
		t.Fatal(err)
	}
	at := time.Unix(1795000000, 0)
	for _, c := range []struct{ method, path, body, sig, sum string }{
		{"GET", "/api/v1/admin/work", "", "9685b30d5637c2ae818ed3cf9d0af9f710ee1170ba89ecae49f89e7c0ebaf041634aa069f671a35d62e025c32706b442ae4089ec95fb6faafd7d839b360d6603",
			"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{"POST", "/api/v1/accounts", `{"handle":"stet-rp-01"}`, "25918ae9af995d0eab7b3ed7907feab3bae4a677b7185a0719d00cd76f970a015cb66abcb7c4e3060acb712cb76b9e96370aeb4640e55fe88ad45ac1a890dd00",
			"5d40436d6508ca081ceb53b1a5e252a1b705ded9c0c7a06e0a364861bbc6cb11"},
	} {
		h := k.sign(c.method, c.path, []byte(c.body), at)
		if h.Get("X-Stet-Key") != k.Public || h.Get("X-Stet-Time") != "1795000000" || h.Get("X-Stet-Sig") != c.sig || h.Get("X-Stet-Body") != c.sum {
			t.Fatalf("%s %s signed as %v", c.method, c.path, h)
		}
	}

	// A key this tool makes is one stet can read back: the same three fields.
	made, err := newKey("stet-rp-02")
	if err != nil {
		t.Fatal(err)
	}
	if err := made.save(filepath.Join(dir, "made.json")); err != nil {
		t.Fatal(err)
	}
	again, err := loadKey(filepath.Join(dir, "made.json"))
	if err != nil || again.Public != made.Public || again.Handle != "stet-rp-02" {
		t.Fatalf("a saved key read back as %+v (%v)", again, err)
	}
	if info, _ := os.Stat(filepath.Join(dir, "made.json")); info.Mode().Perm() != 0o600 {
		t.Fatalf("a key file is readable by others: %v", info.Mode())
	}

	// A file whose two halves do not belong together is refused.
	if err := os.WriteFile(path, []byte(`{"private": "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", "public": "`+made.Public+`"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadKey(path); err == nil {
		t.Fatal("a key whose public half is another's was taken")
	}
}
