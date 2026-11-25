package main

import (
	"bytes"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"testing"
)

// The tests never read the markets' real keys, nor stet-runpod's real pool: the keys file and
// stet-runpod's folder are an empty folder's, and no market's key is in the environment.
func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "stet-cloud-test-")
	if err != nil {
		panic(err)
	}
	os.Setenv("STET_CLOUD_CREDENTIALS", filepath.Join(dir, "credentials.env"))
	os.Setenv("STET_RUNPOD_HOME", filepath.Join(dir, "stet-runpod"))
	os.Setenv("STET_CLOUD_HOME", filepath.Join(dir, "stet-cloud"))
	for _, k := range credentialNames {
		os.Unsetenv(k)
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// Every command's settings can be read: none is defined twice.
func TestEveryCommandsSettingsCanBeRead(t *testing.T) {
	for _, cmd := range []string{"keys", "offers", "gpus", "markets", "migrate", "status", "scale", "up", "down", "pause", "resume", "logs", "ssh"} {
		var out, errs bytes.Buffer
		if err := run([]string{cmd, "-h"}, &out, &errs); err != nil && !errors.Is(err, flag.ErrHelp) {
			t.Errorf("%s -h: %v", cmd, err)
		}
	}
}

// Season 3 default CLI flags match the documented fleet cap, disk floor, idle exit, and VRAM tiers.
func TestSeason3FlagDefaults(t *testing.T) {
	var o Options
	fs := flags("scale", &o)
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	if o.MaxWorkers != 6 || o.MaxHourly != 4 || o.Idle != 20 || o.Disk != 120 || o.MinGPUMemory != 16 || o.MinCuratorGPUMemory != 24 {
		t.Fatalf("unexpected defaults: max=%d max-hourly=%.1f idle=%d disk=%d min-gpu-memory=%d min-curator-gpu-memory=%d",
			o.MaxWorkers, o.MaxHourly, o.Idle, o.Disk, o.MinGPUMemory, o.MinCuratorGPUMemory)
	}
}
