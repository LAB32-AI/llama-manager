package main

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestLineWriterSplitsLines(t *testing.T) {
	var got []string
	w := &lineWriter{add: func(s string) { got = append(got, s) }}
	for _, chunk := range []string{"alpha\nbe", "ta\r\n\ngam", "ma"} {
		if n, err := w.Write([]byte(chunk)); err != nil || n != len(chunk) {
			t.Fatalf("Write(%q) = %d, %v", chunk, n, err)
		}
	}
	w.Flush()
	want := []string{"alpha", "beta", "", "gamma"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("lines = %q, want %q", got, want)
	}
}

func TestLineWriterTruncatesLongLines(t *testing.T) {
	var got []string
	w := &lineWriter{add: func(s string) { got = append(got, s) }}
	long := strings.Repeat("x", 3*maxLogLine)
	// Feed it in small pieces, as a pipe would.
	data := []byte(long + "\nnext\n")
	for len(data) > 0 {
		n := 1000
		if n > len(data) {
			n = len(data)
		}
		w.Write(data[:n])
		data = data[n:]
	}
	if len(got) != 2 {
		t.Fatalf("got %d lines, want 2", len(got))
	}
	if want := strings.Repeat("x", maxLogLine) + truncatedMarker; got[0] != want {
		t.Fatalf("long line = %d bytes, want %d bytes ending in the truncation marker", len(got[0]), len(want))
	}
	if got[1] != "next" {
		t.Fatalf("line after the long one = %q, want %q", got[1], "next")
	}
}

// TestLongLogLineDoesNotWedgeServer is the P1 #5 regression: bufio.Scanner
// stopped at the first line over 64 KiB, nothing drained the pipe any more, and
// the child blocked forever on its next write.
func TestLongLogLineDoesNotWedgeServer(t *testing.T) {
	old := startingPollInterval
	startingPollInterval = 50 * time.Millisecond
	defer func() { startingPollInterval = old }()

	rig := newFakeRig(t, "long-line", "f1")
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("f1")

	if err := m.StartInstance("f1"); err != nil {
		t.Fatal(err)
	}
	// The fake only starts serving after both writes complete.
	waitFor(t, 3*time.Second, "running after the long line", func() bool { return inst.State() == StateRunning })

	var truncated, after bool
	for _, l := range inst.Logs() {
		truncated = truncated || strings.HasSuffix(l, truncatedMarker)
		after = after || l == "after-long-line"
	}
	if !truncated || !after {
		t.Fatalf("logs: truncated=%v after=%v; want both", truncated, after)
	}
}

// TestCrashReasonCapturedBeforeExit: the final line a crashing server prints
// must be in the log by the time the instance is marked crashed.
func TestCrashReasonCapturedBeforeExit(t *testing.T) {
	dir := t.TempDir()
	bin := dir + "/crash.sh"
	script := "#!/bin/sh\necho 'loading model'\necho 'fatal: out of device memory' >&2\nexit 3\n"
	if err := writeExec(bin, script); err != nil {
		t.Fatal(err)
	}
	rig := newFakeRig(t, "healthy", "f1")
	rig.cfg.ServerBin = bin
	rig.cfg.RestartDelay = duration{10 * time.Second}
	m := NewManager(rig.cfg)
	defer m.Shutdown()
	inst := m.Get("f1")

	for i := 0; i < 20; i++ {
		inst.mu.Lock()
		inst.logs = newRingBuffer(logBufferSize)
		inst.mu.Unlock()
		if err := m.StartInstance("f1"); err != nil {
			t.Fatal(err)
		}
		waitFor(t, 2*time.Second, "crash", func() bool { return inst.State() == StateRestarting })
		logs := inst.Logs()
		if len(logs) == 0 || logs[len(logs)-1] != "fatal: out of device memory" {
			t.Fatalf("round %d: logs at crash time = %q, want the fatal line last", i, logs)
		}
		if err := m.StopInstance("f1"); err != nil {
			t.Fatal(err)
		}
	}
}
