//go:build linux

package main

import (
	"os/exec"
	"syscall"
	"testing"
)

func TestChildGetsParentDeathSignal(t *testing.T) {
	cmd := exec.Command("true")
	setProcAttrs(cmd)
	if cmd.SysProcAttr.Pdeathsig != syscall.SIGKILL || !cmd.SysProcAttr.Setpgid {
		t.Fatalf("SysProcAttr = %+v, want Setpgid and Pdeathsig=SIGKILL", cmd.SysProcAttr)
	}
}
