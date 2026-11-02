//go:build unix

package main

import (
	"os"
	"os/exec"
	"syscall"
)

// setProcAttrs puts the child in its own process group so stop signals reach
// everything it spawned (e.g. the real llama-server behind a wrapper script).
func setProcAttrs(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	setParentDeathSignal(cmd.SysProcAttr)
}

// signalGroup signals the child's whole process group. A group outlives its
// leader while other members remain, and POSIX doesn't reuse a pid while its
// process group exists, so this is safe even after the leader was reaped.
func signalGroup(p *os.Process, sig syscall.Signal) {
	if err := syscall.Kill(-p.Pid, sig); err != nil {
		_ = p.Signal(sig)
	}
}

func terminateGroup(p *os.Process) { signalGroup(p, syscall.SIGTERM) }

func killGroup(p *os.Process) { signalGroup(p, syscall.SIGKILL) }
