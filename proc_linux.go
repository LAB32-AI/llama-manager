//go:build linux

package main

import "syscall"

// setParentDeathSignal kills the child if the manager dies without stopping it
// (panic, kill -9) when not running under systemd's cgroup cleanup. Pdeathsig
// fires when the forking OS thread exits; that only happens for goroutines that
// exit while locked to a thread, which nothing in this program does.
func setParentDeathSignal(a *syscall.SysProcAttr) {
	a.Pdeathsig = syscall.SIGKILL
}
