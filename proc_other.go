//go:build !unix

package main

import (
	"os"
	"os/exec"
)

func setProcAttrs(*exec.Cmd) {}

func terminateGroup(p *os.Process) { _ = p.Kill() }

func killGroup(p *os.Process) { _ = p.Kill() }
