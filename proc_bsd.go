//go:build unix && !linux

package main

import "syscall"

func setParentDeathSignal(*syscall.SysProcAttr) {}
