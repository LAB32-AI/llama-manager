//go:build linux

package main

import (
	"syscall"
	"time"
)

func getSystemUptime() time.Duration {
	var info syscall.Sysinfo_t
	if err := syscall.Sysinfo(&info); err != nil {
		return 0
	}
	return time.Duration(info.Uptime) * time.Second
}
