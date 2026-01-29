//go:build darwin

package main

import (
	"os/exec"
	"strconv"
	"strings"
	"time"
)

func getSystemUptime() time.Duration {
	out, err := exec.Command("sysctl", "-n", "kern.boottime").Output()
	if err != nil {
		return 0
	}
	s := string(out)
	if idx := strings.Index(s, "sec = "); idx >= 0 {
		s = s[idx+6:]
		if end := strings.Index(s, ","); end >= 0 {
			s = s[:end]
		}
		bootSec, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
		if err != nil {
			return 0
		}
		return time.Since(time.Unix(bootSec, 0))
	}
	return 0
}
