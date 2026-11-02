//go:build !linux && !darwin

package main

import "time"

// getSystemUptime has no portable implementation; report 0 rather than
// failing to build on other platforms.
func getSystemUptime() time.Duration { return 0 }
