//go:build !(linux || darwin || freebsd)

package main

import "errors"

func fsStat(string) (dev, total, free uint64, err error) {
	return 0, 0, 0, errors.New("disk usage is not supported on this platform")
}
