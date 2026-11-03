//go:build linux || darwin || freebsd

package main

import "syscall"

// fsStat reports the device a path lives on and its filesystem's size and the
// space an unprivileged process can still write (ext4's root reserve counts
// as used, since downloads can't use it).
func fsStat(path string) (dev, total, free uint64, err error) {
	var st syscall.Stat_t
	if err = syscall.Stat(path, &st); err != nil {
		return
	}
	var fs syscall.Statfs_t
	if err = syscall.Statfs(path, &fs); err != nil {
		return
	}
	bs := uint64(fs.Bsize)
	return uint64(st.Dev), uint64(fs.Blocks) * bs, uint64(fs.Bavail) * bs, nil
}
