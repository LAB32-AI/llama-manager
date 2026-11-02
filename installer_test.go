//go:build unix

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// installerRig runs service_install.sh from a scratch copy of the repo with
// stub id/whoami/systemctl/go/sudo/chown, installing into temp dirs.
type installerRig struct {
	repo, bin, conf, unit, log string
	env                        []string
}

func newInstallerRig(t *testing.T) *installerRig {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir()) // the script reports resolved paths (/var -> /private/var on macOS)
	if err != nil {
		t.Fatal(err)
	}
	r := &installerRig{
		repo: filepath.Join(root, "repo"), bin: filepath.Join(root, "bin"),
		conf: filepath.Join(root, "etc"), unit: filepath.Join(root, "systemd"),
		log: filepath.Join(root, "calls.log"),
	}
	for _, f := range []string{"service_install.sh", "llama-manager.service", "example.config.yaml"} {
		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		mustWriteFile(t, filepath.Join(r.repo, f), 0)
		if err := os.WriteFile(filepath.Join(r.repo, f), data, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	shims := filepath.Join(root, "shims")
	stub := func(name, body string) {
		if err := writeExec(filepath.Join(shims, name), "#!/bin/sh\n"+body); err != nil {
			t.Fatal(err)
		}
	}
	mustWriteFile(t, filepath.Join(shims, ".keep"), 0)
	stub("id", `case "$1" in -u) echo "$FAKE_UID";; -gn) echo testgroup;; *) [ "$1" = testuser ];; esac`)
	stub("whoami", `echo "$FAKE_WHOAMI"`)
	stub("systemctl", `echo "systemctl $*" >> "$LOG"; [ "$1" = status ] && exit 3; exit 0`)
	stub("go", `echo "go $* in $(pwd)" >> "$LOG"; printf fresh > llama-manager`)
	stub("chown", `echo "chown $*" >> "$LOG"`)
	stub("sudo", `echo "sudo $*" >> "$LOG"; if [ "$1" = -u ]; then shift 2; exec "$@"; fi; exit 0`)
	r.env = append(os.Environ(),
		"PATH="+shims+string(os.PathListSeparator)+os.Getenv("PATH"),
		"LOG="+r.log, "BIN_DIR="+r.bin, "CONF_DIR="+r.conf, "UNIT_DIR="+r.unit,
		"SERVICE_USER=", "SUDO_USER=", "FAKE_UID=0", "FAKE_WHOAMI=testuser",
	)
	return r
}

func (r *installerRig) run(t *testing.T, dir string, extraEnv ...string) (string, error) {
	t.Helper()
	return r.runScript(t, filepath.Join(r.repo, "service_install.sh"), dir, extraEnv...)
}

func (r *installerRig) runScript(t *testing.T, script, dir string, extraEnv ...string) (string, error) {
	t.Helper()
	cmd := exec.Command("bash", script)
	cmd.Dir = dir
	cmd.Env = append(append([]string(nil), r.env...), extraEnv...)
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func (r *installerRig) calls() string {
	b, _ := os.ReadFile(r.log)
	return string(b)
}

// TestInstallerFreshInstall (P2): a fresh install started the service on the
// unedited example config (auto-starting a 24B -hf download), and the binary
// was cp'd over the target.
func TestInstallerFreshInstall(t *testing.T) {
	r := newInstallerRig(t)
	out, err := r.run(t, r.repo, "SUDO_USER=testuser")
	if err != nil {
		t.Fatalf("installer failed: %v\n%s", err, out)
	}
	if b, _ := os.ReadFile(filepath.Join(r.bin, "llama-manager")); string(b) != "fresh" {
		t.Errorf("installed binary = %q, want the fresh build", b)
	}
	if _, err := os.Stat(filepath.Join(r.bin, "llama-manager.new")); !os.IsNotExist(err) {
		t.Error("temporary llama-manager.new left behind")
	}
	if fi, err := os.Stat(filepath.Join(r.conf, "config.yaml")); err != nil || fi.Mode().Perm() != 0o640 {
		t.Errorf("config not installed with 0640: %v %v", fi, err)
	}
	unit, _ := os.ReadFile(filepath.Join(r.unit, "llama-manager.service"))
	if !strings.Contains(string(unit), "User=testuser") || !strings.Contains(string(unit), "Group=testgroup") {
		t.Errorf("unit file user/group not set:\n%s", unit)
	}
	calls := r.calls()
	if !strings.Contains(calls, "sudo -u testuser") {
		t.Errorf("build did not run as the invoking user:\n%s", calls)
	}
	if !strings.Contains(calls, "systemctl enable llama-manager") {
		t.Errorf("service not enabled:\n%s", calls)
	}
	if strings.Contains(calls, "systemctl restart") || strings.Contains(calls, "systemctl start") {
		t.Errorf("service started on the unedited example config:\n%s", calls)
	}
}

// TestInstallerUpgradeRebuildsAndRestarts (P2): a re-run after `git pull`
// reinstalled the stale binary (it only built when none existed), and a failing
// `systemctl status` aborted the script under set -e.
func TestInstallerUpgradeRebuildsAndRestarts(t *testing.T) {
	r := newInstallerRig(t)
	mustWriteFile(t, filepath.Join(r.conf, "config.yaml"), 1)
	if err := os.WriteFile(filepath.Join(r.repo, "llama-manager"), []byte("stale"), 0o755); err != nil {
		t.Fatal(err)
	}
	out, err := r.run(t, r.repo, "SUDO_USER=testuser")
	if err != nil {
		t.Fatalf("installer failed: %v\n%s", err, out)
	}
	if b, _ := os.ReadFile(filepath.Join(r.bin, "llama-manager")); string(b) != "fresh" {
		t.Errorf("installed binary = %q, want a rebuild", b)
	}
	if !strings.Contains(r.calls(), "systemctl restart llama-manager") {
		t.Errorf("existing install not restarted:\n%s", r.calls())
	}
}

// TestInstallerRefusesRoot (P2): from a root shell (no SUDO_USER) the service
// was silently installed as root.
func TestInstallerRefusesRoot(t *testing.T) {
	r := newInstallerRig(t)
	out, err := r.run(t, r.repo, "FAKE_WHOAMI=root")
	if err == nil || !strings.Contains(out, "refusing to install the service as root") {
		t.Fatalf("installer did not refuse root: %v\n%s", err, out)
	}
	if _, err := os.Stat(filepath.Join(r.bin, "llama-manager")); !os.IsNotExist(err) {
		t.Error("binary installed despite refusing")
	}
	if out, err := r.run(t, r.repo, "FAKE_WHOAMI=root", "SERVICE_USER=testuser"); err != nil {
		t.Fatalf("SERVICE_USER override failed: %v\n%s", err, out)
	}
}

// TestInstallerReexecUsesAbsolutePath (P2): `exec sudo "$0"` failed for
// `bash service_install.sh` ($0 has no slash) and after the build step's cd.
func TestInstallerReexecUsesAbsolutePath(t *testing.T) {
	for name, invoke := range map[string]func(r *installerRig) (script, dir string){
		"bash service_install.sh":        func(r *installerRig) (string, string) { return "service_install.sh", r.repo },
		"bash ./repo/service_install.sh": func(r *installerRig) (string, string) { return "./repo/service_install.sh", filepath.Dir(r.repo) },
	} {
		t.Run(name, func(t *testing.T) {
			r := newInstallerRig(t)
			script, dir := invoke(r)
			if out, err := r.runScript(t, script, dir, "FAKE_UID=1000"); err != nil {
				t.Fatalf("installer failed: %v\n%s", err, out)
			}
			want := "sudo bash " + filepath.Join(r.repo, "service_install.sh") + " --built"
			if !strings.Contains(r.calls(), want) {
				t.Fatalf("re-exec:\n%s\nwant %q", r.calls(), want)
			}
			if !strings.Contains(r.calls(), "go build -o llama-manager . in "+r.repo) {
				t.Errorf("build did not run in the repo:\n%s", r.calls())
			}
		})
	}
}
