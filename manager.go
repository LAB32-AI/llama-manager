package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"reflect"
	"sync"
	"time"
)

// restartStableWindow is the minimum uptime that marks a run as "stable". If an
// instance stays up at least this long before exiting, the supervisor treats the
// crash as a fresh, likely-transient failure (e.g. an intermittent backend/GPU
// abort) and resets the restart budget — so a model that runs fine for a long
// time then occasionally aborts recovers indefinitely instead of burning through
// MaxRestarts and being permanently given up on. Only fast, repeated crashes (a
// model that dies on startup) keep accumulating toward MaxRestarts. It must
// exceed a normal model's load time so a crash during/just-after loading still
// counts as a real failure. Var, not const, so tests can shorten it.
var restartStableWindow = 120 * time.Second

type supervisorHandle struct {
	cancel context.CancelFunc
	done   chan struct{}
}

type Manager struct {
	cfg        *Config
	mu         sync.RWMutex
	instances  []*Instance
	byName     map[string]*Instance
	supervisor map[*Instance]*supervisorHandle
	wg         sync.WaitGroup
	stopCh     chan struct{}
	closing    bool // set by Shutdown; guarded by mu
}

var errShuttingDown = errors.New("manager is shutting down")

func NewManager(cfg *Config) *Manager {
	m := &Manager{
		cfg:        cfg,
		byName:     make(map[string]*Instance),
		supervisor: make(map[*Instance]*supervisorHandle),
		stopCh:     make(chan struct{}),
	}
	cfg.mu.RLock()
	confs := cfg.managedInstancesLocked()
	cfg.mu.RUnlock()
	for _, ic := range confs {
		inst := NewInstance(ic, cfg)
		m.instances = append(m.instances, inst)
		m.byName[ic.Name] = inst
	}
	return m
}

func (m *Manager) Instances() []*Instance {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]*Instance, len(m.instances))
	copy(result, m.instances)
	return result
}

func (m *Manager) Get(name string) *Instance {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.byName[name]
}

// StartAll launches the instances marked auto_start on boot. Others stay
// stopped until started manually — important on a single-GPU-pool rig where
// several configured models can't run at once.
func (m *Manager) StartAll() {
	m.mu.RLock()
	insts := make([]*Instance, len(m.instances))
	copy(insts, m.instances)
	m.mu.RUnlock()
	for _, inst := range insts {
		if inst.conf.AutoStart {
			inst.opMu.Lock()
			if m.isManaged(inst) {
				_ = m.supervise(inst)
			}
			inst.opMu.Unlock()
		}
	}
}

// lockInstance returns the named instance with its opMu held, or an error if it
// doesn't exist or was removed while we waited for the lock.
func (m *Manager) lockInstance(name string) (*Instance, error) {
	inst := m.Get(name)
	if inst == nil {
		return nil, fmt.Errorf("instance %q not found", name)
	}
	inst.opMu.Lock()
	if !m.isManaged(inst) {
		inst.opMu.Unlock()
		return nil, fmt.Errorf("instance %q not found", name)
	}
	return inst, nil
}

func (m *Manager) StartInstance(name string) error {
	inst, err := m.lockInstance(name)
	if err != nil {
		return err
	}
	defer inst.opMu.Unlock()
	inst.ResetRestarts()
	return m.supervise(inst)
}

// detachSupervisor removes the supervisor goroutine for inst (if any), cancels
// it, and blocks until it has fully exited. Waiting is essential: the exiting
// goroutine calls inst.Stop() on its way out (runWithRestart, ctx.Done case),
// so a caller that re-supervises without waiting would have its freshly-started
// process killed by the old goroutine's teardown — the /restart bug.
func (m *Manager) detachSupervisor(inst *Instance) {
	m.mu.Lock()
	h := m.supervisor[inst]
	delete(m.supervisor, inst)
	m.mu.Unlock()
	if h != nil {
		h.cancel()
		<-h.done
	}
}

func (m *Manager) StopInstance(name string) error {
	inst, err := m.lockInstance(name)
	if err != nil {
		return err
	}
	defer inst.opMu.Unlock()
	m.detachSupervisor(inst)
	return inst.Stop()
}

func (m *Manager) RestartInstance(name string) error {
	inst, err := m.lockInstance(name)
	if err != nil {
		return err
	}
	defer inst.opMu.Unlock()
	m.detachSupervisor(inst)
	if err := inst.Stop(); err != nil {
		return err
	}
	inst.ResetRestarts()
	return m.supervise(inst)
}

func (m *Manager) AddInstance(ic InstanceConf) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.byName[ic.Name]; ok {
		return fmt.Errorf("instance %q already exists", ic.Name)
	}
	inst := NewInstance(ic, m.cfg)
	m.instances = append(m.instances, inst)
	m.byName[ic.Name] = inst
	return nil
}

func (m *Manager) RemoveInstance(name string) {
	m.mu.Lock()
	inst := m.byName[name]
	if inst == nil {
		m.mu.Unlock()
		return
	}
	delete(m.byName, name)
	for i, in := range m.instances {
		if in.conf.Name == name {
			m.instances = append(m.instances[:i], m.instances[i+1:]...)
			break
		}
	}
	m.mu.Unlock()
	inst.opMu.Lock()
	defer inst.opMu.Unlock()
	m.detachSupervisor(inst)
	_ = inst.Stop()
}

func (m *Manager) hasSupervisor(inst *Instance) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, ok := m.supervisor[inst]
	return ok
}

// ReconcileInstances makes the managed set match confs (an imported config):
// unchanged instances keep running untouched, removed ones are stopped,
// changed ones are replaced and restarted if they were being supervised, and
// new ones are added stopped. Everything that goes away is stopped before any
// replacement starts, so a port moving between instances can't collide.
func (m *Manager) ReconcileInstances(confs []InstanceConf) (added, updated, removed []string) {
	want := make(map[string]InstanceConf, len(confs))
	for _, ic := range confs {
		want[ic.Name] = ic
	}
	existing := map[string]bool{}
	var restart []string
	for _, inst := range m.Instances() {
		name := inst.conf.Name
		existing[name] = true
		ic, keep := want[name]
		if keep && reflect.DeepEqual(ic, inst.conf) {
			continue
		}
		wasSupervised := m.hasSupervisor(inst)
		m.RemoveInstance(name)
		if !keep {
			removed = append(removed, name)
			continue
		}
		updated = append(updated, name)
		if wasSupervised {
			restart = append(restart, name)
		}
	}
	for _, ic := range confs {
		if existing[ic.Name] && m.Get(ic.Name) != nil {
			continue
		}
		if err := m.AddInstance(ic); err != nil {
			log.Printf("[%s] import: %v", ic.Name, err)
			continue
		}
		if !existing[ic.Name] {
			added = append(added, ic.Name)
		}
	}
	m.mu.Lock()
	ordered := make([]*Instance, 0, len(m.instances))
	for _, ic := range confs {
		if inst := m.byName[ic.Name]; inst != nil {
			ordered = append(ordered, inst)
		}
	}
	m.instances = ordered
	m.mu.Unlock()
	for _, name := range restart {
		if err := m.StartInstance(name); err != nil {
			log.Printf("[%s] import: restart failed: %v", name, err)
		}
	}
	return added, updated, removed
}

// supervise starts inst and hands the run to a supervisor goroutine that
// restarts it after crashes. The first start is synchronous so the caller gets
// its error (bad binary, fork failure) instead of a silent no-op. It must be
// called with inst.opMu held.
func (m *Manager) supervise(inst *Instance) error {
	// Ensure any prior supervisor goroutine has fully exited before we spawn a
	// new one, so its teardown can't kill the process we're about to start.
	m.detachSupervisor(inst)

	m.mu.RLock()
	closing := m.closing
	m.mu.RUnlock()
	if closing {
		return errShuttingDown
	}

	p, err := inst.Start()
	if err != nil {
		log.Printf("[%s] failed to start: %v", inst.conf.Name, err)
		inst.markStartFailed(err)
		return err
	}

	ctx, cancel := context.WithCancel(context.Background())
	handle := &supervisorHandle{cancel: cancel, done: make(chan struct{})}

	// Register and wg.Add under the same lock Shutdown takes to set closing,
	// so a supervisor is either counted by Shutdown's wg.Wait or never runs.
	m.mu.Lock()
	if m.closing {
		m.mu.Unlock()
		cancel()
		_ = inst.Stop()
		return errShuttingDown
	}
	m.supervisor[inst] = handle
	m.wg.Add(1)
	m.mu.Unlock()

	go func() {
		defer m.wg.Done()
		defer close(handle.done)
		m.runWithRestart(inst, ctx, p)
		m.mu.Lock()
		if m.supervisor[inst] == handle {
			delete(m.supervisor, inst)
		}
		m.mu.Unlock()
	}()
	return nil
}

func (m *Manager) isManaged(inst *Instance) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.byName[inst.conf.Name] == inst
}

// runWithRestart supervises run p and every restart after it. p is nil when the
// previous start attempt failed; that counts as a crash, so a binary that is
// briefly missing (mid-rebuild) is retried within the restart budget.
func (m *Manager) runWithRestart(inst *Instance, ctx context.Context, p *proc) {
	for {
		if p != nil {
			healthDone := make(chan struct{})
			go func() {
				defer close(healthDone)
				m.healthCheckLoop(inst, ctx, p)
			}()

			select {
			case <-p.exited:
				<-healthDone
			case <-ctx.Done():
				_ = inst.Stop()
				<-healthDone
				return
			case <-m.stopCh:
				_ = inst.Stop()
				<-healthDone
				return
			}

			if inst.State() != StateCrashed {
				return
			}

			// A run that lasted past the stable window is a fresh, likely-transient
			// failure rather than a crash-loop — clear the accumulated restart budget
			// so it can recover indefinitely. Fast, repeated crashes (uptime below the
			// window) keep their count and still trip MaxRestarts below.
			if time.Since(p.startedAt) >= restartStableWindow {
				inst.ResetRestarts()
			}
		}

		inst.IncrementRestarts()
		count := inst.RestartCount()
		sp := m.cfg.supervisorParams()
		if sp.maxRestarts > 0 && count > sp.maxRestarts {
			log.Printf("[%s] reached max restarts (%d), giving up", inst.conf.Name, sp.maxRestarts)
			return
		}

		if !inst.transition(nil, StateCrashed, StateRestarting) {
			return
		}
		log.Printf("[%s] restarting in %s (restart %d)", inst.conf.Name, sp.restartDelay, count)

		select {
		case <-time.After(sp.restartDelay):
		case <-ctx.Done():
			inst.transition(nil, StateRestarting, StateStopped)
			return
		case <-m.stopCh:
			inst.transition(nil, StateRestarting, StateStopped)
			return
		}
		if ctx.Err() != nil || !m.isManaged(inst) {
			inst.transition(nil, StateRestarting, StateStopped)
			return
		}

		var err error
		if p, err = inst.Start(); err != nil {
			log.Printf("[%s] failed to start: %v", inst.conf.Name, err)
			inst.markStartFailed(err)
			p = nil
		}
	}
}

// startingPollInterval caps the health-check period while an instance is still
// loading, so it is marked running soon after it becomes ready instead of up to
// a full health_check_interval later. Var so tests can shorten it.
var startingPollInterval = 2 * time.Second

func nextHealthDelay(interval time.Duration, state InstanceState) time.Duration {
	if interval <= 0 {
		interval = 30 * time.Second
	}
	if state == StateStarting && interval > startingPollInterval {
		return startingPollInterval
	}
	return interval
}

// healthCheckLoop watches one run. Failed checks only count once the instance
// has been healthy: while it is still starting, llama-server legitimately
// answers 503 (loading) or isn't listening yet (-hf download), however long
// that takes.
func (m *Manager) healthCheckLoop(inst *Instance, ctx context.Context, p *proc) {
	// checkCtx aborts an in-flight check as soon as the run ends, so the
	// supervisor's wait for this loop never stalls on health_check_timeout.
	checkCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		select {
		case <-p.done:
			cancel()
		case <-checkCtx.Done():
		}
	}()

	timer := time.NewTimer(nextHealthDelay(m.cfg.supervisorParams().healthInterval, StateStarting))
	defer timer.Stop()

	failures := 0
	for {
		select {
		case <-timer.C:
		case <-p.done:
			return
		case <-ctx.Done():
			return
		}

		sp := m.cfg.supervisorParams()
		unhealthyAfter := sp.unhealthyAfter
		killEnabled := unhealthyAfter > 0
		state := inst.State()
		if state == StateStarting || state == StateRunning || state == StateUnhealthy {
			healthy := inst.CheckHealth(checkCtx)
			select {
			case <-p.done:
				return // stopped or exited while the check was in flight
			default:
			}
			if healthy {
				failures = 0
				if state != StateRunning && inst.transition(p, state, StateRunning) {
					log.Printf("[%s] health check passed, marked running", inst.conf.Name)
				}
			} else if state != StateStarting {
				failures++
				if state == StateRunning && inst.transition(p, StateRunning, StateUnhealthy) {
					if killEnabled {
						log.Printf("[%s] health check failed (%d/%d), demoting to unhealthy", inst.conf.Name, failures, unhealthyAfter)
					} else {
						log.Printf("[%s] health check failed, demoting to unhealthy", inst.conf.Name)
					}
				}
				if killEnabled && failures >= unhealthyAfter {
					inst.killProcess(p, fmt.Sprintf("health checks failed %d times in a row", failures))
					return
				}
			}
		}
		timer.Reset(nextHealthDelay(sp.healthInterval, inst.State()))
	}
}

func (m *Manager) Shutdown() {
	m.mu.Lock()
	if m.closing {
		m.mu.Unlock()
		return
	}
	m.closing = true
	close(m.stopCh)
	insts := make([]*Instance, len(m.instances))
	copy(insts, m.instances)
	m.mu.Unlock()
	log.Println("shutting down all instances...")
	var stops sync.WaitGroup
	for _, inst := range insts {
		stops.Add(1)
		go func() {
			defer stops.Done()
			_ = inst.Stop()
		}()
	}
	stops.Wait()
	m.wg.Wait()
	log.Println("all instances stopped")
}
