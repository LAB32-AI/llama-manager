package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

type supervisorHandle struct {
	cancel context.CancelFunc
	done   chan struct{}
}

type Manager struct {
	cfg        *Config
	mu         sync.RWMutex
	instances  []*Instance
	byName     map[string]*Instance
	supervisor map[string]*supervisorHandle
	wg         sync.WaitGroup
	stopCh     chan struct{}
}

func NewManager(cfg *Config) *Manager {
	m := &Manager{
		cfg:        cfg,
		byName:     make(map[string]*Instance),
		supervisor: make(map[string]*supervisorHandle),
		stopCh:     make(chan struct{}),
	}
	for _, ic := range cfg.Instances {
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
			m.supervise(inst)
		}
	}
}

func (m *Manager) StartInstance(name string) error {
	m.mu.RLock()
	inst := m.byName[name]
	m.mu.RUnlock()
	if inst == nil {
		return fmt.Errorf("instance %q not found", name)
	}
	inst.ResetRestarts()
	m.supervise(inst)
	return nil
}

// detachSupervisor removes the supervisor goroutine for name (if any), cancels
// it, and blocks until it has fully exited. Waiting is essential: the exiting
// goroutine calls inst.Stop() on its way out (runWithRestart, ctx.Done case),
// so a caller that re-supervises without waiting would have its freshly-started
// process killed by the old goroutine's teardown — the /restart bug.
func (m *Manager) detachSupervisor(name string) {
	m.mu.Lock()
	h := m.supervisor[name]
	delete(m.supervisor, name)
	m.mu.Unlock()
	if h != nil {
		h.cancel()
		<-h.done
	}
}

func (m *Manager) StopInstance(name string) error {
	m.mu.RLock()
	inst := m.byName[name]
	m.mu.RUnlock()
	if inst == nil {
		return fmt.Errorf("instance %q not found", name)
	}
	m.detachSupervisor(name)
	return inst.Stop()
}

func (m *Manager) RestartInstance(name string) error {
	m.mu.RLock()
	inst := m.byName[name]
	m.mu.RUnlock()
	if inst == nil {
		return fmt.Errorf("instance %q not found", name)
	}
	if err := m.StopInstance(name); err != nil {
		return err
	}
	inst.ResetRestarts()
	m.supervise(inst)
	return nil
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
	m.detachSupervisor(name)
	_ = inst.Stop()
}

func (m *Manager) supervise(inst *Instance) {
	// Ensure any prior supervisor goroutine has fully exited before we spawn a
	// new one, so its teardown can't kill the process we're about to start.
	m.detachSupervisor(inst.conf.Name)

	ctx, cancel := context.WithCancel(context.Background())
	handle := &supervisorHandle{cancel: cancel, done: make(chan struct{})}

	m.mu.Lock()
	m.supervisor[inst.conf.Name] = handle
	m.mu.Unlock()

	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		defer close(handle.done)
		m.runWithRestart(inst, ctx)
		m.mu.Lock()
		if m.supervisor[inst.conf.Name] == handle {
			delete(m.supervisor, inst.conf.Name)
		}
		m.mu.Unlock()
	}()
}

func (m *Manager) isManaged(inst *Instance) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.byName[inst.conf.Name] == inst
}

func (m *Manager) runWithRestart(inst *Instance, ctx context.Context) {
	for {
		if !m.isManaged(inst) {
			return
		}
		if ctx.Err() != nil {
			return
		}
		exitCh, err := inst.Start()
		if err != nil {
			log.Printf("[%s] failed to start: %v", inst.conf.Name, err)
			return
		}

		go m.healthCheckLoop(inst, ctx)

		select {
		case <-exitCh:
		case <-ctx.Done():
			_ = inst.Stop()
			<-exitCh // drain the exit goroutine (sees Stopped, won't mark crashed)
			return
		case <-m.stopCh:
			_ = inst.Stop()
			<-exitCh
			return
		}

		if inst.State() == StateStopped {
			return
		}

		inst.IncrementRestarts()
		count := inst.RestartCount()
		if m.cfg.MaxRestarts > 0 && count >= m.cfg.MaxRestarts {
			log.Printf("[%s] reached max restarts (%d), giving up", inst.conf.Name, m.cfg.MaxRestarts)
			return
		}

		inst.SetState(StateRestarting)
		log.Printf("[%s] restarting in %s (restart %d)", inst.conf.Name, m.cfg.RestartDelay.Duration, count)

		select {
		case <-time.After(m.cfg.RestartDelay.Duration):
		case <-ctx.Done():
			inst.SetState(StateStopped)
			return
		case <-m.stopCh:
			inst.SetState(StateStopped)
			return
		}
	}
}

func (m *Manager) healthCheckLoop(inst *Instance, ctx context.Context) {
	inst.mu.Lock()
	stopCh := inst.stopCh
	inst.mu.Unlock()

	if stopCh == nil {
		return
	}

	ticker := time.NewTicker(m.cfg.HealthCheckInterval.Duration)
	defer ticker.Stop()

	unhealthyAfter := m.cfg.UnhealthyAfter
	killEnabled := unhealthyAfter > 0

	failures := 0
	for {
		select {
		case <-ticker.C:
			state := inst.State()
			if state != StateStarting && state != StateRunning && state != StateUnhealthy {
				continue
			}
			if inst.CheckHealth() {
				failures = 0
				if state != StateRunning {
					inst.SetState(StateRunning)
					log.Printf("[%s] health check passed, marked running", inst.conf.Name)
				}
				continue
			}
			failures++
			if state == StateRunning {
				inst.SetState(StateUnhealthy)
				if killEnabled {
					log.Printf("[%s] health check failed (%d/%d), demoting to unhealthy", inst.conf.Name, failures, unhealthyAfter)
				} else {
					log.Printf("[%s] health check failed, demoting to unhealthy", inst.conf.Name)
				}
			}
			if killEnabled && failures >= unhealthyAfter {
				inst.killProcess(fmt.Sprintf("health checks failed %d times in a row", failures))
				return
			}
		case <-stopCh:
			return
		case <-ctx.Done():
			return
		}
	}
}

func (m *Manager) Shutdown() {
	log.Println("shutting down all instances...")
	close(m.stopCh)
	m.mu.RLock()
	insts := make([]*Instance, len(m.instances))
	copy(insts, m.instances)
	m.mu.RUnlock()
	for _, inst := range insts {
		_ = inst.Stop()
	}
	m.wg.Wait()
	log.Println("all instances stopped")
}
