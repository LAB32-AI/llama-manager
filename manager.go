package main

import (
	"log"
	"sync"
	"time"
)

type Manager struct {
	cfg       *Config
	mu        sync.RWMutex
	instances []*Instance
	byName    map[string]*Instance
	wg        sync.WaitGroup
	stopCh    chan struct{}
}

func NewManager(cfg *Config) *Manager {
	m := &Manager{
		cfg:    cfg,
		byName: make(map[string]*Instance),
		stopCh: make(chan struct{}),
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

func (m *Manager) StartAll() {
	m.mu.RLock()
	insts := make([]*Instance, len(m.instances))
	copy(insts, m.instances)
	m.mu.RUnlock()
	for _, inst := range insts {
		m.supervise(inst)
	}
}

func (m *Manager) StartInstance(name string) error {
	m.mu.RLock()
	inst := m.byName[name]
	m.mu.RUnlock()
	if inst == nil {
		return nil
	}
	inst.ResetRestarts()
	m.supervise(inst)
	return nil
}

func (m *Manager) StopInstance(name string) error {
	m.mu.RLock()
	inst := m.byName[name]
	m.mu.RUnlock()
	if inst == nil {
		return nil
	}
	return inst.Stop()
}

func (m *Manager) RestartInstance(name string) error {
	m.mu.RLock()
	inst := m.byName[name]
	m.mu.RUnlock()
	if inst == nil {
		return nil
	}
	inst.ResetRestarts()
	_ = inst.Stop()
	time.Sleep(500 * time.Millisecond)
	m.supervise(inst)
	return nil
}

func (m *Manager) AddInstance(ic InstanceConf) {
	inst := NewInstance(ic, m.cfg)
	m.mu.Lock()
	m.instances = append(m.instances, inst)
	m.byName[ic.Name] = inst
	m.mu.Unlock()
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
	_ = inst.Stop()
}

func (m *Manager) supervise(inst *Instance) {
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		m.runWithRestart(inst)
	}()
}

func (m *Manager) isManaged(inst *Instance) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.byName[inst.conf.Name] == inst
}

func (m *Manager) runWithRestart(inst *Instance) {
	for {
		if !m.isManaged(inst) {
			return
		}
		exitCh, err := inst.Start()
		if err != nil {
			log.Printf("[%s] failed to start: %v", inst.conf.Name, err)
			return
		}

		go m.healthCheckLoop(inst)

		select {
		case <-exitCh:
		case <-m.stopCh:
			_ = inst.Stop()
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
		case <-m.stopCh:
			inst.SetState(StateStopped)
			return
		}
	}
}

func (m *Manager) healthCheckLoop(inst *Instance) {
	inst.mu.Lock()
	stopCh := inst.stopCh
	inst.mu.Unlock()

	if stopCh == nil {
		return
	}

	ticker := time.NewTicker(m.cfg.HealthCheckInterval.Duration)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if inst.State() == StateStarting || inst.State() == StateRunning {
				if inst.CheckHealth() {
					inst.SetState(StateRunning)
				}
			}
		case <-stopCh:
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
