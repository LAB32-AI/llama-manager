package main

import (
	"database/sql"
	"log"
	"sync"
	"time"
)

// The dashboard's activity feed: failed requests, crashes, restarts, Funnel
// and download changes, config edits. The last ones are kept in memory and
// in chats.db (dash_events, 30 days).

const (
	maxEvents    = 200
	eventsKeep   = 30 * 24 * time.Hour
	maxEventsRow = 5000
)

// Event kinds.
const (
	evError       = "error"
	evCrash       = "crash"
	evStartFailed = "start_failed"
	evRestart     = "restart"
	evGaveUp      = "gave_up"
	evUnhealthy   = "unhealthy"
	evKilled      = "killed"
	evReady       = "ready"
	evStopped     = "stopped"
	evFunnel      = "funnel"
	evDownload    = "download"
	evConfig      = "config"
	evManager     = "manager"
)

type dashEvent struct {
	T     int64  `json:"t"`
	Kind  string `json:"kind"`
	Model string `json:"model,omitempty"`
	Msg   string `json:"msg"`
}

type eventLog struct {
	mu      sync.Mutex
	list    []dashEvent // oldest first
	pending []dashEvent // not yet written
	db      *sql.DB
	closed  bool
}

// add records an event. It only touches memory, so it is safe under other
// locks (an instance's); writing happens in flush.
func (l *eventLog) add(kind, model, msg string) {
	if l == nil {
		return
	}
	e := dashEvent{T: time.Now().Unix(), Kind: kind, Model: model, Msg: msg}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.list = append(l.list, e)
	if len(l.list) > maxEvents {
		l.list = append([]dashEvent(nil), l.list[len(l.list)-maxEvents:]...)
	}
	if !l.closed {
		l.pending = append(l.pending, e)
	}
}

// recent is the newest n events, newest first.
func (l *eventLog) recent(n int) []dashEvent {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	out := []dashEvent{}
	for i := len(l.list) - 1; i >= 0 && len(out) < n; i-- {
		out = append(out, l.list[i])
	}
	return out
}

func (l *eventLog) attach(db *sql.DB) {
	if l == nil || db == nil {
		return
	}
	rows, err := db.Query(`SELECT t, kind, model, msg FROM dash_events ORDER BY id DESC LIMIT ?`, maxEvents)
	if err != nil {
		log.Printf("dashboard: reading events: %v", err)
	}
	var old []dashEvent
	if err == nil {
		for rows.Next() {
			var e dashEvent
			if rows.Scan(&e.T, &e.Kind, &e.Model, &e.Msg) == nil {
				old = append([]dashEvent{e}, old...)
			}
		}
		rows.Close()
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.list = append(old, l.list...)
	if len(l.list) > maxEvents {
		l.list = append([]dashEvent(nil), l.list[len(l.list)-maxEvents:]...)
	}
	l.db = db
}

func (l *eventLog) flush() {
	if l == nil {
		return
	}
	l.mu.Lock()
	db, pending := l.db, l.pending
	if db == nil || len(pending) == 0 {
		l.mu.Unlock()
		return
	}
	l.pending = nil
	l.mu.Unlock()
	tx, err := db.Begin()
	if err == nil {
		for _, e := range pending {
			if _, err = tx.Exec(`INSERT INTO dash_events (t, kind, model, msg) VALUES (?, ?, ?, ?)`, e.T, e.Kind, e.Model, e.Msg); err != nil {
				break
			}
		}
		if err == nil {
			err = tx.Commit()
		} else {
			_ = tx.Rollback()
		}
	}
	if err != nil {
		log.Printf("dashboard: writing events: %v (kept for the next try)", err)
		l.mu.Lock()
		l.pending = append(pending, l.pending...)
		l.mu.Unlock()
	}
}

func (l *eventLog) prune(now time.Time) {
	if l == nil {
		return
	}
	l.mu.Lock()
	db := l.db
	l.mu.Unlock()
	if db == nil {
		return
	}
	_, _ = db.Exec(`DELETE FROM dash_events WHERE t < ?`, now.Add(-eventsKeep).Unix())
	_, _ = db.Exec(`DELETE FROM dash_events WHERE id <= (SELECT MAX(id) FROM dash_events) - ?`, maxEventsRow)
}

func (l *eventLog) close() {
	if l == nil {
		return
	}
	l.flush()
	l.mu.Lock()
	l.closed = true
	l.pending = nil
	l.mu.Unlock()
}
