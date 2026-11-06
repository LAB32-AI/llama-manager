package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

func main() {
	configPath := flag.String("config", "config.yaml", "path to config file")
	flag.StringVar(&chatDBPath, "chat-db", "", "chat history database (default: chats.db next to the config file)")
	flag.Parse()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	if err := run(*configPath, sigCh, nil); err != nil {
		log.Fatal(err)
	}
}

// httpShutdownGrace is how long in-flight requests get to finish on shutdown
// before their connections are closed.
var httpShutdownGrace = 5 * time.Second

// chatDBPath overrides where the chat history lives. It is a flag rather
// than a config.yaml key so neither the settings API nor a config import can
// point the server at a different file to write.
var chatDBPath string

func openChats(configPath string, mgr *Manager) (*ChatService, *ChatStore) {
	path := chatDBPath
	if path == "" {
		path = filepath.Join(filepath.Dir(configPath), "chats.db")
	}
	store, err := OpenChatStore(path)
	if err != nil {
		log.Printf("chat history disabled: %v", err)
		return nil, nil
	}
	log.Printf("chat history: %s", path)
	return NewChatService(store, mgr), store
}

// run serves the manager until a signal arrives on stop. ready, if set, is
// called with the listening address once requests are being accepted.
func run(configPath string, stop chan os.Signal, ready func(addr string)) error {
	cfg, err := loadConfig(configPath)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	log.Printf("loaded %d instance(s) from %s", len(cfg.Instances), configPath)

	// Bind before starting anything: if the port is taken we must fail before
	// any llama-server is spawned, or those children would outlive us.
	ln, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.ManagerPort))
	if err != nil {
		return fmt.Errorf("listening on port %d: %w", cfg.ManagerPort, err)
	}

	SweepIncomplete()
	mgr := NewManager(cfg)
	mgr.StartAll()

	ws := NewWebServer(mgr, cfg, NewDownloadManager())
	chats, store := openChats(configPath, mgr)
	ws.chats = chats
	if store != nil {
		mgr.events.attach(store.db)
		mgr.stats.attach(store.db)
	}
	mgr.events.add(evManager, "", "llama-manager started")
	if chats != nil {
		chats.reader = ws.reader
		chats.images = ws.imageGen
	}
	apiCtx, stopAPI := context.WithCancel(context.Background())
	defer stopAPI()
	ws.startAPIState(apiCtx)
	ws.startDashboard(apiCtx)
	if ws.reader != nil {
		log.Printf("images: described by %s (%s) for models that can't see", readerName, ws.reader.model)
	}
	if g := ws.imageGen; g != nil {
		for _, e := range g.models {
			log.Printf("images: %s runs %s on GPU %v", e.Name, e.Preset.Label, e.GPUIDs)
		}
		g.Boot()
	}
	if s := ws.speech; s != nil {
		for _, e := range s.models {
			log.Printf("speech: %s runs %s (%s) on GPU %v", e.Name, e.Preset.Label, e.Preset.Engine, e.GPUIDs)
		}
		s.Boot()
	}
	if t := ws.tika; t != nil {
		go func() {
			if st := t.ProbeNow(); st.OK {
				log.Printf("documents: read by %s at %s", st.Version, t.shown)
			} else {
				log.Printf("documents: Tika at %s isn't answering (%s); the browser reads documents until it does", t.shown, st.Err)
			}
		}()
	}
	closeChats := func() {
		stopAPI()
		if chats != nil {
			chats.Shutdown(5 * time.Second)
			ws.usage.close()
			mgr.events.add(evManager, "", "llama-manager stopped")
			mgr.stats.close()
			mgr.events.close()
			store.Close()
		}
		if ws.imageGen != nil {
			ws.imageGen.Close()
		}
		if ws.speech != nil {
			ws.speech.Close()
		}
	}
	srv := &http.Server{Handler: ws}
	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.Serve(ln) }()
	log.Printf("web UI available at http://localhost:%d", cfg.ManagerPort)
	if ready != nil {
		ready(ln.Addr().String())
	}
	// Funnel starts once requests are being accepted, and stops before
	// anything else, so the internet never reaches a half-started or
	// half-stopped manager.
	if ws.funnel != nil {
		log.Printf("internet: /v1 through Tailscale Funnel on port %d (public_api in config.yaml)", ws.funnel.port)
		ws.funnel.Start()
	}

	select {
	case sig := <-stop:
		// Restore default signal handling so a second Ctrl-C/SIGTERM kills us
		// even if a llama-server refuses to die.
		signal.Stop(stop)
		log.Printf("received %v, shutting down (signal again to force exit)", sig)
	case err := <-serveErr:
		ws.funnel.Close()
		closeChats()
		mgr.Shutdown()
		return fmt.Errorf("http server: %w", err)
	}
	ws.funnel.Close()

	// Stop taking requests first, so nothing can start an instance while the
	// manager is stopping them.
	ctx, cancel := context.WithTimeout(context.Background(), httpShutdownGrace)
	if err := srv.Shutdown(ctx); err != nil {
		_ = srv.Close()
	}
	cancel()
	closeChats()
	mgr.Shutdown()
	if err := <-serveErr; err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("http server: %w", err)
	}
	return nil
}
