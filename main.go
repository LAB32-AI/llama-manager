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
	"syscall"
	"time"
)

func main() {
	configPath := flag.String("config", "config.yaml", "path to config file")
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

	srv := &http.Server{Handler: NewWebServer(mgr, cfg, NewDownloadManager())}
	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.Serve(ln) }()
	log.Printf("web UI available at http://localhost:%d", cfg.ManagerPort)
	if ready != nil {
		ready(ln.Addr().String())
	}

	select {
	case sig := <-stop:
		// Restore default signal handling so a second Ctrl-C/SIGTERM kills us
		// even if a llama-server refuses to die.
		signal.Stop(stop)
		log.Printf("received %v, shutting down (signal again to force exit)", sig)
	case err := <-serveErr:
		mgr.Shutdown()
		return fmt.Errorf("http server: %w", err)
	}

	// Stop taking requests first, so nothing can start an instance while the
	// manager is stopping them.
	ctx, cancel := context.WithTimeout(context.Background(), httpShutdownGrace)
	if err := srv.Shutdown(ctx); err != nil {
		_ = srv.Close()
	}
	cancel()
	mgr.Shutdown()
	if err := <-serveErr; err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("http server: %w", err)
	}
	return nil
}
