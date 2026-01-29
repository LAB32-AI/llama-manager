package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	configPath := flag.String("config", "config.yaml", "path to config file")
	flag.Parse()

	cfg, err := loadConfig(*configPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	log.Printf("loaded %d instance(s) from %s", len(cfg.Instances), *configPath)

	mgr := NewManager(cfg)
	mgr.StartAll()

	dlm := NewDownloadManager(cfg.ServerBin)
	srv := NewWebServer(mgr, cfg, dlm)
	httpServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.ManagerPort),
		Handler: srv,
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		log.Println("received shutdown signal")
		mgr.Shutdown()
		if err := httpServer.Close(); err != nil {
			log.Printf("error closing http server: %v", err)
		}
	}()

	log.Printf("web UI available at http://localhost:%d", cfg.ManagerPort)
	if err := httpServer.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("http server error: %v", err)
	}
}
