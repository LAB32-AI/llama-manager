package main

import (
	"fmt"
	"strconv"
	"strings"
)

// launchSpec is everything needed to exec one llama-server run.
type launchSpec struct {
	bin  string
	args []string
	env  []string // added to the manager's own environment
	host string   // the --host value
}

// buildLaunch turns an instance and the current global settings into a
// command line. Per-instance overrides win over globals; extra_args go last so
// operators can override any manager default.
func buildLaunch(conf InstanceConf, cfg *Config) (launchSpec, error) {
	if conf.role == roleImageGen {
		return buildImageGenLaunch(conf, cfg)
	}
	if conf.role == roleSpeech {
		if spec, ok, err := buildSpeechLaunch(conf, cfg); ok {
			return spec, err
		}
	}
	cfg.mu.RLock()
	spec := launchSpec{bin: cfg.ServerBin, host: cfg.Host}
	ngl, mainGPU, ctxLen := cfg.NGL, cfg.MainGPU, cfg.ContextLength
	cacheK, cacheV, jinja := cfg.CacheTypeK, cfg.CacheTypeV, cfg.Jinja
	extraArgs := append([]string(nil), cfg.ExtraArgs...)
	gpuEnv := cfg.GPUEnvVar()
	cfg.mu.RUnlock()

	if conf.Jinja != nil {
		jinja = *conf.Jinja
	}
	if conf.ExtraArgs != nil {
		extraArgs = *conf.ExtraArgs
	}
	if conf.NGL != nil {
		ngl = *conf.NGL
	}
	if conf.ContextLength != nil {
		ctxLen = *conf.ContextLength
	}
	if conf.CacheTypeK != nil {
		cacheK = *conf.CacheTypeK
	}
	if conf.CacheTypeV != nil {
		cacheV = *conf.CacheTypeV
	}

	args := []string{}
	if isLocalModel(conf.Model) {
		args = append(args, "-m", conf.Model)
	} else {
		if err := validateModelRef(conf.Model); err != nil {
			return launchSpec{}, err
		}
		args = append(args, "-hf", conf.Model)
	}
	args = append(args,
		"--port", strconv.Itoa(conf.Port),
		"--host", spec.host,
		"-ngl", strconv.Itoa(ngl),
		"-c", strconv.Itoa(ctxLen),
	)

	if gpuEnv != "" {
		// The visibility variable renumbers the instance's GPUs from 0, so
		// main_gpu indexes into gpu_ids. With a single GPU only device 0 exists.
		if n := len(conf.GPUIDs); n > 1 {
			mg := mainGPU
			if mg >= n {
				mg = 0
			}
			ratio := fmt.Sprintf("%.2f", 1.0/float64(n))
			parts := make([]string, n)
			for i := range parts {
				parts[i] = ratio
			}
			args = append(args, "-mg", strconv.Itoa(mg), "--tensor-split", strings.Join(parts, ","))
		} else {
			args = append(args, "-mg", "0")
		}
		spec.env = append(spec.env, gpuEnv+"="+strings.Join(intsToStrings(conf.GPUIDs), ","))
		if gpuEnv == "CUDA_VISIBLE_DEVICES" {
			// CUDA numbers devices fastest-first by default; match nvidia-smi's
			// PCI-bus order, which is what the dashboard and gpu_ids use.
			spec.env = append(spec.env, "CUDA_DEVICE_ORDER=PCI_BUS_ID")
		}
	}

	if cacheK != "" {
		args = append(args, "-ctk", cacheK)
	}
	if cacheV != "" {
		args = append(args, "-ctv", cacheV)
	}
	if jinja {
		args = append(args, "--jinja")
	}
	args = append(args, "--metrics", "--log-verbosity", "2")
	spec.args = append(args, extraArgs...)
	return spec, nil
}
