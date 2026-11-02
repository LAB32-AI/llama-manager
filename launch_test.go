package main

import (
	"reflect"
	"strings"
	"testing"
)

func launchCfg(backend string, mainGPU int) *Config {
	return &Config{
		ServerBin: "/bin/llama-server", Host: "0.0.0.0", GPUBackend: backend,
		NGL: 99, MainGPU: mainGPU, ContextLength: 16384,
		CacheTypeK: "q8_0", CacheTypeV: "q8_0", Jinja: true,
		ExtraArgs: []string{"-np", "1"},
	}
}

func joined(a []string) string { return strings.Join(a, " ") }

// TestBuildLaunch covers argv/env construction, which had no tests at all.
func TestBuildLaunch(t *testing.T) {
	off := false
	ngl, ctx, ctk := 20, 4096, "f16"
	inst := []string{"--cache-reuse", "256"}
	cases := []struct {
		name     string
		cfg      *Config
		conf     InstanceConf
		wantArgs string
		wantEnv  []string
	}{
		{
			name:     "metal, local model, globals",
			cfg:      launchCfg("metal", 0),
			conf:     InstanceConf{Model: "/m/a.gguf", Port: 9001, GPUIDs: []int{0}},
			wantArgs: "-m /m/a.gguf --port 9001 --host 0.0.0.0 -ngl 99 -c 16384 -ctk q8_0 -ctv q8_0 --jinja --metrics --log-verbosity 2 -np 1",
		},
		{
			name:     "rocm single GPU: only device 0 is visible, whatever main_gpu says",
			cfg:      launchCfg("rocm", 1),
			conf:     InstanceConf{Model: "/m/a.gguf", Port: 9001, GPUIDs: []int{2}},
			wantArgs: "-m /m/a.gguf --port 9001 --host 0.0.0.0 -ngl 99 -c 16384 -mg 0 -ctk q8_0 -ctv q8_0 --jinja --metrics --log-verbosity 2 -np 1",
			wantEnv:  []string{"HIP_VISIBLE_DEVICES=2"},
		},
		{
			name:     "rocm multi-GPU honours main_gpu and splits evenly",
			cfg:      launchCfg("rocm", 1),
			conf:     InstanceConf{Model: "/m/a.gguf", Port: 9001, GPUIDs: []int{0, 1, 2, 3}},
			wantArgs: "-m /m/a.gguf --port 9001 --host 0.0.0.0 -ngl 99 -c 16384 -mg 1 --tensor-split 0.25,0.25,0.25,0.25 -ctk q8_0 -ctv q8_0 --jinja --metrics --log-verbosity 2 -np 1",
			wantEnv:  []string{"HIP_VISIBLE_DEVICES=0,1,2,3"},
		},
		{
			name:     "main_gpu beyond the instance's GPUs falls back to 0",
			cfg:      launchCfg("rocm_rocr", 5),
			conf:     InstanceConf{Model: "/m/a.gguf", Port: 9001, GPUIDs: []int{2, 3}},
			wantArgs: "-m /m/a.gguf --port 9001 --host 0.0.0.0 -ngl 99 -c 16384 -mg 0 --tensor-split 0.50,0.50 -ctk q8_0 -ctv q8_0 --jinja --metrics --log-verbosity 2 -np 1",
			wantEnv:  []string{"ROCR_VISIBLE_DEVICES=2,3"},
		},
		{
			name:     "cuda pins PCI-bus device order",
			cfg:      launchCfg("cuda", 0),
			conf:     InstanceConf{Model: "/m/a.gguf", Port: 9001, GPUIDs: []int{1}},
			wantArgs: "-m /m/a.gguf --port 9001 --host 0.0.0.0 -ngl 99 -c 16384 -mg 0 -ctk q8_0 -ctv q8_0 --jinja --metrics --log-verbosity 2 -np 1",
			wantEnv:  []string{"CUDA_VISIBLE_DEVICES=1", "CUDA_DEVICE_ORDER=PCI_BUS_ID"},
		},
		{
			name: "vulkan, HF model, per-instance overrides replace globals",
			cfg:  launchCfg("vulkan", 0),
			conf: InstanceConf{Model: "unsloth/GLM-4.5-Air-GGUF:UD-Q4_K_XL", Port: 9002, GPUIDs: []int{0},
				NGL: &ngl, ContextLength: &ctx, CacheTypeK: &ctk, Jinja: &off, ExtraArgs: &inst},
			wantArgs: "-hf unsloth/GLM-4.5-Air-GGUF:UD-Q4_K_XL --port 9002 --host 0.0.0.0 -ngl 20 -c 4096 -mg 0 -ctk f16 -ctv q8_0 --metrics --log-verbosity 2 --cache-reuse 256",
			wantEnv:  []string{"GGML_VK_VISIBLE_DEVICES=0"},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			spec, err := buildLaunch(c.conf, c.cfg)
			if err != nil {
				t.Fatal(err)
			}
			if spec.bin != "/bin/llama-server" {
				t.Errorf("bin = %q", spec.bin)
			}
			if got := joined(spec.args); got != c.wantArgs {
				t.Errorf("args:\n got  %s\n want %s", got, c.wantArgs)
			}
			if !reflect.DeepEqual(spec.env, c.wantEnv) {
				t.Errorf("env = %q, want %q", spec.env, c.wantEnv)
			}
		})
	}
}

func TestBuildLaunchRejectsInvalidHFModel(t *testing.T) {
	if _, err := buildLaunch(InstanceConf{Model: "bad model:../x", Port: 1, GPUIDs: []int{0}}, launchCfg("metal", 0)); err == nil {
		t.Fatal("invalid HF reference reached the command line")
	}
}
