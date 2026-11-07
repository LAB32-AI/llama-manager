package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"sort"
	"strings"
	"time"
)

// hostAddress is one IPv4 address other machines may use to reach the
// manager, shown in the UI next to the /v1 endpoint.
type hostAddress struct {
	Iface string `json:"iface"`
	IP    string `json:"ip"`
	Kind  string `json:"kind"` // lan, vpn or other
}

var cgnat = &net.IPNet{IP: net.IPv4(100, 64, 0, 0), Mask: net.CIDRMask(10, 32)}

func addressKind(iface string, ip net.IP) string {
	name := strings.ToLower(iface)
	for _, p := range []string{"utun", "tun", "tap", "wg", "tailscale", "zt", "ppp", "ipsec"} {
		if strings.HasPrefix(name, p) {
			return "vpn"
		}
	}
	if cgnat.Contains(ip) {
		return "vpn"
	}
	if ip.IsPrivate() {
		return "lan"
	}
	return "other"
}

func hostAddresses() []hostAddress {
	out := []hostAddress{}
	ifaces, err := net.Interfaces()
	if err != nil {
		return out
	}
	for _, ifc := range ifaces {
		if ifc.Flags&net.FlagUp == 0 || ifc.Flags&net.FlagLoopback != 0 {
			continue
		}
		addrs, err := ifc.Addrs()
		if err != nil {
			continue
		}
		for _, a := range addrs {
			n, ok := a.(*net.IPNet)
			if !ok {
				continue
			}
			ip := n.IP.To4()
			if ip == nil || ip.IsLinkLocalUnicast() {
				continue
			}
			out = append(out, hostAddress{Iface: ifc.Name, IP: ip.String(), Kind: addressKind(ifc.Name, ip)})
		}
	}
	rank := map[string]int{"lan": 0, "vpn": 1, "other": 2}
	sort.SliceStable(out, func(i, j int) bool {
		if rank[out[i].Kind] != rank[out[j].Kind] {
			return rank[out[i].Kind] < rank[out[j].Kind]
		}
		return out[i].Iface < out[j].Iface
	})
	return out
}

// InstanceProps is what the chat UI needs to know about a running model.
type InstanceProps struct {
	NCtx     int    `json:"n_ctx"`
	Slots    int    `json:"slots"`
	Vision   bool   `json:"vision"`
	Audio    bool   `json:"audio"`
	Thinking bool   `json:"thinking"`
	Build    string `json:"build,omitempty"`
	// Sampling is what the instance uses when a request leaves it out
	// (temperature, top_p, min_p, top_k, repeat_penalty), for the chat
	// settings' placeholders.
	Sampling map[string]float64 `json:"sampling,omitempty"`
}

// templateThinks guesses from the chat template whether the model reasons
// before answering (Qwen3, DeepSeek-R1, gpt-oss, Magistral and similar).
func templateThinks(tmpl string) bool {
	for _, marker := range []string{"enable_thinking", "<think>", "reasoning_content", "<|channel|>", "[THINK]"} {
		if strings.Contains(tmpl, marker) {
			return true
		}
	}
	return false
}

func (inst *Instance) FetchProps(ctx context.Context) (InstanceProps, error) {
	var p InstanceProps
	if s := inst.State(); s != StateRunning {
		return p, fmt.Errorf("%s is %s", inst.conf.Name, s)
	}
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, inst.baseURL()+"/props", nil)
	if err != nil {
		return p, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return p, fmt.Errorf("reading %s props: %w", inst.conf.Name, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return p, fmt.Errorf("%s props: %s", inst.conf.Name, resp.Status)
	}
	var raw struct {
		DGS struct {
			NCtx   int            `json:"n_ctx"`
			Params map[string]any `json:"params"`
		} `json:"default_generation_settings"`
		NCtx       int `json:"n_ctx"`
		TotalSlots int `json:"total_slots"`
		Modalities struct {
			Vision bool `json:"vision"`
			Audio  bool `json:"audio"`
		} `json:"modalities"`
		ChatTemplate string `json:"chat_template"`
		BuildInfo    string `json:"build_info"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&raw); err != nil {
		return p, fmt.Errorf("%s props: %w", inst.conf.Name, err)
	}
	p.NCtx = raw.DGS.NCtx
	if p.NCtx == 0 {
		p.NCtx = raw.NCtx
	}
	p.Slots, p.Vision, p.Audio = raw.TotalSlots, raw.Modalities.Vision, raw.Modalities.Audio
	p.Thinking, p.Build = templateThinks(raw.ChatTemplate), raw.BuildInfo
	for _, k := range []string{"temperature", "top_p", "min_p", "top_k", "repeat_penalty"} {
		if v, ok := raw.DGS.Params[k].(float64); ok {
			if p.Sampling == nil {
				p.Sampling = map[string]float64{}
			}
			p.Sampling[k] = v
		}
	}
	return p, nil
}
