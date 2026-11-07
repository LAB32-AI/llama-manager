package main

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

// ChatConf is the general chat settings (Settings, Chat): what every chat in
// the UI uses unless its own Chat settings set something else. A sampling
// value left unset isn't sent, so the instance's own applies (llama-server's
// default, or its --temp/--top-p... flags). Only the UI's chats use these,
// never /v1.
type ChatConf struct {
	// SystemPrompt goes first in every chat, before the chat's own prompt; a
	// chat can leave it out ("sys_general": false in its params).
	SystemPrompt  string   `yaml:"system_prompt,omitempty" json:"system_prompt"`
	Temperature   *float64 `yaml:"temperature,omitempty" json:"temperature"`
	TopP          *float64 `yaml:"top_p,omitempty" json:"top_p"`
	MinP          *float64 `yaml:"min_p,omitempty" json:"min_p"`
	TopK          *int     `yaml:"top_k,omitempty" json:"top_k"`
	RepeatPenalty *float64 `yaml:"repeat_penalty,omitempty" json:"repeat_penalty"`
	// MaxTokens caps each reply; 0 means no cap.
	MaxTokens int `yaml:"max_tokens,omitempty" json:"max_tokens"`
	// Voice ("instance|voice id") and VoiceLang are the read-aloud defaults.
	// The browser sends them with each speech request; the server keeps them
	// so every browser shares them.
	Voice     string `yaml:"voice,omitempty" json:"voice"`
	VoiceLang string `yaml:"voice_lang,omitempty" json:"voice_lang"`
}

const maxSystemPrompt = 32 << 10

var voiceLangRe = regexp.MustCompile(`^[a-z]{2,3}$`)

// clean trims the text fields and checks every value is in range.
func (c ChatConf) clean() (ChatConf, error) {
	c.SystemPrompt = strings.TrimSpace(c.SystemPrompt)
	c.Voice = strings.TrimSpace(c.Voice)
	c.VoiceLang = strings.ToLower(strings.TrimSpace(c.VoiceLang))
	if c.VoiceLang == "auto" {
		c.VoiceLang = ""
	}
	if len(c.SystemPrompt) > maxSystemPrompt {
		return c, fmt.Errorf("chat.system_prompt is longer than %d KB", maxSystemPrompt>>10)
	}
	if !utf8.ValidString(c.SystemPrompt) {
		return c, fmt.Errorf("chat.system_prompt is not valid UTF-8")
	}
	for _, f := range []struct {
		name   string
		v      *float64
		lo, hi float64
	}{{"temperature", c.Temperature, 0, 5}, {"top_p", c.TopP, 0, 1}, {"min_p", c.MinP, 0, 1}, {"repeat_penalty", c.RepeatPenalty, 0, 5}} {
		if f.v != nil && (math.IsNaN(*f.v) || *f.v < f.lo || *f.v > f.hi) {
			return c, fmt.Errorf("chat.%s must be from %g to %g", f.name, f.lo, f.hi)
		}
	}
	if c.TopK != nil && (*c.TopK < 0 || *c.TopK > 10000) {
		return c, fmt.Errorf("chat.top_k must be from 0 to 10000")
	}
	if c.MaxTokens < 0 || c.MaxTokens > 1<<20 {
		return c, fmt.Errorf("chat.max_tokens must be from 0 (no limit) to %d", 1<<20)
	}
	if len(c.Voice) > 200 || strings.ContainsFunc(c.Voice, func(r rune) bool { return r < ' ' }) {
		return c, fmt.Errorf("chat.voice must be instance|voice")
	}
	if c.VoiceLang != "" && !voiceLangRe.MatchString(c.VoiceLang) {
		return c, fmt.Errorf("chat.voice_lang must be a language code such as en, or auto")
	}
	return c, nil
}

// chatParams is the subset of a chat's saved settings sent to llama-server,
// with the general chat settings filled in where the chat sets nothing. The
// UI stores them loosely (a number may be a string), so each is optional.
type chatParams struct {
	System string
	Think  bool
	Size   string // WxH, for image-gen
	// LoRAs are the image models' picked LoRAs, by instance name.
	LoRAs map[string][]loraPick
	opts  map[string]any
}

func parseChatParams(raw json.RawMessage, gen ChatConf) chatParams {
	p := chatParams{Think: true, opts: map[string]any{}}
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil {
		m = nil
	}
	num := func(k string) (float64, bool) {
		switch v := m[k].(type) {
		case float64:
			return v, true
		case string:
			f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
			return f, err == nil
		}
		return 0, false
	}
	var parts []string
	if b, ok := m["sys_general"].(bool); (!ok || b) && gen.SystemPrompt != "" {
		parts = append(parts, gen.SystemPrompt)
	}
	if s, ok := m["sys"].(string); ok && strings.TrimSpace(s) != "" {
		parts = append(parts, strings.TrimSpace(s))
	}
	p.System = strings.Join(parts, "\n\n")
	if b, ok := m["think"].(bool); ok {
		p.Think = b
	}
	if s, ok := m["size"].(string); ok {
		p.Size = s
	}
	if byInst, ok := m["loras"].(map[string]any); ok {
		p.LoRAs = map[string][]loraPick{}
		for inst, v := range byInst {
			items, _ := v.([]any)
			for _, it := range items {
				o, _ := it.(map[string]any)
				id, _ := o["id"].(string)
				w, ok := o["w"].(float64)
				if !ok {
					w = 1
				}
				if id != "" {
					p.LoRAs[inst] = append(p.LoRAs[inst], loraPick{ID: id, W: w})
				}
			}
		}
	}
	for _, o := range []struct {
		key, opt string
		gen      *float64
	}{{"temp", "temperature", gen.Temperature}, {"top_p", "top_p", gen.TopP}, {"min_p", "min_p", gen.MinP}, {"rep", "repeat_penalty", gen.RepeatPenalty}} {
		if v, ok := num(o.key); ok && v >= 0 {
			p.opts[o.opt] = v
		} else if o.gen != nil {
			p.opts[o.opt] = *o.gen
		}
	}
	if v, ok := num("top_k"); ok && v >= 0 {
		p.opts["top_k"] = int(v)
	} else if gen.TopK != nil {
		p.opts["top_k"] = *gen.TopK
	}
	// A chat's 0 means no limit even when the general settings have one.
	if v, ok := num("max"); ok && v >= 0 {
		if v > 0 {
			p.opts["max_tokens"] = int(v)
		}
	} else if gen.MaxTokens > 0 {
		p.opts["max_tokens"] = gen.MaxTokens
	}
	if v, ok := num("seed"); ok {
		p.opts["seed"] = int64(v)
	}
	return p
}
