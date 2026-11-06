package main

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"gopkg.in/yaml.v3"
)

// APIKey is one key for the /v1 API: whose it is and how much it may use. A
// key written as a plain string in config.yaml (the older form) has no name
// and no limits.
type APIKey struct {
	Name string `yaml:"name,omitempty" json:"name"`
	Key  string `yaml:"key" json:"key"`
	// RPM is requests per minute, Concurrent requests at the same time; 0 is
	// no limit.
	RPM        int `yaml:"rpm,omitempty" json:"rpm"`
	Concurrent int `yaml:"concurrent,omitempty" json:"concurrent"`
}

const (
	maxKeyRPM        = 100000
	maxKeyConcurrent = 1000
	maxKeyNameLen    = 64
)

func (k *APIKey) UnmarshalYAML(n *yaml.Node) error {
	if n.Kind == yaml.ScalarNode {
		*k = APIKey{}
		return n.Decode(&k.Key)
	}
	if n.Kind != yaml.MappingNode {
		return fmt.Errorf("api_keys: line %d: a key is a string or {name, key, rpm, concurrent}", n.Line)
	}
	// Node.Decode doesn't inherit the file decoder's KnownFields, so check
	// here: a typo'd "rmp: 30" must not silently mean no limit.
	for i := 0; i+1 < len(n.Content); i += 2 {
		if f := n.Content[i].Value; !slices.Contains([]string{"name", "key", "rpm", "concurrent"}, f) {
			return fmt.Errorf("api_keys: line %d: unknown field %q (use name, key, rpm, concurrent)", n.Content[i].Line, f)
		}
	}
	type plain APIKey
	var p plain
	if err := n.Decode(&p); err != nil {
		return err
	}
	*k = APIKey(p)
	return nil
}

func (k APIKey) MarshalYAML() (any, error) {
	if k.Name == "" && k.RPM == 0 && k.Concurrent == 0 {
		return k.Key, nil
	}
	type plain APIKey
	return plain(k), nil
}

// UnmarshalJSON takes a plain string too, as the settings API did before keys
// had names.
func (k *APIKey) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		*k = APIKey{Key: s}
		return nil
	}
	type plain APIKey
	var p plain
	if err := json.Unmarshal(b, &p); err != nil {
		return err
	}
	*k = APIKey(p)
	return nil
}

// keyID names a key in the usage table and the rate limiter without keeping
// the key itself.
func keyID(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:8])
}

// label is how logs and messages name a key: its name, else its last four
// characters.
func (k APIKey) label() string {
	if k.Name != "" {
		return k.Name
	}
	if len(k.Key) <= 4 {
		return "…"
	}
	return "…" + k.Key[len(k.Key)-4:]
}

var apiKeyRe = regexp.MustCompile(`^[\x21-\x7e]{1,256}$`)

// cleanAPIKeys trims the keys and names and drops blank keys and repeats. A
// key is up to 256 printable ASCII characters without spaces, as it travels in
// a header; a name is up to 64 characters without control characters (it is
// written to the log) and unique.
func cleanAPIKeys(in []APIKey) ([]APIKey, error) {
	var out []APIKey
	names := map[string]bool{}
	for _, k := range in {
		k.Key, k.Name = strings.TrimSpace(k.Key), strings.TrimSpace(k.Name)
		if k.Key == "" || slices.ContainsFunc(out, func(o APIKey) bool { return o.Key == k.Key }) {
			continue
		}
		if !apiKeyRe.MatchString(k.Key) {
			return nil, fmt.Errorf("api_keys: a key must be 1-256 printable characters without spaces")
		}
		if utf8.RuneCountInString(k.Name) > maxKeyNameLen || strings.ContainsFunc(k.Name, unicode.IsControl) || !utf8.ValidString(k.Name) {
			return nil, fmt.Errorf("api_keys: the name %q must be up to %d characters, on one line", k.Name, maxKeyNameLen)
		}
		if k.Name != "" {
			if names[strings.ToLower(k.Name)] {
				return nil, fmt.Errorf("api_keys: two keys are named %q; give each its own name", k.Name)
			}
			names[strings.ToLower(k.Name)] = true
		}
		if k.RPM < 0 || k.RPM > maxKeyRPM {
			return nil, fmt.Errorf("api_keys: %s: rpm must be 0 (no limit) to %d", k.label(), maxKeyRPM)
		}
		if k.Concurrent < 0 || k.Concurrent > maxKeyConcurrent {
			return nil, fmt.Errorf("api_keys: %s: concurrent must be 0 (no limit) to %d", k.label(), maxKeyConcurrent)
		}
		out = append(out, k)
	}
	return out, nil
}

// matchKey finds the configured key among those r sends. Every pair is
// compared in constant time, so the timing doesn't tell which key is close.
func matchKey(r *http.Request, keys []APIKey) (APIKey, bool) {
	var found APIKey
	ok := false
	for _, s := range presentedKeys(r) {
		for _, k := range keys {
			if subtle.ConstantTimeCompare([]byte(s), []byte(k.Key)) == 1 && !ok {
				found, ok = k, true
			}
		}
	}
	return found, ok
}

// keyByID finds a configured key by keyID.
func keyByID(keys []APIKey, id string) (APIKey, bool) {
	for _, k := range keys {
		if keyID(k.Key) == id {
			return k, true
		}
	}
	return APIKey{}, false
}
