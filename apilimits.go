package main

import (
	"fmt"
	"math"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// Wrong keys from one address in a minute before it gets 429s, so a key
// can't be guessed over the internet.
const (
	badKeyLimit  = 10
	badKeyWindow = time.Minute
)

// apiLimiter enforces each key's requests per minute and requests at once,
// the cap on requests from the internet, and the wrong-key counter.
type apiLimiter struct {
	mu       sync.Mutex
	buckets  map[string]*rateBucket // key id -> its requests-per-minute bucket
	inFlight map[string]int         // key id -> requests running
	public   int                    // requests from the internet running
	badKeys  map[string]*badKeyRun  // client address -> wrong keys this window
}

// rateBucket is a token bucket that holds a minute's worth of requests and
// refills continuously.
type rateBucket struct {
	tokens float64
	last   time.Time
}

type badKeyRun struct {
	n     int
	start time.Time
}

func newAPILimiter() *apiLimiter {
	return &apiLimiter{buckets: map[string]*rateBucket{}, inFlight: map[string]int{}, badKeys: map[string]*badKeyRun{}}
}

// limitErr is a 429 in OpenAI's shape, with Retry-After.
type limitErr struct {
	retry time.Duration
	msg   string
}

func (e *limitErr) write(w http.ResponseWriter) {
	w.Header().Set("Retry-After", strconv.Itoa(int(math.Ceil(max(e.retry, time.Second).Seconds()))))
	writeAPIError(w, http.StatusTooManyRequests, "rate_limit_exceeded", "", e.msg)
}

// admit reserves room for one request with key k (k.Key "" when the server
// has no keys). It returns the func that frees the room once the request is
// done, or the 429 to answer. Nothing is taken when it refuses.
func (l *apiLimiter) admit(k APIKey, public bool, publicCap int, now time.Time) (func(), *limitErr) {
	id := ""
	if k.Key != "" {
		id = keyID(k.Key)
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if id != "" && k.Concurrent > 0 && l.inFlight[id] >= k.Concurrent {
		return nil, &limitErr{time.Second, fmt.Sprintf("Key %s already has %d request(s) running, its limit. Try again when one finishes.", k.label(), k.Concurrent)}
	}
	if public && publicCap > 0 && l.public >= publicCap {
		return nil, &limitErr{time.Second, fmt.Sprintf("This server is already answering %d requests from the internet, its limit. Try again shortly.", publicCap)}
	}
	var b *rateBucket
	if id != "" && k.RPM > 0 {
		capacity, rate := float64(k.RPM), float64(k.RPM)/60
		b = l.buckets[id]
		if b == nil {
			b = &rateBucket{tokens: capacity, last: now}
			l.buckets[id] = b
		}
		b.tokens = min(capacity, b.tokens+now.Sub(b.last).Seconds()*rate)
		b.last = now
		if b.tokens < 1 {
			wait := time.Duration((1 - b.tokens) / rate * float64(time.Second))
			return nil, &limitErr{wait, fmt.Sprintf("Rate limit reached for key %s: %d requests per minute. Try again in %d s.", k.label(), k.RPM, int(math.Ceil(wait.Seconds())))}
		}
		b.tokens--
	}
	if id != "" {
		l.inFlight[id]++
	}
	if public {
		l.public++
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			l.mu.Lock()
			defer l.mu.Unlock()
			if id != "" {
				if l.inFlight[id]--; l.inFlight[id] <= 0 {
					delete(l.inFlight, id)
				}
			}
			if public {
				l.public--
			}
		})
	}, nil
}

// blocked is the 429 for an address that sent too many wrong keys lately.
func (l *apiLimiter) blocked(ip string, now time.Time) *limitErr {
	l.mu.Lock()
	defer l.mu.Unlock()
	run := l.badKeys[ip]
	if run == nil || now.Sub(run.start) > badKeyWindow || run.n < badKeyLimit {
		return nil
	}
	return &limitErr{run.start.Add(badKeyWindow).Sub(now), "Too many requests with a wrong API key from this address. Try again in a minute."}
}

// blockedCount is how many addresses are blocked now.
func (l *apiLimiter) blockedCount(now time.Time) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	n := 0
	for _, run := range l.badKeys {
		if now.Sub(run.start) <= badKeyWindow && run.n >= badKeyLimit {
			n++
		}
	}
	return n
}

// failed counts a wrong key from ip.
func (l *apiLimiter) failed(ip string, now time.Time) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.badKeys) > 4096 {
		for k, r := range l.badKeys {
			if now.Sub(r.start) > badKeyWindow {
				delete(l.badKeys, k)
			}
		}
	}
	run := l.badKeys[ip]
	if run == nil || now.Sub(run.start) > badKeyWindow {
		l.badKeys[ip] = &badKeyRun{n: 1, start: now}
		return
	}
	run.n++
}
