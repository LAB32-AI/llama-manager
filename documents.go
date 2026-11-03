package main

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// Documents (PDF, Word, Excel...) attached in the chat are stored as the
// original file plus its text; only the text goes to the model. Tika reads
// them when config.yaml has a tika section, the browser otherwise or when
// Tika is down. Either way the browser sends the text with the message.
const (
	maxDocBytes     = 32 << 20
	maxDocTextBytes = maxTextFileBytes
	maxAttachBytes  = 64 << 20 // every attachment of one message, originals and text
)

type docStatus struct {
	Tika          *tikaStatus `json:"tika"` // nil: not set up, the browser reads documents
	MaxDocBytes   int         `json:"max_doc_bytes"`
	MaxTextBytes  int         `json:"max_text_bytes"`
	MaxImageBytes int         `json:"max_image_bytes"`
	MaxFiles      int         `json:"max_files"`
	MaxTotalBytes int         `json:"max_total_bytes"`
}

type tikaStatus struct {
	URL     string `json:"url"`
	Checked bool   `json:"checked"`
	OK      bool   `json:"ok"`
	Version string `json:"version,omitempty"`
	Error   string `json:"error,omitempty"`
}

func (ws *WebServer) docStatus() docStatus {
	d := docStatus{MaxDocBytes: maxDocBytes, MaxTextBytes: maxDocTextBytes, MaxImageBytes: maxImageBytes, MaxFiles: maxAttachments, MaxTotalBytes: maxAttachBytes}
	if ws.tika != nil {
		st := ws.tika.Status()
		d.Tika = &tikaStatus{URL: ws.tika.shown, Checked: !st.At.IsZero(), OK: st.OK, Version: st.Version, Error: st.Err}
	}
	return d
}

func writeExtractErr(w http.ResponseWriter, status int, reason, msg string) {
	writeJSON(w, status, map[string]string{"error": msg, "reason": reason})
}

// handleExtract reads one document with Tika: POST the file as the body,
// its type as Content-Type and its URL-escaped name in X-File-Name.
func (ws *WebServer) handleExtract(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name, err := url.PathUnescape(r.Header.Get("X-File-Name"))
	if err != nil || strings.TrimSpace(name) == "" {
		name = "the document"
	}
	name = truncateRunes(strings.TrimSpace(name), 120)
	if ws.tika == nil {
		writeExtractErr(w, http.StatusServiceUnavailable, "off", "Tika isn't set up on this server; the browser reads documents itself.")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxDocBytes)
	data, err := io.ReadAll(r.Body)
	var tooBig *http.MaxBytesError
	switch {
	case errors.As(err, &tooBig):
		writeExtractErr(w, http.StatusRequestEntityTooLarge, "too_large", fmt.Sprintf("%s is over %d MB. Attach a smaller document.", name, maxDocBytes>>20))
		return
	case err != nil:
		writeExtractErr(w, http.StatusBadRequest, "unreadable", "reading the upload: "+err.Error())
		return
	case len(data) == 0:
		writeExtractErr(w, http.StatusBadRequest, "unreadable", name+" is empty.")
		return
	}
	res, err := ws.tika.Extract(r.Context(), data, name, r.Header.Get("Content-Type"))
	var ee *extractError
	switch {
	case errors.As(err, &ee):
		writeExtractErr(w, ee.Status, ee.Reason, ee.Msg)
	case err != nil:
		writeExtractErr(w, http.StatusBadGateway, "unreachable", err.Error())
	default:
		writeJSON(w, http.StatusOK, res)
	}
}

// fence wraps text in a code fence longer than any run of backticks inside
// it, so a document can't close the fence early.
func fence(text string) string {
	longest, run := 0, 0
	for i := 0; i < len(text); i++ {
		if text[i] == '`' {
			run++
			longest = max(longest, run)
		} else {
			run = 0
		}
	}
	f := strings.Repeat("`", max(3, longest+1))
	return f + "\n" + strings.TrimRight(text, "\n") + "\n" + f
}
