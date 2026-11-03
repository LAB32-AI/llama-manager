package main

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	maxChatSendBody   = 100 << 20 // maxAttachBytes as base64, plus the JSON around it
	maxAttachments    = 8
	maxImageBytes     = 10 << 20
	maxTextFileBytes  = 2 << 20
	maxMessageRunes   = 200_000
	chatEventsPing    = 15 * time.Second
	defaultChatsLimit = 500
)

var chatIDRe = regexp.MustCompile(`^[0-9a-f]{18}$`)

var imageMimes = map[string]bool{"image/png": true, "image/jpeg": true, "image/gif": true, "image/webp": true}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (ws *WebServer) chatErr(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, errChatNotFound):
		http.Error(w, "chat or message not found", http.StatusNotFound)
	case errors.Is(err, errChatBusy):
		http.Error(w, err.Error(), http.StatusConflict)
	default:
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
}

func (ws *WebServer) chatsAvailable(w http.ResponseWriter) bool {
	if ws.chats == nil {
		http.Error(w, "chat history is unavailable: the database could not be opened (see the llama-manager log)", http.StatusServiceUnavailable)
		return false
	}
	return true
}

// handleChats serves GET (list or ?q= search) and POST (create) on /api/chats.
func (ws *WebServer) handleChats(w http.ResponseWriter, r *http.Request) {
	if !ws.chatsAvailable(w) {
		return
	}
	switch r.Method {
	case http.MethodGet:
		limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
		if limit == 0 {
			limit = defaultChatsLimit
		}
		list, err := ws.chats.store.ListChats(r.URL.Query().Get("q"), limit)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		for i := range list {
			list[i].Streaming = ws.chats.Active(list[i].ID)
		}
		writeJSON(w, http.StatusOK, list)
	case http.MethodPost:
		var req struct {
			Title    string          `json:"title"`
			Instance string          `json:"instance"`
			Params   json.RawMessage `json:"params"`
		}
		r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
			return
		}
		c, err := ws.chats.store.CreateChat(req.Title, req.Instance, req.Params)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusCreated, c)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

type chatDetail struct {
	Chat     Chat          `json:"chat"`
	Messages []ChatMessage `json:"messages"`
}

func (ws *WebServer) chatDetail(id string) (chatDetail, error) {
	c, err := ws.chats.store.GetChat(id)
	if err != nil {
		return chatDetail{}, err
	}
	c.Streaming = ws.chats.Active(id)
	msgs, err := ws.chats.store.Messages(id)
	return chatDetail{Chat: c, Messages: msgs}, err
}

// handleChat routes /api/chats/{id}[/...].
func (ws *WebServer) handleChat(w http.ResponseWriter, r *http.Request) {
	if !ws.chatsAvailable(w) {
		return
	}
	parts := strings.Split(strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/chats/"), "/"), "/")
	id := parts[0]
	if !chatIDRe.MatchString(id) {
		http.NotFound(w, r)
		return
	}
	switch {
	case len(parts) == 1:
		ws.handleChatRoot(w, r, id)
	case len(parts) == 2 && parts[1] == "messages":
		ws.handleChatSend(w, r, id)
	case len(parts) == 3 && parts[1] == "messages":
		ws.handleChatMessage(w, r, id, parts[2])
	case len(parts) == 4 && parts[1] == "messages" && parts[3] == "regenerate":
		ws.handleChatRegenerate(w, r, id, parts[2])
	case len(parts) == 2 && parts[1] == "stop":
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, http.StatusOK, map[string]bool{"stopped": ws.chats.Stop(id)})
	case len(parts) == 2 && parts[1] == "events":
		ws.handleChatEvents(w, r, id)
	case len(parts) == 2 && parts[1] == "export":
		ws.handleChatExport(w, r, id)
	default:
		http.NotFound(w, r)
	}
}

func (ws *WebServer) handleChatRoot(w http.ResponseWriter, r *http.Request, id string) {
	switch r.Method {
	case http.MethodGet:
		d, err := ws.chatDetail(id)
		if err != nil {
			ws.chatErr(w, err)
			return
		}
		writeJSON(w, http.StatusOK, d)
	case http.MethodPatch:
		var p ChatPatch
		r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
		if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
			http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
			return
		}
		c, err := ws.chats.store.UpdateChat(id, p)
		if err != nil {
			ws.chatErr(w, err)
			return
		}
		writeJSON(w, http.StatusOK, c)
	case http.MethodDelete:
		ws.chats.Stop(id)
		if err := ws.chats.store.DeleteChat(id); err != nil {
			ws.chatErr(w, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

type sendAttachment struct {
	Kind string `json:"kind"`
	Name string `json:"name"`
	Mime string `json:"mime"`
	Data string `json:"data"` // base64
	// A document also carries its extracted text and who read it (tika or
	// browser); Data is then the original file. An image may carry the image
	// reader's description, with extract "reader".
	Text    *string `json:"text,omitempty"`
	Extract string  `json:"extract,omitempty"`
}

func decodeAttachments(in []sendAttachment) ([]Attachment, error) {
	if len(in) > maxAttachments {
		return nil, fmt.Errorf("attach at most %d files per message", maxAttachments)
	}
	var out []Attachment
	total := 0
	for _, a := range in {
		data, err := base64.StdEncoding.DecodeString(a.Data)
		if err != nil {
			return nil, fmt.Errorf("%s: attachment data must be base64", a.Name)
		}
		name := truncateRunes(strings.TrimSpace(strings.ReplaceAll(a.Name, "/", "_")), 120)
		if name == "" {
			name = "attachment"
		}
		att := Attachment{AttachmentMeta: AttachmentMeta{Kind: a.Kind, Name: name, Mime: a.Mime}, Data: data}
		switch {
		case a.Kind == "image":
			if !imageMimes[a.Mime] {
				return nil, fmt.Errorf("%s: images must be PNG, JPEG, GIF or WebP", name)
			}
			if len(data) > maxImageBytes {
				return nil, fmt.Errorf("%s is over %d MB; attach a smaller image", name, maxImageBytes>>20)
			}
			if a.Text != nil {
				if a.Extract != extractReader {
					return nil, fmt.Errorf("%s: an image's description must come from the image reader", name)
				}
				if len(*a.Text) > maxReaderTextBytes+4<<10 || !utf8.ValidString(*a.Text) {
					return nil, fmt.Errorf("%s: the image's description must be UTF-8 and at most %d KB", name, maxReaderTextBytes>>10)
				}
				if att.Text, _ = cleanDocText(*a.Text, false); att.Text != "" {
					att.Extract = extractReader
				}
			}
		case a.Kind == "file" && a.Text != nil:
			if a.Extract != "tika" && a.Extract != "browser" {
				return nil, fmt.Errorf("%s: extract must be tika or browser", name)
			}
			if len(data) > maxDocBytes {
				return nil, fmt.Errorf("%s is over %d MB; attach a smaller document", name, maxDocBytes>>20)
			}
			if len(*a.Text) > maxDocTextBytes+64<<10 || !utf8.ValidString(*a.Text) {
				return nil, fmt.Errorf("%s: the document's text must be UTF-8 and at most %d MB", name, maxDocTextBytes>>20)
			}
			att.Mime = tikaContentType(a.Mime)
			att.Text, _ = cleanDocText(*a.Text, att.Mime == "application/pdf")
			if att.Text == "" {
				return nil, fmt.Errorf("%s has no text to send", name)
			}
			att.Extract = a.Extract
		case a.Kind == "file":
			if len(data) > maxTextFileBytes {
				return nil, fmt.Errorf("%s is over %d MB; attach a smaller text file", name, maxTextFileBytes>>20)
			}
			if !utf8.Valid(data) {
				return nil, fmt.Errorf("%s isn't a text file; only text files, documents and images can be attached", name)
			}
			att.Mime = "text/plain"
		default:
			return nil, fmt.Errorf("%s: unknown attachment kind %q", name, a.Kind)
		}
		att.Size = int64(len(data))
		if total += len(data) + len(att.Text); total > maxAttachBytes {
			return nil, fmt.Errorf("the attachments add up to more than %d MB; send some of them in another message", maxAttachBytes>>20)
		}
		out = append(out, att)
	}
	return out, nil
}

// beginReply checks the instance, reserves the chat, runs mutate to store
// the turn, then starts generating. The browser gets the updated chat and
// follows the reply through /events.
func (ws *WebServer) beginReply(w http.ResponseWriter, chatID, instance string, mutate func(instance string) (turn, error)) {
	c, err := ws.chats.store.GetChat(chatID)
	if err != nil {
		ws.chatErr(w, err)
		return
	}
	if instance == "" {
		instance = c.Instance
	}
	inst, err := ws.chats.runningInstance(instance)
	if err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	if err := ws.chats.claim(chatID); err != nil {
		ws.chatErr(w, err)
		return
	}
	t, err := mutate(instance)
	if err != nil {
		ws.chats.release(chatID)
		ws.chatErr(w, err)
		return
	}
	c, err = ws.chats.store.GetChat(chatID)
	if err != nil {
		ws.chats.release(chatID)
		ws.chatErr(w, err)
		return
	}
	ws.chats.start(t, inst, parseChatParams(c.Params))
	d, err := ws.chatDetail(chatID)
	if err != nil {
		ws.chatErr(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, d)
}

func (ws *WebServer) handleChatSend(w http.ResponseWriter, r *http.Request, id string) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Content     string           `json:"content"`
		Instance    string           `json:"instance"`
		Attachments []sendAttachment `json:"attachments"`
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxChatSendBody)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}
	atts, err := decodeAttachments(req.Attachments)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	content := strings.TrimSpace(req.Content)
	if content == "" && len(atts) == 0 {
		http.Error(w, "message is empty", http.StatusBadRequest)
		return
	}
	if utf8.RuneCountInString(content) > maxMessageRunes {
		http.Error(w, "message is too long", http.StatusBadRequest)
		return
	}
	ws.beginReply(w, id, req.Instance, func(instance string) (turn, error) {
		return ws.chats.store.AppendTurn(id, content, instance, atts)
	})
}

// handleChatMessage: PUT edits a user message and replies again; PATCH
// selects which version of a reply is shown and sent back to the model.
func (ws *WebServer) handleChatMessage(w http.ResponseWriter, r *http.Request, chatID, msgID string) {
	switch r.Method {
	case http.MethodPut:
		var req struct {
			Content  string `json:"content"`
			Instance string `json:"instance"`
		}
		r.Body = http.MaxBytesReader(w, r.Body, maxChatBody)
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
			return
		}
		content := strings.TrimSpace(req.Content)
		if content == "" {
			http.Error(w, "message is empty", http.StatusBadRequest)
			return
		}
		ws.beginReply(w, chatID, req.Instance, func(instance string) (turn, error) {
			return ws.chats.store.EditUserMessage(chatID, msgID, content, instance)
		})
	case http.MethodPatch:
		var req struct {
			VI *int `json:"vi"`
		}
		r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.VI == nil {
			http.Error(w, `body must be {"vi": <version index>}`, http.StatusBadRequest)
			return
		}
		if ws.chats.Active(chatID) {
			http.Error(w, errChatBusy.Error(), http.StatusConflict)
			return
		}
		if err := ws.chats.store.SelectVariant(chatID, msgID, *req.VI); err != nil {
			ws.chatErr(w, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (ws *WebServer) handleChatRegenerate(w http.ResponseWriter, r *http.Request, chatID, msgID string) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Instance string `json:"instance"`
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxJSONBody)
	_ = json.NewDecoder(r.Body).Decode(&req) // the body is optional
	ws.beginReply(w, chatID, req.Instance, func(instance string) (turn, error) {
		return ws.chats.store.AddVariant(chatID, msgID, instance)
	})
}

func writeSSE(w http.ResponseWriter, event string, data any) error {
	b, _ := json.Marshal(data)
	_, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
	return err
}

// handleChatEvents streams the chat's reply as server-sent events: a
// snapshot of the text so far, then deltas, done, an optional title, and end.
// With nothing running it sends idle and closes.
func (ws *WebServer) handleChatEvents(w http.ResponseWriter, r *http.Request, id string) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	fmt.Fprint(w, "retry: 2000\n\n")
	snap, ch, unsub, ok := ws.chats.Subscribe(id)
	if !ok {
		writeSSE(w, "idle", map[string]any{})
		flusher.Flush()
		return
	}
	defer unsub()
	writeSSE(w, "snapshot", snap)
	flusher.Flush()
	ping := time.NewTicker(chatEventsPing)
	defer ping.Stop()
	for {
		select {
		case ev, open := <-ch:
			if !open {
				writeSSE(w, "end", map[string]any{})
				flusher.Flush()
				return
			}
			if writeSSE(w, ev.Type, ev.Data) != nil {
				return
			}
			flusher.Flush()
		case <-ping.C:
			if _, err := fmt.Fprint(w, ": ping\n\n"); err != nil {
				return
			}
			flusher.Flush()
		case <-r.Context().Done():
			return
		}
	}
}

func (ws *WebServer) handleChatExport(w http.ResponseWriter, r *http.Request, id string) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	d, err := ws.chatDetail(id)
	if err != nil {
		ws.chatErr(w, err)
		return
	}
	var b strings.Builder
	fmt.Fprintf(&b, "# %s\n\n", d.Chat.Title)
	for i, m := range d.Messages {
		if i > 0 {
			b.WriteString("\n\n---\n\n")
		}
		if m.Role == "user" {
			b.WriteString("**You**\n\n" + m.Content)
			for _, a := range m.Attachments {
				fmt.Fprintf(&b, "\n\n_Attached: %s_", a.Name)
			}
			continue
		}
		fmt.Fprintf(&b, "**%s**\n\n", m.Instance)
		if m.VI >= 0 && m.VI < len(m.Variants) {
			b.WriteString(m.Variants[m.VI].Content)
		}
	}
	b.WriteString("\n")
	name := strings.Trim(regexp.MustCompile(`[^A-Za-z0-9]+`).ReplaceAllString(strings.ToLower(d.Chat.Title), "-"), "-")
	if name == "" {
		name = "chat"
	}
	w.Header().Set("Content-Type", "text/markdown; charset=utf-8")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename="%s.md"`, name))
	_, _ = w.Write([]byte(b.String()))
}

// handleAttachment serves a stored attachment. Images are shown inline;
// ?as=text shows what the model read (for an image, its description);
// anything else downloads (a document as the original file), and nothing can
// run scripts.
func (ws *WebServer) handleAttachment(w http.ResponseWriter, r *http.Request) {
	if !ws.chatsAvailable(w) {
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	id := strings.TrimPrefix(r.URL.Path, "/api/attachments/")
	if !chatIDRe.MatchString(id) {
		http.NotFound(w, r)
		return
	}
	a, err := ws.chats.store.GetAttachment(id)
	if err != nil {
		ws.chatErr(w, err)
		return
	}
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Content-Security-Policy", "default-src 'none'; sandbox")
	w.Header().Set("Cache-Control", "private, max-age=86400, immutable")
	body := a.Data
	asText := r.URL.Query().Get("as") == "text"
	switch {
	case asText && a.Kind == "image" && a.Text != "":
		body = []byte(a.Text)
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Header().Set("Content-Disposition", contentDisposition("inline", a.Name+".txt"))
	case a.Kind == "image" && imageMimes[a.Mime]:
		w.Header().Set("Content-Type", a.Mime)
	case asText && a.Kind == "file":
		if a.isDoc() {
			body = []byte(a.Text)
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Header().Set("Content-Disposition", contentDisposition("inline", strings.TrimSuffix(a.Name, ".txt")+".txt"))
	case a.isDoc():
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Content-Disposition", contentDisposition("attachment", a.Name))
	default:
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Header().Set("Content-Disposition", contentDisposition("attachment", a.Name))
	}
	_, _ = w.Write(body)
}

// contentDisposition names a download; non-ASCII names use RFC 2231 so they
// survive the header.
func contentDisposition(kind, name string) string {
	if v := mime.FormatMediaType(kind, map[string]string{"filename": name}); v != "" {
		return v
	}
	return kind
}
