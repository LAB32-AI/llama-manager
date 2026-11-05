package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	imageFileTTL      = time.Hour // as long as OpenAI keeps its image URLs
	maxImageFileBytes = 256 << 20
)

// isOpenAIImageModel is true for the image model names OpenAI clients send
// by default; they get the first image model.
func isOpenAIImageModel(m string) bool {
	m = strings.ToLower(m)
	return strings.HasPrefix(m, "dall-e") || strings.HasPrefix(m, "gpt-image") || strings.HasPrefix(m, "chatgpt-image")
}

// v1ImageModel picks the image model a /v1 request names by instance or
// preset; OpenAI's names, or none, pick the first one (for an edit, the
// first that edits).
func (ws *WebServer) v1ImageModel(name string, edit bool) (imageEntry, *Instance, *apiError) {
	if ws.imageGen == nil {
		return imageEntry{}, nil, apiErrf(http.StatusNotFound, "model_not_found", "image generation is off; turn it on in the image_generation section of config.yaml")
	}
	e, ok := ws.imageGen.pick(name)
	if name == "" || (!ok && isOpenAIImageModel(name)) {
		e, ok = ws.imageGen.pick("")
		if edit {
			for _, x := range ws.imageGen.models {
				if x.Preset.Edit != editNone {
					e = x
					break
				}
			}
		}
	}
	if !ok {
		return imageEntry{}, nil, &apiError{status: http.StatusNotFound, code: "model_not_found", param: "model",
			msg: fmt.Sprintf("model not found: %s (image models: %s)", name, strings.Join(ws.imageGen.names(), ", "))}
	}
	inst := ws.mgr.Get(e.Name)
	if inst == nil {
		return imageEntry{}, nil, apiErrf(http.StatusNotFound, "model_not_found", "%s doesn't exist", e.Name)
	}
	if s := inst.State(); s != StateRunning {
		return imageEntry{}, nil, apiErrf(http.StatusServiceUnavailable, "model_not_ready", "%s is %s, not running", e.Name, s)
	}
	return e, inst, nil
}

// imageReply is how the caller wants the pictures back.
type imageReply struct {
	format string // "url", "b64_json", or "" for both
	stream bool
	edit   bool
	size   string
}

func checkImageFormat(f string) *apiError {
	if f != "" && f != "url" && f != "b64_json" {
		return &apiError{status: http.StatusBadRequest, param: "response_format", msg: "response_format must be url or b64_json"}
	}
	return nil
}

// takeImageOpts removes what only OpenAI's own models take (sd-server applies
// each model's own defaults) and returns how to answer.
func takeImageOpts(req map[string]any) (imageReply, *apiError) {
	var o imageReply
	if v, ok := req["response_format"]; ok && v != nil {
		o.format, _ = v.(string)
		if o.format == "" {
			o.format = "?"
		}
	}
	if err := checkImageFormat(o.format); err != nil {
		return o, err
	}
	o.stream, _ = req["stream"].(bool)
	if s, ok := req["size"].(string); ok {
		if s == "" || s == "auto" {
			delete(req, "size")
		} else {
			o.size = s
		}
	}
	for _, k := range []string{"response_format", "stream", "partial_images", "quality", "style", "background", "moderation", "user", "input_fidelity"} {
		delete(req, k)
	}
	return o, nil
}

// handleV1Images is OpenAI's image generation, forwarded to the image model's
// own /v1/images/generations. The manager adds its LoRAs (v1LoRAs), and
// "lora": [{"path", "multiplier"}] picks more, by id or name as /api/status
// lists them.
func (ws *WebServer) handleV1Images(w http.ResponseWriter, r *http.Request) {
	if !allowMethods(w, r, http.MethodPost) {
		return
	}
	body, aerr := readAPIBody(w, r, maxJSONBody)
	if aerr != nil {
		aerr.write(w)
		return
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil || req == nil {
		writeAPIError(w, http.StatusBadRequest, "", "", "We could not parse the JSON body of your request")
		return
	}
	m, _ := req["model"].(string)
	e, inst, aerr := ws.v1ImageModel(m, false)
	if aerr != nil {
		aerr.write(w)
		return
	}
	o, aerr := takeImageOpts(req)
	if aerr != nil {
		aerr.write(w)
		return
	}
	delete(req, "model")
	if err := e.v1LoRAs(req); err != nil {
		writeAPIError(w, http.StatusBadRequest, "", "", e.Name+": "+err.Error())
		return
	}
	data, _ := json.Marshal(req)
	up, err := http.NewRequestWithContext(r.Context(), http.MethodPost, inst.baseURL()+"/v1/images/generations", bytes.NewReader(data))
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, "", "", err.Error())
		return
	}
	up.Header.Set("Content-Type", "application/json")
	ws.relayImages(w, r, e.Name, up, o)
}

// handleV1ImageEdits is OpenAI's image edit (multipart: image[] or image,
// mask, prompt...), forwarded to sd-server's /v1/images/edits with the
// prompt's LoRAs checked like a generation's.
func (ws *WebServer) handleV1ImageEdits(w http.ResponseWriter, r *http.Request) {
	if !allowMethods(w, r, http.MethodPost) {
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxAPIBody)
	if err := r.ParseMultipartForm(16 << 20); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeAPIError(w, http.StatusRequestEntityTooLarge, "request_too_large", "", fmt.Sprintf("the request is over %d MB", maxAPIBody>>20))
			return
		}
		writeAPIError(w, http.StatusBadRequest, "", "", "send the edit as multipart/form-data with the picture in image[] (or image), like OpenAI's images.edit: "+err.Error())
		return
	}
	defer r.MultipartForm.RemoveAll()
	e, inst, aerr := ws.v1ImageModel(r.FormValue("model"), true)
	if aerr != nil {
		aerr.write(w)
		return
	}
	if e.Preset.Edit == editNone {
		var editors []string
		for _, x := range ws.imageGen.models {
			if x.Preset.Edit != editNone {
				editors = append(editors, x.Name)
			}
		}
		writeAPIError(w, http.StatusBadRequest, "model_not_supported", "model", fmt.Sprintf("%s (%s) makes images from a description only and can't edit; image models that edit: %s", e.Name, e.Preset.Label, orNone(editors)))
		return
	}
	files := append(r.MultipartForm.File["image[]"], r.MultipartForm.File["image"]...)
	if len(files) == 0 {
		writeAPIError(w, http.StatusBadRequest, "", "image", "image is required: the picture to edit, in image[] or image")
		return
	}
	if n := e.Preset.MaxRefs; n > 0 && len(files) > n {
		writeAPIError(w, http.StatusBadRequest, "", "image", fmt.Sprintf("%s edits at most %d pictures at once", e.Name, n))
		return
	}
	prompt := r.FormValue("prompt")
	if strings.TrimSpace(prompt) == "" {
		writeAPIError(w, http.StatusBadRequest, "", "prompt", "prompt is required: what to change")
		return
	}
	req := map[string]any{"prompt": prompt}
	if l := r.FormValue("lora"); l != "" {
		var v any
		if err := json.Unmarshal([]byte(l), &v); err != nil {
			writeAPIError(w, http.StatusBadRequest, "", "lora", `lora must be JSON like [{"path": "z-pixel-art", "multiplier": 0.8}]`)
			return
		}
		req["lora"] = v
	}
	if err := e.v1LoRAs(req); err != nil {
		writeAPIError(w, http.StatusBadRequest, "", "", e.Name+": "+err.Error())
		return
	}
	o := imageReply{format: r.FormValue("response_format"), stream: r.FormValue("stream") == "true", edit: true}
	if aerr := checkImageFormat(o.format); aerr != nil {
		aerr.write(w)
		return
	}
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	_ = mw.WriteField("prompt", req["prompt"].(string))
	for _, k := range []string{"n", "output_format", "output_compression"} {
		if v := r.FormValue(k); v != "" {
			_ = mw.WriteField(k, v)
		}
	}
	if s := r.FormValue("size"); s != "" && s != "auto" {
		_ = mw.WriteField("size", s)
		o.size = s
	}
	add := func(field string, fh *multipart.FileHeader) *apiError {
		f, err := fh.Open()
		if err != nil {
			return apiErrf(http.StatusBadRequest, "", "reading %s: %v", field, err)
		}
		defer f.Close()
		data, err := io.ReadAll(f)
		if err != nil {
			return apiErrf(http.StatusBadRequest, "", "reading %s: %v", field, err)
		}
		mime := http.DetectContentType(data)
		ext := imageExt(mime)
		if ext == "" {
			return &apiError{status: http.StatusBadRequest, param: field, msg: fmt.Sprintf("%s must be a PNG, JPEG or WebP picture", strings.TrimSuffix(field, "[]"))}
		}
		part, err := mw.CreatePart(textproto.MIMEHeader{
			"Content-Disposition": {fmt.Sprintf(`form-data; name=%q; filename=%q`, field, strings.TrimSuffix(field, "[]")+ext)},
			"Content-Type":        {mime},
		})
		if err != nil {
			return apiErrf(http.StatusInternalServerError, "", "%v", err)
		}
		_, _ = part.Write(data)
		return nil
	}
	for _, fh := range files {
		if aerr := add("image[]", fh); aerr != nil {
			aerr.write(w)
			return
		}
	}
	if masks := r.MultipartForm.File["mask"]; len(masks) > 0 {
		if aerr := add("mask", masks[0]); aerr != nil {
			aerr.write(w)
			return
		}
	}
	_ = mw.Close()
	up, err := http.NewRequestWithContext(r.Context(), http.MethodPost, inst.baseURL()+"/v1/images/edits", &buf)
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, "", "", err.Error())
		return
	}
	up.Header.Set("Content-Type", mw.FormDataContentType())
	ws.relayImages(w, r, e.Name, up, o)
}

func orNone(s []string) string {
	if len(s) == 0 {
		return "none"
	}
	return strings.Join(s, ", ")
}

func imageExt(mime string) string {
	switch mime {
	case "image/png":
		return ".png"
	case "image/jpeg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	}
	return ""
}

var zeroImageUsage = map[string]any{"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
	"input_tokens_details": map[string]int{"text_tokens": 0, "image_tokens": 0}}

// relayImages sends up to sd-server and answers in OpenAI's shape: each
// picture as b64_json, as a URL served for an hour, or (when the caller
// didn't say) both, so clients reading either field work. With "stream"
// the pictures come as image_generation.completed (image_edit.completed)
// events.
func (ws *WebServer) relayImages(w http.ResponseWriter, r *http.Request, name string, up *http.Request, o imageReply) {
	resp, err := chatClient.Do(up)
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		writeAPIError(w, http.StatusBadGateway, "", "", "proxying to "+name+": "+err.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		status := resp.StatusCode
		if status >= 500 {
			status = http.StatusBadGateway
		}
		writeAPIError(w, status, "", "", name+": "+sdError(resp))
		return
	}
	var out struct {
		Created int64 `json:"created"`
		Data    []struct {
			B64     string `json:"b64_json"`
			Revised string `json:"revised_prompt"`
		} `json:"data"`
		OutputFormat string `json:"output_format"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4*maxAPIBody)).Decode(&out); err != nil {
		writeAPIError(w, http.StatusBadGateway, "", "", name+" sent an answer llama-manager can't read")
		return
	}
	if out.Created == 0 {
		out.Created = time.Now().Unix()
	}
	items := make([]map[string]any, 0, len(out.Data))
	for _, d := range out.Data {
		data, err := base64.StdEncoding.DecodeString(d.B64)
		mime := http.DetectContentType(data)
		if err != nil || len(data) == 0 || imageExt(mime) == "" {
			writeAPIError(w, http.StatusBadGateway, "", "", name+" sent an image llama-manager can't read")
			return
		}
		if out.OutputFormat == "" {
			out.OutputFormat = strings.TrimPrefix(mime, "image/")
		}
		item := map[string]any{}
		if o.format != "url" || o.stream {
			item["b64_json"] = d.B64
		}
		if o.format != "b64_json" && !o.stream {
			item["url"] = ws.imageURL(r, data, mime)
		}
		if d.Revised != "" {
			item["revised_prompt"] = d.Revised
		}
		items = append(items, item)
	}
	if !o.stream {
		writeJSON(w, http.StatusOK, map[string]any{"created": out.Created, "data": items, "output_format": out.OutputFormat, "usage": zeroImageUsage})
		return
	}
	typ, size := "image_generation.completed", o.size
	if o.edit {
		typ = "image_edit.completed"
	}
	if size == "" {
		size = "auto"
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	for _, it := range items {
		ev, _ := json.Marshal(map[string]any{"type": typ, "b64_json": it["b64_json"], "created_at": out.Created, "size": size,
			"quality": "auto", "background": "auto", "output_format": out.OutputFormat, "usage": zeroImageUsage})
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", typ, ev)
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// imageURL keeps a picture for an hour and returns where to fetch it. The
// link is unguessable, so like OpenAI's it needs no API key.
func (ws *WebServer) imageURL(r *http.Request, data []byte, mime string) string {
	id := "img_" + randHex(16) + imageExt(mime)
	ws.imageFiles.put(id, data, mime)
	scheme := "http"
	if r.TLS != nil || strings.EqualFold(r.Header.Get("X-Forwarded-Proto"), "https") {
		scheme = "https"
	}
	return scheme + "://" + r.Host + "/v1/images/files/" + id
}

func (ws *WebServer) handleV1ImageFile(w http.ResponseWriter, r *http.Request) {
	if !allowMethods(w, r, http.MethodGet, http.MethodHead) {
		return
	}
	it, ok := ws.imageFiles.get(strings.TrimPrefix(r.URL.Path, "/v1/images/files/"))
	if !ok {
		writeAPIError(w, http.StatusNotFound, "", "", "this image has expired (pictures are kept for an hour) or never existed")
		return
	}
	h := w.Header()
	h.Set("Content-Type", it.mime)
	h.Set("Content-Length", strconv.Itoa(len(it.data)))
	h.Set("Cache-Control", "private, max-age=3600")
	h.Set("X-Content-Type-Options", "nosniff")
	h.Set("Content-Security-Policy", "default-src 'none'; sandbox")
	if r.Method == http.MethodGet {
		_, _ = w.Write(it.data)
	}
}

// blobCache keeps files in memory for a while, at most max bytes, dropping
// the oldest first.
type blobCache struct {
	mu    sync.Mutex
	max   int
	ttl   time.Duration
	size  int
	items map[string]*blobItem
	order []string // oldest first
}

type blobItem struct {
	data    []byte
	mime    string
	expires time.Time
}

func newBlobCache(max int, ttl time.Duration) *blobCache {
	return &blobCache{max: max, ttl: ttl, items: map[string]*blobItem{}}
}

func (c *blobCache) put(id string, data []byte, mime string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if old, ok := c.items[id]; ok {
		c.size -= len(old.data)
	} else {
		c.order = append(c.order, id)
	}
	c.items[id] = &blobItem{data: data, mime: mime, expires: time.Now().Add(c.ttl)}
	c.size += len(data)
	c.evictLocked(time.Now())
}

func (c *blobCache) get(id string) (*blobItem, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.evictLocked(time.Now())
	it, ok := c.items[id]
	return it, ok
}

func (c *blobCache) evictLocked(now time.Time) {
	for len(c.order) > 0 {
		id := c.order[0]
		it := c.items[id]
		if it != nil && c.size <= c.max && now.Before(it.expires) {
			return
		}
		c.order = c.order[1:]
		if it != nil {
			delete(c.items, id)
			c.size -= len(it.data)
		}
	}
}
