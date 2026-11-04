package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// What a reply looks like read aloud: markdown becomes plain sentences, code
// blocks and links are left out, and the text is cut into parts a voice model
// speaks in one request.

var (
	thinkBlockRe  = regexp.MustCompile(`(?s)<think>.*?</think>`)
	headingRe     = regexp.MustCompile(`^#{1,6}\s+`)
	listItemRe    = regexp.MustCompile(`^(?:[-*+•]|\d{1,3}[.)])\s+`)
	ruleRe        = regexp.MustCompile(`^(?:[-*_]\s*){3,}$`)
	tableSepRe    = regexp.MustCompile(`^\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?$`)
	mdImageRe     = regexp.MustCompile(`!\[[^\]]*\]\([^)]*\)`)
	mdLinkRe      = regexp.MustCompile(`\[([^\]]+)\]\([^)]*\)`)
	autoLinkRe    = regexp.MustCompile(`<(?:https?|mailto):[^>\s]+>`)
	bareURLRe     = regexp.MustCompile(`(?:https?://|www\.)\S+`)
	inlineCodeRe  = regexp.MustCompile("`+([^`]*)`+")
	strongRe      = regexp.MustCompile(`\*\*([^*]+)\*\*|__([^_]+)__`)
	emRe          = regexp.MustCompile(`\*([^*\s][^*]*)\*`)
	strikeRe      = regexp.MustCompile(`~~([^~]+)~~`)
	htmlTagRe     = regexp.MustCompile(`</?[A-Za-z][^>]*>`)
	spacesRe      = regexp.MustCompile(`[ \t\x{00A0}]+`)
	spaceBeforeRe = regexp.MustCompile(`\s+([,.;:!?])`)
)

// speechText turns a markdown reply into the text to speak, one paragraph or
// item per line. Code blocks, images, URLs and table rules are dropped.
func speechText(md string) string {
	s := thinkBlockRe.ReplaceAllString(md, "")
	if i := strings.Index(s, "<think>"); i >= 0 {
		s = s[:i] // reasoning that never closed
	}
	s = strings.ReplaceAll(s, "\r\n", "\n")
	var out []string
	fence := ""
	for _, line := range strings.Split(s, "\n") {
		t := strings.TrimSpace(line)
		if fence != "" {
			if strings.HasPrefix(t, fence) {
				fence = ""
			}
			continue
		}
		if strings.HasPrefix(t, "```") || strings.HasPrefix(t, "~~~") {
			fence = t[:3]
			continue
		}
		for strings.HasPrefix(t, ">") {
			t = strings.TrimSpace(t[1:])
		}
		if t == "" || ruleRe.MatchString(t) {
			out = append(out, "")
			continue
		}
		sentence := false
		switch {
		case strings.HasPrefix(t, "|"):
			if tableSepRe.MatchString(t) {
				continue
			}
			var cells []string
			for _, c := range strings.Split(strings.Trim(t, "|"), "|") {
				if c = strings.TrimSpace(c); c != "" {
					cells = append(cells, c)
				}
			}
			t, sentence = strings.Join(cells, ", "), true
		case headingRe.MatchString(t):
			t, sentence = headingRe.ReplaceAllString(t, ""), true
		case listItemRe.MatchString(t):
			t, sentence = listItemRe.ReplaceAllString(t, ""), true
		}
		t = speechInline(t)
		if t == "" {
			continue
		}
		if sentence {
			t = endSentence(t)
		}
		out = append(out, t)
	}
	// Lines of one paragraph run on; blank lines and items stay apart.
	var paras []string
	var cur []string
	flush := func() {
		if len(cur) > 0 {
			paras = append(paras, strings.Join(cur, " "))
			cur = nil
		}
	}
	for _, l := range out {
		if l == "" {
			flush()
			continue
		}
		cur = append(cur, l)
		if strings.ContainsRune(".!?…:;。！？", lastRune(l)) {
			flush()
		}
	}
	flush()
	return strings.TrimSpace(strings.Join(paras, "\n"))
}

func speechInline(t string) string {
	t = mdImageRe.ReplaceAllString(t, "")
	t = mdLinkRe.ReplaceAllString(t, "$1")
	t = autoLinkRe.ReplaceAllString(t, "")
	t = bareURLRe.ReplaceAllString(t, "")
	t = inlineCodeRe.ReplaceAllString(t, "$1")
	t = strongRe.ReplaceAllString(t, "$1$2")
	t = emRe.ReplaceAllString(t, "$1")
	t = strikeRe.ReplaceAllString(t, "$1")
	t = htmlTagRe.ReplaceAllString(t, "")
	t = strings.NewReplacer("*", "", "`", "", "#", "", "|", ", ", "\\", "").Replace(t)
	t = spacesRe.ReplaceAllString(t, " ")
	t = spaceBeforeRe.ReplaceAllString(t, "$1")
	return strings.Trim(strings.TrimSpace(t), ",")
}

func lastRune(s string) rune {
	r, _ := utf8.DecodeLastRuneInString(s)
	return r
}

// endSentence gives a heading, list item or table row a full stop, so the
// voice pauses after it instead of running into the next one.
func endSentence(t string) string {
	if strings.ContainsRune(".!?…:;。！？", lastRune(t)) {
		return t
	}
	return t + "."
}

func isSentenceEnd(r rune) bool { return strings.ContainsRune(".!?…。！？", r) }

// sentences splits text at sentence ends and line breaks.
func sentences(text string) []string {
	var out []string
	for _, para := range strings.Split(text, "\n") {
		rs := []rune(para)
		start := 0
		for i, r := range rs {
			if !isSentenceEnd(r) {
				continue
			}
			next := i + 1
			for next < len(rs) && (isSentenceEnd(rs[next]) || strings.ContainsRune(`"')]»”’`, rs[next])) {
				next++
			}
			cjk := r == '。' || r == '！' || r == '？'
			if next < len(rs) && !cjk && !unicode.IsSpace(rs[next]) {
				continue // 3.14, e.g., file.txt
			}
			if next <= start {
				continue
			}
			if s := strings.TrimSpace(string(rs[start:next])); s != "" {
				out = append(out, s)
			}
			start = next
		}
		if s := strings.TrimSpace(string(rs[min(start, len(rs)):])); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// speechParts groups sentences into parts of at most limit runes (first for
// the first part, so playback starts sooner). A sentence longer than that is
// cut at a comma or a space.
func speechParts(text string, first, limit int) []string {
	var parts []string
	var cur strings.Builder
	curLen := 0
	room := func() int {
		if len(parts) == 0 {
			return first
		}
		return limit
	}
	push := func() {
		if curLen > 0 {
			parts = append(parts, cur.String())
			cur.Reset()
			curLen = 0
		}
	}
	for _, s := range sentences(text) {
		for {
			n := utf8.RuneCountInString(s)
			if curLen > 0 && curLen+1+n > room() {
				push()
			}
			if n <= room() {
				if curLen > 0 {
					cur.WriteByte(' ')
					curLen++
				}
				cur.WriteString(s)
				curLen += n
				break
			}
			head, tail := cutLong(s, room())
			cur.WriteString(head)
			curLen = utf8.RuneCountInString(head)
			push()
			s = tail
		}
	}
	push()
	return parts
}

// cutLong splits s within n runes, after the last comma or else the last space.
func cutLong(s string, n int) (string, string) {
	rs := []rune(s)
	cut := -1
	for i := n - 1; i > n/3; i-- {
		if rs[i] == ',' || rs[i] == ';' || rs[i] == '，' {
			cut = i + 1
			break
		}
	}
	if cut < 0 {
		for i := n - 1; i > n/3; i-- {
			if unicode.IsSpace(rs[i]) {
				cut = i
				break
			}
		}
	}
	if cut < 0 {
		cut = n
	}
	return strings.TrimSpace(string(rs[:cut])), strings.TrimSpace(string(rs[cut:]))
}

// latinWords are frequent short words that tell Latin-script languages apart.
var latinWords = map[string][]string{
	"en": {"the", "and", "is", "of", "to", "you", "that", "it", "this", "with", "are", "for"},
	"de": {"der", "die", "und", "ist", "nicht", "das", "ich", "sie", "mit", "ein", "eine", "auf"},
	"fr": {"le", "la", "les", "et", "est", "une", "des", "que", "pas", "pour", "dans", "vous"},
	"es": {"el", "los", "las", "que", "es", "y", "una", "por", "para", "con", "del", "está"},
	"it": {"il", "che", "è", "non", "una", "per", "gli", "della", "sono", "con", "questo", "anche"},
	"pt": {"não", "que", "uma", "os", "para", "com", "é", "você", "isso", "mais", "das", "também"},
	"nl": {"het", "een", "en", "is", "niet", "van", "de", "dat", "je", "op", "zijn", "voor"},
	"pl": {"nie", "się", "jest", "że", "na", "to", "w", "jak", "dla", "czy", "ale", "jego"},
	"sv": {"och", "är", "att", "det", "inte", "som", "en", "på", "för", "med", "jag", "har"},
	"tr": {"ve", "bir", "bu", "için", "değil", "ile", "çok", "da", "de", "ne", "olarak", "gibi"},
}

// detectLang guesses the language code of text from its script, and for
// Latin script from common words. English is the fallback.
func detectLang(text string) string {
	var cyr, han, kana, hangul, arabic, hebrew, greek, latin int
	for _, r := range text {
		switch {
		case unicode.Is(unicode.Cyrillic, r):
			cyr++
		case unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r):
			kana++
		case unicode.Is(unicode.Han, r):
			han++
		case unicode.Is(unicode.Hangul, r):
			hangul++
		case unicode.Is(unicode.Arabic, r):
			arabic++
		case unicode.Is(unicode.Hebrew, r):
			hebrew++
		case unicode.Is(unicode.Greek, r):
			greek++
		case unicode.Is(unicode.Latin, r):
			latin++
		}
	}
	type script struct {
		lang string
		n    int
	}
	best := script{"", latin}
	for _, s := range []script{{"ru", cyr}, {"ja", kana + han}, {"ko", hangul}, {"ar", arabic}, {"he", hebrew}, {"el", greek}} {
		if s.n > best.n {
			best = s
		}
	}
	switch {
	case best.lang == "ja" && kana*4 < han:
		return "zh"
	case best.lang != "":
		return best.lang
	}
	words := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool { return !unicode.IsLetter(r) })
	score := map[string]int{}
	for lang, ws := range latinWords {
		set := map[string]bool{}
		for _, w := range ws {
			set[w] = true
		}
		for _, w := range words {
			if set[w] {
				score[lang]++
			}
		}
	}
	lang, top, second := "en", score["en"], 0
	for l, n := range score {
		if l == "en" {
			continue
		}
		if n > top {
			lang, second, top = l, top, n
		} else if n > second {
			second = n
		}
	}
	if lang != "en" && (top < 3 || float64(top) < 1.5*float64(second)) {
		return "en"
	}
	return lang
}

// isWAV reports whether b starts like a RIFF/WAVE file.
func isWAV(b []byte) bool {
	return len(b) >= 12 && string(b[0:4]) == "RIFF" && string(b[8:12]) == "WAVE"
}

var errNotWAV = errors.New("not a WAV file")

// wavChunks returns a WAV file's format chunk and its samples. A streamed
// file's unknown data size (0xFFFFFFFF) means the rest of the file.
func wavChunks(b []byte) (fmtChunk, data []byte, err error) {
	if !isWAV(b) {
		return nil, nil, errNotWAV
	}
	for p := 12; p+8 <= len(b); {
		id, size := string(b[p:p+4]), int(binary.LittleEndian.Uint32(b[p+4:p+8]))
		body := p + 8
		if body+size > len(b) {
			size = len(b) - body
		}
		switch id {
		case "fmt ":
			fmtChunk = b[body : body+size]
		case "data":
			data = b[body : body+size]
		}
		if fmtChunk != nil && data != nil {
			return fmtChunk, data, nil
		}
		p = body + size + size%2
	}
	return nil, nil, errNotWAV
}

// joinWAVs puts WAV files with the same format one after the other.
func joinWAVs(files [][]byte) ([]byte, error) {
	var format []byte
	var samples bytes.Buffer
	for _, f := range files {
		fc, data, err := wavChunks(f)
		if err != nil {
			return nil, err
		}
		if format == nil {
			format = fc
		} else if !bytes.Equal(format, fc) {
			return nil, errors.New("the parts have different audio formats")
		}
		samples.Write(data)
	}
	if format == nil {
		return nil, errNotWAV
	}
	var out bytes.Buffer
	le := binary.LittleEndian
	pad := samples.Len() % 2
	out.WriteString("RIFF")
	_ = binary.Write(&out, le, uint32(4+8+len(format)+len(format)%2+8+samples.Len()+pad))
	out.WriteString("WAVEfmt ")
	_ = binary.Write(&out, le, uint32(len(format)))
	out.Write(format)
	if len(format)%2 == 1 {
		out.WriteByte(0)
	}
	out.WriteString("data")
	_ = binary.Write(&out, le, uint32(samples.Len()))
	out.Write(samples.Bytes())
	if pad == 1 {
		out.WriteByte(0)
	}
	return out.Bytes(), nil
}
