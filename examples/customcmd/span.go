package main

import (
	"regexp"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// The span-copy binder: the mechanism fix for the extractor's
// unseen-name wall. The trained extractor GENERATES slot values
// from its learned vocabulary, so a name it never saw comes out
// as a name it did ("quinn" -> "henrik") — four rounds of
// training-data changes scored exactly zero on fresh open-class
// names. This pass takes the other road: for the script the
// matcher chose, each declared slot's value is LIFTED VERBATIM
// from the user's own line, anchored on the slot words the
// script's declaration already carries ("from", "to", "in the
// … folder", "called", "at", "on"). A lifted value is by
// construction a substring of the line, word for word — it
// cannot be a substitution, because it was never generated.
//
// Precedence: a span found with its anchor present WINS over
// the extractor's value for that slot — the extractor's known
// failure is exactly there. The extractor still fills slots
// with no anchor in the line, and still handles lines the span
// pass cannot parse. Nothing is retrained; the matcher, the
// router, and every data file are untouched. The pass runs
// wherever values are seated: the single-line presentation
// (present, and through it the tie and correction flows), plan
// steps (buildPlan), and routine steps (expandRoutineSteps).

// spanWord is one whitespace-delimited word of a line, with
// the byte offsets of its raw form and a normalised form
// (lowercased, edge punctuation stripped) for matching.
type spanWord struct {
	raw        string
	norm       string
	start, end int
}

var reSpanWord = regexp.MustCompile(`\S+`)

// spanNorm normalises a word for anchor and stop matching:
// lowercase, with sentence punctuation and quotes trimmed
// from the edges. An internal dot survives ("cat.txt" keeps
// its dot — only the edges are trimmed).
func spanNorm(w string) string {
	return strings.Trim(strings.ToLower(w), ".,;:!?\"'")
}

func spanWords(line string) []spanWord {
	var out []spanWord
	for _, loc := range reSpanWord.FindAllStringIndex(line, -1) {
		raw := line[loc[0]:loc[1]]
		out = append(out, spanWord{raw: raw, norm: spanNorm(raw), start: loc[0], end: loc[1]})
	}
	return out
}

// slotAnchorPhrases gives, for a slot name, the anchor
// phrases that can introduce its value in a line, in
// priority order. The table is derived from the phrasings
// the shipped registry and the generic corpus actually use
// — it is deliberately small, not an ontology. A phrase is
// a sequence of normalised words matched adjacently.
func slotAnchorPhrases(slot string) [][]string {
	switch slot {
	case "from":
		return [][]string{{"from"}}
	case "to":
		return [][]string{{"over", "to"}, {"into"}, {"to"}}
	case "folder", "directory", "dir":
		return [][]string{{"inside"}, {"into"}, {"in"}}
	case "env":
		return [][]string{{"into"}, {"in"}}
	case "file":
		return [][]string{{"the", "file"}, {"a", "file"}, {"file"}}
	case "time":
		return [][]string{{"at"}}
	case "day", "date":
		return [][]string{{"on"}, {"for"}, {"at"}}
	case "title", "name":
		return [][]string{{"called"}, {"titled"}, {"named"}, {"known", "as"}, {"as"}}
	case "place":
		return [][]string{{"into"}, {"in"}, {"on"}}
	default:
		// An unknown slot's own name is its anchor of last
		// resort ("the shelf in …" style seats aside, a
		// line that names the slot usually means it).
		return [][]string{{slot}}
	}
}

// infinitiveHeads are the words after which a bare "to" is
// an infinitive marker, not a destination anchor: "i want
// to migrate the database from jim to jake" must anchor its
// to-slot on the second "to", not the first.
var infinitiveHeads = map[string]bool{
	"want": true, "going": true, "need": true, "needs": true,
	"have": true, "has": true, "got": true, "trying": true,
	"like": true, "love": true, "prefer": true, "plan": true,
	"planning": true, "about": true, "ready": true,
}

// anchorHit is one slot's claimed anchor: the word range
// [start, end) of the phrase in the line.
type anchorHit struct {
	slot       string
	start, end int
}

// claimAnchors finds each declared slot's anchor in the
// line: slots claim in declared order, each taking the
// earliest occurrence of its highest-priority phrase that
// passes the guards and does not overlap an anchor already
// claimed. A slot with no surviving occurrence gets no hit
// and is left to the extractor.
func claimAnchors(decl cmdmatch.Declaration, words []spanWord) []anchorHit {
	var hits []anchorHit
	claimed := func(start, end int) bool {
		for _, h := range hits {
			if start < h.end && h.start < end {
				return true
			}
		}
		return false
	}
	for _, slot := range decl.Slots {
		phrases := slotAnchorPhrases(slot)
		found := false
		for _, phrase := range phrases {
			if found {
				break
			}
			for i := 0; i+len(phrase) <= len(words); i++ {
				ok := true
				for j, pw := range phrase {
					if words[i+j].norm != pw {
						ok = false
						break
					}
				}
				if !ok || claimed(i, i+len(phrase)) {
					continue
				}
				// The infinitive guard: a bare "to" right
				// after a verb of wanting is not an anchor.
				if slot == "to" && len(phrase) == 1 && phrase[0] == "to" &&
					i > 0 && infinitiveHeads[words[i-1].norm] {
					continue
				}
				hits = append(hits, anchorHit{slot: slot, start: i, end: i + len(phrase)})
				found = true
				break
			}
		}
	}
	return hits
}

// spanFillers are words that carry no value content: they
// neither count as content when a span is being collected
// nor survive span cleanup at the edges.
var spanFillers = map[string]bool{
	"a": true, "an": true, "the": true, "one": true,
	"another": true, "my": true, "your": true, "his": true,
	"her": true, "its": true, "our": true, "their": true,
	"this": true, "that": true, "of": true, "please": true,
	"thanks": true,
}

// spanStopWords are the words that end a span once content
// has begun: the declaration's own content words. A span is
// the value the user seated between the anchors; when the
// description's own words resume ("… emeka ferry the
// crates"), the value has ended. Filler words and the
// decl's own anchor words are excluded — they are seats,
// not stops.
func spanStopWords(decl cmdmatch.Declaration) map[string]bool {
	stop := map[string]bool{}
	anchors := map[string]bool{}
	for _, slot := range decl.Slots {
		for _, phrase := range slotAnchorPhrases(slot) {
			for _, w := range phrase {
				anchors[w] = true
			}
		}
	}
	for _, w := range spanWords(decl.Description) {
		if spanFillers[w.norm] || anchors[w.norm] {
			continue
		}
		stop[w.norm] = true
	}
	return stop
}

// collectRegion returns the indices of the words forming
// the span that starts at word index from: it runs until
// the next claimed anchor, a word carrying sentence-stop
// punctuation (included, then the span ends), a
// description stop word (excluded — but only once real
// content has begun, so a leading seat word like the
// "folder" in "from folder jim" is collected and cleaned
// rather than killing the span), or the end of the line.
func collectRegion(words []spanWord, from int, hits []anchorHit, stop map[string]bool) []int {
	var idxs []int
	content := false
	for i := from; i < len(words); i++ {
		hit := false
		for _, h := range hits {
			if h.start == i {
				hit = true
				break
			}
		}
		if hit {
			break
		}
		if stop[words[i].norm] && content {
			break
		}
		idxs = append(idxs, i)
		if !spanFillers[words[i].norm] {
			content = true
		}
		// A word closing a clause ends the span after it:
		// "to jake, and …" seats jake, not the next clause.
		// (A dotted token like "cat.txt" ends with neither
		// mark and is collected as the value it is.)
		if strings.HasSuffix(words[i].raw, ",") || strings.HasSuffix(words[i].raw, ";") {
			break
		}
	}
	return idxs
}

// spanContainers are the container nouns that seat a place
// value in the corpus's phrasings ("in the X folder", "from
// the X depot", "into the X bin"): at a span's edge they
// are the seat, not the value, and are trimmed.
var spanContainers = map[string]bool{
	"folder": true, "directory": true, "dir": true,
	"depot": true, "shelf": true, "box": true, "bin": true,
	"desk": true, "board": true, "pantry": true, "tray": true,
	"well": true, "environment": true,
}

// placeSlot reports whether a slot belongs to the place
// family, whose spans get determiner and container trims.
func spanPlaceSlot(slot string) bool {
	switch slot {
	case "from", "to", "folder", "directory", "dir", "place", "env":
		return true
	}
	return false
}

// cleanSpan turns a collected region into the slot's value:
// the verbatim substring of the line spanning the kept
// words, with the seat words trimmed from its edges. Place
// spans drop leading determiners and container nouns and
// trailing containers and courtesies; title/name spans drop
// leading determiners; every span drops trailing courtesies
// and sentence punctuation, and a possessive 's on the last
// word ("quinn's shelf" seats the value quinn). The result
// is still a substring of the line — trimming only ever
// shortens it from the edges.
func cleanSpan(slot string, words []spanWord, idxs []int, line string) string {
	if len(idxs) == 0 {
		return ""
	}
	if slot == "file" {
		// A file value is one token ("cat.txt"): the word
		// right after the anchor, edge punctuation trimmed.
		return strings.Trim(words[idxs[0]].raw, ".,;:!?\"'")
	}
	lo, hi := 0, len(idxs)-1
	place := spanPlaceSlot(slot)
	for lo <= hi {
		n := words[idxs[lo]].norm
		if spanFillers[n] && n != "please" && n != "thanks" || place && spanContainers[n] {
			lo++
			continue
		}
		break
	}
	for hi >= lo {
		n := words[idxs[hi]].norm
		if n == "please" || n == "thanks" || place && (spanContainers[n] || spanFillers[n]) {
			hi--
			continue
		}
		break
	}
	if lo > hi {
		return ""
	}
	v := line[words[idxs[lo]].start:words[idxs[hi]].end]
	v = strings.Trim(v, ".,;:!?\"'")
	if strings.HasSuffix(v, "'s") {
		v = v[:len(v)-2]
	}
	return v
}

// quotedSpan reads a quoted value at the head of a region:
// when the first region word opens a quote, the value is
// the text between the quotes — still a verbatim substring
// of the line. It reports false when the region does not
// open with a quote or the quote never closes.
func quotedSpan(words []spanWord, idxs []int, line string) (string, bool) {
	if len(idxs) == 0 {
		return "", false
	}
	first := words[idxs[0]]
	var q byte
	if strings.HasPrefix(first.raw, "\"") {
		q = '"'
	} else if strings.HasPrefix(first.raw, "'") {
		q = '\''
	} else {
		return "", false
	}
	for _, ix := range idxs {
		w := words[ix]
		if strings.HasSuffix(w.raw, string(q)) && (ix != idxs[0] || len(w.raw) > 1) {
			return line[first.start+1 : w.end-1], true
		}
	}
	return "", false
}

// spanBind lifts a declaration's slot values verbatim from
// a line. It returns a map from slot name to value, holding
// only the slots it could anchor and lift; the rest are the
// extractor's to fill. Every value returned is a substring
// of the line, word for word.
func spanBind(decl cmdmatch.Declaration, line string) map[string]string {
	out := map[string]string{}
	if len(decl.Slots) == 0 {
		return out
	}
	words := spanWords(line)
	if len(words) == 0 {
		return out
	}
	hits := claimAnchors(decl, words)
	stop := spanStopWords(decl)
	hitOf := map[string]anchorHit{}
	for _, h := range hits {
		hitOf[h.slot] = h
	}
	for _, slot := range decl.Slots {
		h, ok := hitOf[slot]
		if !ok {
			continue
		}
		idxs := collectRegion(words, h.end, hits, stop)
		if v, quoted := quotedSpan(words, idxs, line); quoted {
			out[slot] = v
			continue
		}
		if v := cleanSpan(slot, words, idxs, line); v != "" {
			out[slot] = v
		}
	}

	// The between seat: "split the harvest between january
	// & march" carries two ordered values after "between"
	// and no from/to anchors at all. When at least two
	// slots are still unfilled, the two sides of the split
	// fill the first two of them, in declared order.
	var unfilled []string
	for _, slot := range decl.Slots {
		if _, ok := out[slot]; !ok {
			unfilled = append(unfilled, slot)
		}
	}
	if len(unfilled) >= 2 {
		for i, w := range words {
			if w.norm != "between" {
				continue
			}
			idxs := collectRegion(words, i+1, hits, stop)
			split := -1
			for k, ix := range idxs {
				if words[ix].norm == "and" || words[ix].norm == "&" {
					split = k
					break
				}
			}
			if split > 0 && split < len(idxs)-1 {
				if v := cleanSpan(unfilled[0], words, idxs[:split], line); v != "" {
					out[unfilled[0]] = v
				}
				if v := cleanSpan(unfilled[1], words, idxs[split+1:], line); v != "" {
					out[unfilled[1]] = v
				}
			}
			break
		}
	}

	// The env seat: the shipped registry's one env script
	// seats its value before the word "database" ("seed
	// the test database", "fill the staging database with
	// fake data") — a seat no in/into anchor reaches. The
	// word before "database", when it is a real word and
	// not a determiner, is the value.
	for _, slot := range decl.Slots {
		if slot != "env" {
			continue
		}
		if _, ok := out[slot]; ok {
			continue
		}
		for i, w := range words {
			if w.norm == "database" && i > 0 {
				prev := words[i-1]
				if !spanFillers[prev.norm] && prev.norm != "" {
					out[slot] = strings.Trim(prev.raw, ".,;:!?\"'")
				}
				break
			}
		}
	}
	return out
}

// bindSlotValues seats the values for a declaration from a
// line: the extractor's fields positionally, with any
// span-lifted value overriding its slot. When the span
// pass bound nothing, the extractor's fields are returned
// untouched — the baseline behaviour, byte for byte. The
// extractor's overflow fields (values it heard beyond the
// declared slots) are always kept: they are reported to
// the user as also-heard, never passed to the command —
// a pinned behaviour the span pass does not get to hide.
func bindSlotValues(decl cmdmatch.Declaration, line string, fields []string) []string {
	spans := spanBind(decl, line)
	if len(spans) == 0 {
		return fields
	}
	out := make([]string, len(decl.Slots))
	for i, slot := range decl.Slots {
		if v, ok := spans[slot]; ok {
			out[i] = v
		} else if i < len(fields) {
			out[i] = fields[i]
		}
	}
	for len(out) > 0 && out[len(out)-1] == "" {
		out = out[:len(out)-1]
	}
	if len(fields) > len(decl.Slots) {
		out = append(out, fields[len(decl.Slots):]...)
	}
	return out
}
