package main

import (
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Teach-time variant synthesis: when an entry is taught —
// through the declaration door, the teach-on-ask flow, or
// a correction landing a new example on an existing entry
// — the entry's own sentences sprout a few paraphrase
// variants on the spot, and the variants are stored as
// ordinary examples on the entry. One sentence from the
// user then matches in several shapes, with no retraining
// and no new machinery in the matcher: it reads examples,
// and variants ARE examples.
//
// Two moves only, both built from machinery the repo
// already has:
//
//   - COUSIN SUBSTITUTION: one word of a sentence is
//     replaced by a cousin from cmdmatch's synonym
//     families — the same table the matcher expands
//     profiles with. As an expanded word the cousin
//     already half-matched; as a written example word it
//     matches at full example weight, and it joins the
//     registry vocabulary (document frequencies, the typo
//     repair's neighbourhood) in its own right.
//   - VALUE SUBSTITUTION: a slot value heard in a
//     sentence is swapped for another generic value of
//     the same seat, drawn from the pools below — the
//     common values of examples/genericcommands.jsonl,
//     the corpus the extractor trains on (jim and jake,
//     holidays and library, the weekdays, the clock
//     times). A want the user teaches with one pair of
//     folder names then also matches the pairs the corpus
//     made ordinary.
//
// The hygiene rules are the feature's shape: at most
// maxSynthesizedExamples variants per teach or
// correction event; every variant differs from its source
// sentence in exactly one word or one value; no variant
// duplicates an existing example or another variant; and
// variants are generated only from the entry's ORIGINAL
// sentences — never from a synthesized one, so the set
// cannot drift generation over generation.
//
// Two kinds of words are never cousin-substituted: the
// entry's slot ANCHOR words (the "file" in "delete the
// file …" is a seat, not a topic) and the slot VALUES
// heard in the sentence (a folder the user named
// "photos" is a value, not the topic word "photos").
// Descriptions keep their exact shape for the same
// reason — synthesis only ever ADDS examples; the
// description itself is never rewritten.

// maxSynthesizedExamples caps the variants one teach or
// correction event may add to an entry.
const maxSynthesizedExamples = 6

// The value pools, by seat: the generic corpus's common
// values (see the file header). The file seat has no
// pool — the corpus carries no generic file values (its
// file shapes seat names), so value substitution simply
// does not fire for it.
var (
	placeValues = []string{"jim", "jake", "holidays", "photos", "backup", "library", "garden", "vault", "studio", "attic", "trips", "birthdays", "winter"}
	dayValues   = []string{"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
	timeValues  = []string{"7am", "9am", "12pm", "1pm", "3pm", "6pm"}
	nameValues  = []string{"lucy", "emma", "nora", "ruby", "iris", "hugo", "omar", "adam", "ella", "vera"}
)

// seatValuePool returns the generic values a slot's seat
// draws from, or nil when the seat has none.
func seatValuePool(slot string) []string {
	switch slot {
	case "from", "to", "folder", "directory", "dir", "place", "env":
		return placeValues
	case "day", "date":
		return dayValues
	case "time":
		return timeValues
	case "title", "name":
		return nameValues
	}
	return nil
}

// synthesizeExamples generates the variants for a NEW
// entry: the sources are its description and its examples
// as declared (originals only — at the persist hooks,
// nothing synthesized exists yet).
func synthesizeExamples(decl cmdmatch.Declaration) []string {
	existing := map[string]bool{}
	var sources []string
	add := func(s string) {
		if s != "" && !existing[s] {
			existing[s] = true
			sources = append(sources, s)
		}
	}
	add(decl.Description)
	for _, ex := range decl.Examples {
		add(ex)
	}
	return synthesizeFrom(decl, sources, existing)
}

// synthesizeFrom is the generator's core: variants of the
// given source sentences, in the context of decl (its
// slots anchor and protect, its description supplies the
// span pass's stop words), skipping anything already in
// existing. Value substitutions are emitted first — they
// are the scarcer move, since a generic description seats
// no values at all — and cousin substitutions fill the
// remaining budget.
func synthesizeFrom(decl cmdmatch.Declaration, sources []string, existing map[string]bool) []string {
	seen := map[string]bool{}
	for s := range existing {
		seen[s] = true
	}
	var out []string
	emit := func(v string) bool {
		if v != "" && !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
		return len(out) >= maxSynthesizedExamples
	}
	for _, src := range sources {
		if valueVariants(decl, src, emit) {
			return out
		}
	}
	for _, src := range sources {
		if cousinVariants(decl, src, emit) {
			return out
		}
	}
	return out
}

// valueVariants emits src with one slot value swapped for
// a pool value of the same seat — at most two swaps per
// seat, and never a value another seat of the same
// sentence already holds ("from jim to jake" must not
// sprout "from jake to jake").
func valueVariants(decl cmdmatch.Declaration, src string, emit func(string) bool) bool {
	spans := spanBind(decl, src)
	if len(spans) == 0 {
		return false
	}
	bound := map[string]bool{}
	for _, v := range spans {
		bound[v] = true
	}
	for _, slot := range decl.Slots {
		v, ok := spans[slot]
		if !ok {
			continue
		}
		emitted := 0
		for _, pv := range seatValuePool(slot) {
			if pv == v || bound[pv] {
				continue
			}
			if nv := replaceValue(decl, src, slot, v, pv); nv != "" {
				if emit(nv) {
					return true
				}
				emitted++
				if emitted >= 2 {
					break
				}
			}
		}
	}
	return false
}

// replaceValue swaps the occurrence of v that the slot's
// anchor seats — the one at or after the anchor's end —
// for pv. Searching from the anchor keeps a value that
// also appears earlier in the sentence (as a topic word,
// say) from being the one replaced.
func replaceValue(decl cmdmatch.Declaration, src, slot, v, pv string) string {
	start := 0
	words := spanWords(src)
	for _, h := range claimAnchors(decl, words) {
		if h.slot == slot {
			start = words[h.end-1].end
		}
	}
	i := strings.Index(src[start:], v)
	if i < 0 {
		return ""
	}
	return src[:start+i] + pv + src[start+i+len(v):]
}

// cousinVariants emits src with one word at a time
// replaced by each of its cousins, in table order,
// skipping protected words (slot anchors and heard slot
// values — see the file header).
func cousinVariants(decl cmdmatch.Declaration, src string, emit func(string) bool) bool {
	protected := protectedWords(decl, src)
	for _, w := range spanWords(src) {
		if protected[w.norm] {
			continue
		}
		for _, cousin := range cmdmatch.Cousins(w.norm) {
			if emit(src[:w.start] + surfaceForm(cousin, w.raw) + src[w.end:]) {
				return true
			}
		}
	}
	return false
}

// protectedWords returns the normalised words of src that
// cousin substitution must not touch: every anchor word
// of the entry's slots, and every word of every slot
// value the span pass hears in src.
func protectedWords(decl cmdmatch.Declaration, src string) map[string]bool {
	out := map[string]bool{}
	for _, slot := range decl.Slots {
		for _, phrase := range slotAnchorPhrases(slot) {
			for _, w := range phrase {
				out[w] = true
			}
		}
	}
	for _, v := range spanBind(decl, src) {
		for _, w := range strings.Fields(v) {
			out[spanNorm(w)] = true
		}
	}
	return out
}

// surfaceForm dresses a cousin (written in the table's
// base form) in the replaced word's clothes: a plural
// original takes a plural cousin ("photos" → "pictures",
// not "picture"), and a capitalised original keeps its
// capital. Crude, and enough — the matcher stems both
// sides anyway; this only keeps the stored examples
// readable as sentences.
func surfaceForm(cousin, raw string) string {
	s := cousin
	if strings.HasSuffix(spanNorm(raw), "s") && !strings.HasSuffix(s, "s") {
		s += "s"
	}
	if len(raw) > 0 && raw[0] >= 'A' && raw[0] <= 'Z' && s != "" {
		s = strings.ToUpper(s[:1]) + s[1:]
	}
	return s
}
