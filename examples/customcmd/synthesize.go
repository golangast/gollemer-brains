package main

import (
	"regexp"
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

// Two more moves widen the shapes one taught sentence
// covers, both word-preserving or seat-filling rather
// than substitutive:
//
//   - GENERIC SEATING: a description whose from/to seat
//     is generic ("copy the photos from one folder to
//     another") seats no values, so the value move above
//     never fires on it — yet the values are exactly the
//     words a later want will carry. When the entry
//     declares both seats, the generic seat phrase is
//     rewritten with pool value pairs ("… from jim to
//     jake", "… from holidays to backup"), so the pool's
//     ordinary values join the profile as example words
//     in the seats they belong in.
//   - REQUEST SHAPES: the same want arrives as a bare
//     imperative, a question ("can you …"), a short form
//     with the slot tail dropped, a rearrangement with
//     the to-phrase before the from-phrase, or — from
//     the teach-on-ask door, whose descriptions ARE want
//     lines — framed by "i want to …", whose stripped
//     imperative is the shape later wants take. Each
//     shape is stored verbatim; the matcher stems and
//     stopwords its way past the differences it can, and
//     the stored set simply covers the shapes it cannot.

// maxSynthesizedExamples caps the variants one teach or
// correction event may add to an entry. Six was the
// first round's cap, and the cousin move alone could
// fill it (a verb-rich description's cousins were cut
// at six, and a seat or shape move would have had no
// room at all); ten covers the value, seat and shape
// moves whole and leaves the cousins the rest.
const maxSynthesizedExamples = 10

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
// existing. The emission order spends the budget on the
// scarcest moves first: value substitutions (a generic
// description seats no values at all, so they are rare),
// generic seating (the same scarcity, one level up —
// the values a generic description's wants will carry),
// and the request shapes (a handful per sentence).
// Cousin substitutions fill what budget remains: they
// are the most plentiful move and the most redundant —
// the matcher's own expansion already half-matches a
// cousin, where nothing else covers a missing value or
// an unstored shape at all.
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
		if seatVariants(decl, src, emit) {
			return out
		}
	}
	for _, src := range sources {
		if shapeVariants(decl, src, emit) {
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
		if h.Slot == slot {
			start = words[h.End-1].End
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
		if protected[w.Norm] {
			continue
		}
		for _, cousin := range cmdmatch.Cousins(w.Norm) {
			if emit(src[:w.Start] + surfaceForm(cousin, w.Raw) + src[w.End:]) {
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

// reGenericSeat matches a generic from/to seat phrase —
// "from one folder to another", "from a folder to a
// folder" — the shape a declared description uses when
// it names no values of its own.
var reGenericSeat = regexp.MustCompile(`(?i)\bfrom\s+(one|a|the)\s+(folder|place|directory|dir)\s+to\s+(another|one|a|the)(\s+(folder|place|directory|dir))?\b`)

// seatVariants emits src with its generic from/to seat
// rewritten to seat pool value pairs — at most two pairs,
// drawn in pool order, skipping any pool value the
// source already uses as a word (a folder the sentence
// names "photos" as its topic is not also seated as a
// value). Only an entry declaring both seats earns the
// move: the pairs land in the seats the declaration
// itself names.
func seatVariants(decl cmdmatch.Declaration, src string, emit func(string) bool) bool {
	hasFrom, hasTo := false, false
	for _, s := range decl.Slots {
		if s == "from" {
			hasFrom = true
		}
		if s == "to" {
			hasTo = true
		}
	}
	if !hasFrom || !hasTo {
		return false
	}
	loc := reGenericSeat.FindStringIndex(src)
	if loc == nil {
		return false
	}
	used := map[string]bool{}
	for _, w := range spanWords(src) {
		used[w.Norm] = true
	}
	var pool []string
	for _, v := range placeValues {
		if !used[v] {
			pool = append(pool, v)
		}
	}
	for i := 0; i+1 < len(pool) && i < 4; i += 2 {
		v := src[:loc[0]] + "from " + pool[i] + " to " + pool[i+1] + src[loc[1]:]
		if emit(v) {
			return true
		}
	}
	return false
}

// shapeVariants emits the request shapes of src: the
// question form, the from/to rearrangement, the short
// form with the slot tail dropped, and the bare
// imperative under a want framing. Each is the source's
// own words re-dressed — no word is substituted, so the
// anchor and value protections the substitution moves
// observe are preserved by construction (a rearranged
// value travels with its anchor; a dropped tail drops
// its values whole rather than altering them).
func shapeVariants(decl cmdmatch.Declaration, src string, emit func(string) bool) bool {
	lower := strings.ToLower(src)
	if strings.HasPrefix(lower, "please ") {
		if emit("can you " + src[len("please "):]) {
			return true
		}
	} else if !strings.HasPrefix(lower, "can you") && !strings.HasPrefix(lower, "could you") {
		if emit("can you " + src) {
			return true
		}
	}
	if v := rearrangedForm(decl, src); v != "" {
		if emit(v) {
			return true
		}
	}
	if v := shortForm(decl, src); v != "" {
		if emit(v) {
			return true
		}
	}
	if v := imperativeForm(src); v != "" {
		if emit(v) {
			return true
		}
	}
	return false
}

// rearrangedForm swaps src's from-phrase and to-phrase:
// "copy the photos from one folder to another" becomes
// "copy the photos to another from one folder" — the
// order a want takes when the destination is named
// first. Both anchors must be claimed, from before to,
// and the to-phrase must carry at least one word beyond
// its anchor.
func rearrangedForm(decl cmdmatch.Declaration, src string) string {
	words := spanWords(src)
	var fromHit, toHit *anchorHit
	for _, h := range claimAnchors(decl, words) {
		h := h
		switch h.Slot {
		case "from":
			fromHit = &h
		case "to":
			toHit = &h
		}
	}
	if fromHit == nil || toHit == nil || fromHit.Start >= toHit.Start {
		return ""
	}
	if len(words)-toHit.Start < 2 {
		return ""
	}
	var parts []string
	for _, w := range words[:fromHit.Start] {
		parts = append(parts, w.Raw)
	}
	for _, w := range words[toHit.Start:] {
		parts = append(parts, w.Raw)
	}
	for _, w := range words[fromHit.Start:toHit.Start] {
		parts = append(parts, w.Raw)
	}
	return strings.Join(parts, " ")
}

// shortForm drops src's slot tail: everything from the
// from-anchor on. "copy the photos from one folder to
// another" becomes "copy the photos" — the shape of a
// want that names no seats. The head must keep at least
// two words and something must actually be dropped.
func shortForm(decl cmdmatch.Declaration, src string) string {
	words := spanWords(src)
	for _, h := range claimAnchors(decl, words) {
		if h.Slot != "from" {
			continue
		}
		if h.Start < 2 || h.Start >= len(words) {
			return ""
		}
		var parts []string
		for _, w := range words[:h.Start] {
			parts = append(parts, w.Raw)
		}
		return strings.Join(parts, " ")
	}
	return ""
}

// wantFramings are the first-person frames a teach-on-ask
// want line can carry; the imperative under the frame is
// the shape later wants take.
var wantFramings = []string{"i want to ", "i need to ", "i would like to ", "please "}

// imperativeForm strips a want framing from src's head,
// or returns "" when src wears none.
func imperativeForm(src string) string {
	lower := strings.ToLower(src)
	for _, f := range wantFramings {
		if strings.HasPrefix(lower, f) {
			return src[len(f):]
		}
	}
	return ""
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
