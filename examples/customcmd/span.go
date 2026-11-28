package main

import (
	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/interpret"
)

// The span-copy binder — the pass that lifts a matched
// declaration's slot values verbatim from the user's
// line, anchored on the slot words the declaration
// carries — lives in the interpret package
// (interpret/span.go), under the programs that use it.
// This file is the adapter: the name customcmd has
// always used, plus bindSlotValues, the seating rule
// that combines the binder's values with the extractor
// brain's. Behaviour is unchanged.

func spanBind(decl cmdmatch.Declaration, line string) map[string]string {
	return interpret.SpanBind(decl, line)
}

// The binder's working parts, under the names the teach
// synthesis (synthesize.go) has always used for them.
type spanWord = interpret.SpanWord
type anchorHit = interpret.AnchorHit

func spanWords(line string) []spanWord { return interpret.SpanWords(line) }
func spanNorm(w string) string         { return interpret.SpanNorm(w) }

func slotAnchorPhrases(slot string) [][]string { return interpret.SlotAnchorPhrases(slot) }

func claimAnchors(decl cmdmatch.Declaration, words []spanWord) []anchorHit {
	return interpret.ClaimAnchors(decl, words)
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
