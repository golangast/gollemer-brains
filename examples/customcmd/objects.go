package main

import (
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/interpret"
)

// Session objects in the one prompt. Round 6 built the
// object layer in the interpret package — typed values
// remembered across lines, reference-back by name or as
// "the same folder", type conflicts reported never
// forced — but only library callers could use it: the
// REPL seated each line's values and forgot them. This
// file is the wiring that gives the prompt the same
// memory.
//
// The discipline is the library's own: references are
// read BEFORE the line's seats are noted (a line never
// refers to its own values), a remembered value fills
// only a seat the line itself left empty — or one where
// the binder lifted the reference phrase's own leftover
// ("same") instead of a value — and only when exactly
// one seat fits (ambiguity is silence, not a guess). A
// type conflict between a seat and the session's memory
// is printed, in the open, and the value stays seated.
// When a line carries no reference back and no conflict,
// nothing here prints or changes anything: the
// presentation is byte-identical to the one the REPL
// has always made.

// objectTable returns the session's object table,
// created on first use — demos built by hand in tests
// get one the same way the wired program does.
func (d *demo) objectTable() *interpret.ObjectTable {
	if d.objects == nil {
		d.objects = interpret.NewObjectTable()
	}
	return d.objects
}

// slotSeatType is the type evidence a slot carries on
// its own: its declared slot type when the declaration
// states one, else the slot family's default — the same
// first and last steps of interpret.ValueType's ladder.
func slotSeatType(decl cmdmatch.Declaration, slot string) string {
	if t := decl.SlotTypes[slot]; t != "" {
		return t
	}
	return interpret.SlotFamilyType(slot)
}

// typeFits reports whether a remembered object of type
// refType may fill a seat expecting seatType: equality,
// or satisfaction in either direction of the type
// lattice (a folder fits a place seat; a place-named
// memory fits a folder seat when the seat is the only
// candidate — the exactly-one rule in applySessionObjects
// is what keeps that from becoming a guess).
func typeFits(refType, seatType string) bool {
	if refType == "" || seatType == "" {
		return false
	}
	return refType == seatType ||
		cmdmatch.TypeSatisfies(refType, seatType) ||
		cmdmatch.TypeSatisfies(seatType, refType)
}

// isSameRemnant reports whether a seated value is the
// leftover of a "the same <type>" phrase rather than a
// value: the span binder anchors on the word before the
// phrase ("to the same folder") and lifts "same" — the
// seat word, not the thing seated. A value whose first
// word is "same" (after an optional "the") is that
// leftover.
func isSameRemnant(v string) bool {
	words := strings.Fields(strings.ToLower(v))
	if len(words) > 0 && words[0] == "the" {
		words = words[1:]
	}
	return len(words) > 0 && words[0] == "same"
}

// applySessionObjects runs the object layer over one
// presentation's bound values. It returns the values
// with any remembered fills seated, the set of slot
// indices a fill landed in (so the presentation can
// mark them "(remembered)"), and the type conflicts
// the line's seats raised against the session's memory
// — each a full report line, printed by the caller, the
// value staying seated either way.
//
// When nothing is filled and nothing conflicts, the
// returned slice is the input slice untouched: the
// caller's rendering is then exactly what it was before
// this layer existed.
func (d *demo) applySessionObjects(decl cmdmatch.Declaration, line string, values []string) ([]string, map[int]bool, []string) {
	tbl := d.objectTable()
	// References first, before this line's seats are
	// noted — the library's discipline, for the library's
	// reason.
	refs := tbl.ReferencesIn(line)

	// Work on a slot-length copy; the input is returned
	// as-is unless a fill actually lands.
	work := make([]string, len(decl.Slots))
	copy(work, values)
	remembered := map[int]bool{}
	filled := false

	seated := func(v string) bool {
		for _, w := range work {
			if w == v {
				return true
			}
		}
		return false
	}
	candidates := func(ref interpret.ObjectRef, remnant bool) []int {
		var out []int
		for i, slot := range decl.Slots {
			if remembered[i] {
				continue
			}
			if remnant {
				if !isSameRemnant(work[i]) {
					continue
				}
			} else if work[i] != "" {
				continue
			}
			if typeFits(ref.Type, slotSeatType(decl, slot)) {
				out = append(out, i)
			}
		}
		return out
	}
	fill := func(ref interpret.ObjectRef) {
		// The phrase's own seat first: a slot the binder
		// filled with the leftover "same" is where the
		// reference was spoken into. Only when no seat
		// holds the leftover does an empty seat qualify.
		cands := candidates(ref, true)
		if len(cands) == 0 {
			cands = candidates(ref, false)
		}
		if len(cands) != 1 {
			return // zero or ambiguous: silence, not a guess
		}
		work[cands[0]] = ref.Value
		remembered[cands[0]] = true
		filled = true
	}
	for _, ref := range refs {
		if strings.HasPrefix(ref.Via, "the same ") {
			fill(ref)
		}
	}
	for _, ref := range refs {
		if ref.Via != "named again" || seated(ref.Value) {
			continue
		}
		// A value named again fills an empty seat of a
		// fitting type — never a remnant seat (those
		// belong to "the same …" phrases).
		if cands := candidates(ref, false); len(cands) == 1 {
			work[cands[0]] = ref.Value
			remembered[cands[0]] = true
			filled = true
		}
	}

	out := values
	if filled {
		end := len(work)
		for end > 0 && work[end-1] == "" {
			end--
		}
		out = append([]string{}, work[:end]...)
		if len(values) > len(decl.Slots) {
			out = append(out, values[len(decl.Slots):]...)
		}
	}

	// Note the seats: each seated value becomes a session
	// object under its resolved type, and a seat that
	// contradicts the remembered type is a conflict —
	// reported, never forced; the value stays seated.
	var conflicts []string
	for i, slot := range decl.Slots {
		if work[i] == "" {
			continue
		}
		typ := interpret.ValueType(decl, slot, work[i], tbl.LookupType)
		if _, conflict := tbl.Note(work[i], typ, slot, decl.Name); conflict != "" {
			conflicts = append(conflicts, conflict)
		}
	}
	if !filled {
		return values, remembered, conflicts
	}
	return out, remembered, conflicts
}
