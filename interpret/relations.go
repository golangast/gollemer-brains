package interpret

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// The relations layer at read time: seated values are
// checked against the ranges of the relations their
// command STATED, and the check's outcome is part of
// the reading — a hop line in the trace when the value
// fits, a named violation when it does not. A violation
// never unseats the value: the binder's job is to seat
// what the user said, the relations layer's job is to
// SAY when what they said does not fit the declared
// shape. Reported, never hidden — the pinned rule.

// fileExtensions are the suffixes that make a bare value
// read as a file without any declaration saying so.
var fileExtensions = map[string]bool{
	"txt": true, "sh": true, "go": true, "json": true,
	"md": true, "log": true, "zip": true, "csv": true,
	"png": true, "jpg": true, "jpeg": true, "gif": true,
	"pdf": true, "db": true,
}

var (
	reTimeColon = regexp.MustCompile(`^\d{1,2}:\d{2}(:\d{2})?$`)
	reTimeAmPm  = regexp.MustCompile(`^\d{1,2}(:\d{2})?\s*(am|pm)$`)
)

// InferValueType reads a bare value's type off its
// shape: a clock reading is a time, a dotted name with
// a file extension is a file. Anything else is "" —
// unknown, which is no evidence either way: shape alone
// cannot tell a person from a folder, and the layer
// does not pretend otherwise.
func InferValueType(value string) string {
	v := strings.ToLower(strings.TrimSpace(value))
	if reTimeColon.MatchString(v) || reTimeAmPm.MatchString(v) {
		return "time"
	}
	if i := strings.LastIndex(v, "."); i > 0 && fileExtensions[v[i+1:]] {
		return "file"
	}
	return ""
}

// SlotFamilyType is the type a slot's NAME implies when
// nothing declares one: the place family seats places,
// the file slot seats a file, the time slot a time. It
// is a default, weaker than a declaration — validation
// uses declared and object-known types first.
func SlotFamilyType(slot string) string {
	switch slot {
	case "from", "to", "folder", "directory", "dir", "place", "env":
		return "place"
	case "file":
		return "file"
	case "time", "day", "date":
		return "time"
	case "person", "who":
		return "person"
	}
	return ""
}

// ValueType resolves a seated value's type from the
// strongest evidence available: the declaration's own
// slot type, then a caller-supplied lookup (the session
// object table's memory), then the value's shape, then
// the slot family's default. "" when nothing knows.
func ValueType(decl cmdmatch.Declaration, slot, value string, lookup func(string) string) string {
	if t := decl.SlotTypes[slot]; t != "" {
		return t
	}
	if lookup != nil {
		if t := lookup(value); t != "" {
			return t
		}
	}
	if t := InferValueType(value); t != "" {
		return t
	}
	return SlotFamilyType(slot)
}

// ValidateRelations checks every seated slot that a
// STATED relation runs through against that relation's
// range. It returns one violation line per misfit —
// each naming the relation, the expectation, the value,
// and the type it read as — and one hop line per fit,
// for the trace. A value whose type is unknown produces
// neither: no evidence is not a violation. The slots
// map is never modified.
func ValidateRelations(decl cmdmatch.Declaration, slots map[string]string, lookup func(string) string) (hops, violations []string) {
	for _, rel := range decl.Relations {
		if rel.Slot == "" {
			continue
		}
		value, seated := slots[rel.Slot]
		if !seated {
			continue
		}
		spec, ok := cmdmatch.RelationSpecFor(rel.Name)
		if !ok || spec.Range == "" {
			continue
		}
		got := ValueType(decl, rel.Slot, value, lookup)
		if got == "" {
			continue
		}
		if cmdmatch.TypeSatisfies(got, spec.Range) {
			hops = append(hops, fmt.Sprintf("relation: %s %s %s (%s, in range %s)", decl.Name, rel.Name, value, got, spec.Range))
		} else {
			violations = append(violations, fmt.Sprintf("relation violation: %s expects a %s in slot %s, but %q reads as %s — reported, not dropped; the value stays seated", rel.Name, spec.Range, rel.Slot, value, got))
		}
	}
	return hops, violations
}

// RelationCues reads the relation types a LINE speaks
// of, from its own words: from/to name the move edges,
// cancel/stop name cancels, part of names part_of, needs
// names requires. Sorted, deduplicated — the line's
// relational shape, the third factor of the relational
// composition (relations.go's sibling, ComposeRelational).
func RelationCues(line string) []string {
	words := LineObjectWords(line)
	has := func(ws ...string) bool {
		for _, w := range ws {
			if words[w] {
				return true
			}
		}
		return false
	}
	set := map[string]bool{}
	if has("from") {
		set["moves_from"] = true
	}
	if has("to", "into") {
		set["moves_to"] = true
	}
	if has("cancel", "cancels", "cancelled", "stop", "stops", "stopped", "never") {
		set["cancels"] = true
	}
	if has("contains", "containing", "inside") {
		set["contains"] = true
	}
	if has("needs", "requires", "require") {
		set["requires"] = true
	}
	out := make([]string, 0, len(set))
	for c := range set {
		out = append(out, c)
	}
	sort.Strings(out)
	return out
}
