package cmdmatch

import (
	"sort"
	"strings"
)

// The relations layer: named, typed relationships a
// declaration can stand in, declared once and read in
// both directions. Until this round the system's
// relations were real but unnamed — the from/to slot
// anchors were source/destination roles, Needs was a
// requires edge, disjointness was a cannot-co-occur
// rule — and because they were unnamed, nothing could
// validate them, invert them, or reason over them. This
// file names them.
//
// A RelationSpec is the ontology of one relation type:
// its name, its INVERSE (the same fact read from the
// other side — moves_from and moves_to are one edge,
// not two), and its RANGE: the value type expected to
// sit in the slot the relation runs through. Domain is
// implicit: the subject of a command's relation is the
// command itself. Validation (interpret.ValidateRelations)
// checks a seated value against the range and REPORTS a
// violation — the value is never hidden, dropped, or
// silently reseated, the same pinned rule the
// extra-heard-values behaviour follows.

// Relation is one declared relation on a Declaration:
// the relation's name, the slot it runs through (empty
// when the relation is about the command as a whole),
// and its target (the far side, when the far side is a
// name rather than a seated value — a requires edge
// names the concept it requires).
type Relation struct {
	Name   string `json:"name"`
	Slot   string `json:"slot,omitempty"`
	Target string `json:"target,omitempty"`
}

// RelationSpec is one relation type's contract.
type RelationSpec struct {
	Name    string
	Inverse string // "" when the relation has no named inverse
	Range   string // the value type expected in the slot; "" = any
}

// relationSpecs is the relation vocabulary. It is
// deliberately small — the relations the repo's own
// machinery already used, named.
var relationSpecs = map[string]RelationSpec{
	// A value moves out of a place / into a place: the
	// from/to anchors, as one edge with two readings.
	"moves_from": {Name: "moves_from", Inverse: "moves_to", Range: "place"},
	"moves_to":   {Name: "moves_to", Inverse: "moves_from", Range: "place"},
	// The command acts on the seated value.
	"targets": {Name: "targets", Range: "object"},
	// The command cancels/stops the seated value's
	// thing — the polarity disjointness polices.
	"cancels": {Name: "cancels", Range: "object"},
	// The command runs another named command/concept
	// (a routine invoking its steps), or must be
	// preceded by it (the Needs edge, named).
	"invokes":  {Name: "invokes", Range: "command"},
	"requires": {Name: "requires", Range: "command"},
	// Containment, one edge read both ways.
	"contains": {Name: "contains", Inverse: "part_of", Range: "object"},
	"part_of":  {Name: "part_of", Inverse: "contains", Range: "object"},
}

// RelationSpecFor returns the spec of a relation name.
func RelationSpecFor(name string) (RelationSpec, bool) {
	s, ok := relationSpecs[name]
	return s, ok
}

// KnownRelation reports whether a name is a relation
// type the vocabulary knows.
func KnownRelation(name string) bool {
	_, ok := relationSpecs[name]
	return ok
}

// RelationNames returns the relation vocabulary, sorted.
func RelationNames() []string {
	out := make([]string, 0, len(relationSpecs))
	for n := range relationSpecs {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// InverseRelation returns the name a relation is read
// by from the other side — moves_from read backwards is
// moves_to — or "" when the relation has no named
// inverse (or is unknown). Declared once, usable both
// directions: a declaration that states moves_from has
// also stated moves_to, and a query in either name
// finds the same edge.
func InverseRelation(name string) string {
	if s, ok := relationSpecs[name]; ok {
		return s.Inverse
	}
	return ""
}

// The value-type vocabulary slot types and relation
// ranges speak in. Small on purpose: the types the
// repo's slots actually seat.
var valueTypes = map[string]bool{
	"person":  true,
	"folder":  true,
	"file":    true,
	"time":    true,
	"place":   true,
	"object":  true, // any concrete thing — the universal range
	"command": true, // a named command or concept
	"text":    true,
	"number":  true,
}

// KnownValueType reports whether a name is a value type
// the vocabulary knows.
func KnownValueType(t string) bool { return valueTypes[t] }

// TypeSatisfies reports whether a value of type got may
// sit where type want is expected. The lattice is tiny
// and explicit: everything concrete satisfies "object";
// a folder is a place; equality always satisfies. An
// unknown got ("" — the value's type could not be
// determined) satisfies nothing and proves nothing:
// callers treat it as no evidence, never as a violation.
func TypeSatisfies(got, want string) bool {
	if got == "" || want == "" {
		return false
	}
	if got == want || want == "object" {
		return true
	}
	if want == "place" && got == "folder" {
		return true
	}
	return false
}

// DeriveRelations names the relations a declaration
// already carries implicitly, from its own declared
// shape: the from/to slots ARE moves_from/moves_to, the
// file slot is a targets edge, the Needs list is a
// requires edge per need. Explicit Relations come first
// and win their (name, slot, target) key; derived ones
// fill what the declaration left unnamed. Teach-time
// inference uses this to write the obvious relations
// into the registry at the moment a command is taught,
// so what was implicit becomes inspectable data.
func DeriveRelations(d Declaration) []Relation {
	seen := map[Relation]bool{}
	var out []Relation
	add := func(r Relation) {
		if !seen[r] {
			seen[r] = true
			out = append(out, r)
		}
	}
	for _, r := range d.Relations {
		add(r)
	}
	for _, slot := range d.Slots {
		switch slot {
		case "from":
			add(Relation{Name: "moves_from", Slot: "from"})
		case "to":
			add(Relation{Name: "moves_to", Slot: "to"})
		case "file":
			add(Relation{Name: "targets", Slot: "file"})
		}
	}
	for _, need := range d.Needs {
		add(Relation{Name: "requires", Target: need})
	}
	return out
}

// HasRelation reports whether a declaration carries a
// relation — stated explicitly, or derivable from its
// shape — under a name or its inverse: the both-
// directions read. A query for moves_to finds a
// declaration that only ever stated moves_from.
func HasRelation(d Declaration, name string) bool {
	inv := InverseRelation(name)
	for _, r := range DeriveRelations(d) {
		if r.Name == name || (inv != "" && r.Name == inv) {
			return true
		}
	}
	return false
}

// relationsKey is DeriveRelations' dedup key rendering,
// kept for trace/test readability.
func relationsKey(rs []Relation) string {
	parts := make([]string, len(rs))
	for i, r := range rs {
		s := r.Name
		if r.Slot != "" {
			s += "(" + r.Slot + ")"
		}
		if r.Target != "" {
			s += "->" + r.Target
		}
		parts[i] = s
	}
	return strings.Join(parts, ", ")
}
