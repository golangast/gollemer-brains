package interpret

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// The object layer: values with types, remembered.
// Before this round a seated value was a bare string,
// known for exactly one line. Now a value seated in a
// typed seat becomes an OBJECT: the session's table
// remembers it by the type it was seated as, and later
// lines can refer back to it — by naming it again, or
// as "the same folder" — and get the same object, with
// its type, instead of a fresh untyped string.
//
// A type conflict is reported, never forced: when a
// line seats a value as a folder that the session
// already knows as a file, the reading carries the
// mismatch as a violation and the value stays seated —
// the table keeps the FIRST type it learned, because
// the first seating is the one the session built on.

// Object is one remembered value: what it is, the type
// it carries, and the seat that introduced it.
type Object struct {
	Value   string
	Type    string
	Slot    string
	Command string
}

// ObjectRef is a reading's reference back to a known
// object: the value, its remembered type, and how the
// line referred to it ("named again", "the same folder").
type ObjectRef struct {
	Value string
	Type   string
	Via   string
}

// ObjectTable is a session's memory of the objects its
// lines have introduced, keyed by the value's
// normalised form. Order is kept: "the same folder"
// means the MOST RECENT folder, and recency is a fact
// about the session, not about a map.
type ObjectTable struct {
	byValue map[string]*Object
	order   []*Object
}

// NewObjectTable returns an empty session table.
func NewObjectTable() *ObjectTable {
	return &ObjectTable{byValue: map[string]*Object{}}
}

func objectKey(value string) string {
	return strings.ToLower(strings.Join(strings.Fields(value), " "))
}

// Note records a value seated in a typed seat. It
// returns the object as the table knows it and, when
// the seating's type conflicts with the type the table
// already holds, a conflict report — the object keeps
// its first type either way. A seating with no type
// teaches the table nothing new about an unknown value,
// and never conflicts.
func (t *ObjectTable) Note(value, typ, slot, command string) (Object, string) {
	key := objectKey(value)
	if o, ok := t.byValue[key]; ok {
		if typ != "" && o.Type != "" && typ != o.Type &&
			!cmdmatch.TypeSatisfies(typ, o.Type) && !cmdmatch.TypeSatisfies(o.Type, typ) {
			return *o, fmt.Sprintf("object type mismatch: %q is known this session as a %s (seated by %s), but this line seats it as a %s — reported, not forced; the value stays seated", value, o.Type, o.Command, typ)
		}
		return *o, ""
	}
	o := &Object{Value: value, Type: typ, Slot: slot, Command: command}
	t.byValue[key] = o
	t.order = append(t.order, o)
	return *o, ""
}

// Lookup returns the object a value names, if the
// session knows it.
func (t *ObjectTable) Lookup(value string) (Object, bool) {
	if o, ok := t.byValue[objectKey(value)]; ok {
		return *o, true
	}
	return Object{}, false
}

// LookupType is Lookup as the relations layer consumes
// it: the remembered type, or "".
func (t *ObjectTable) LookupType(value string) string {
	if o, ok := t.Lookup(value); ok {
		return o.Type
	}
	return ""
}

// MostRecentOfType returns the most recently introduced
// object whose type satisfies the wanted type — the
// referent of "the same folder".
func (t *ObjectTable) MostRecentOfType(want string) (Object, bool) {
	for i := len(t.order) - 1; i >= 0; i-- {
		if t.order[i].Type != "" && (t.order[i].Type == want || cmdmatch.TypeSatisfies(t.order[i].Type, want)) {
			return *t.order[i], true
		}
	}
	return Object{}, false
}

// All returns the session's objects in introduction
// order.
func (t *ObjectTable) All() []Object {
	out := make([]Object, len(t.order))
	for i, o := range t.order {
		out[i] = *o
	}
	return out
}

// ReferencesIn is referencesIn for hosts that seat
// values themselves (examples/customcmd's presentation
// path): the objects a line refers back to, computed
// against the table's current memory — call it BEFORE
// noting the line's own seats, exactly as Interpret
// does, or a line would "refer" to its own values.
func (t *ObjectTable) ReferencesIn(line string) []ObjectRef {
	return t.referencesIn(line)
}

// referencesIn finds the objects a line refers BACK to:
// known values it names again, and "the same <type>"
// phrases resolved against the table's memory. Values
// the line itself is about to seat for the first time
// are not references — the caller checks the table
// BEFORE noting the line's own seats.
func (t *ObjectTable) referencesIn(line string) []ObjectRef {
	if len(t.order) == 0 {
		return nil
	}
	words := SpanWords(line)
	norms := make([]string, len(words))
	for i, w := range words {
		norms[i] = w.Norm
	}
	var refs []ObjectRef
	seen := map[string]bool{}
	// "the same folder": the word after "same" names a
	// type; the referent is the most recent such object.
	for i := 0; i+1 < len(norms); i++ {
		if norms[i] != "same" {
			continue
		}
		typ := norms[i+1]
		if !cmdmatch.KnownValueType(typ) {
			continue
		}
		if o, ok := t.MostRecentOfType(typ); ok && !seen[objectKey(o.Value)] {
			seen[objectKey(o.Value)] = true
			refs = append(refs, ObjectRef{Value: o.Value, Type: o.Type, Via: "the same " + typ})
		}
	}
	// Named again: a known value's words appear in the
	// line, in order. Longest values first, so a two-word
	// object is claimed before its first word alone.
	for _, o := range t.order {
		key := objectKey(o.Value)
		if seen[key] {
			continue
		}
		parts := strings.Fields(key)
		if len(parts) == 0 || len(parts) > len(norms) {
			continue
		}
		for i := 0; i+len(parts) <= len(norms); i++ {
			match := true
			for j, p := range parts {
				if norms[i+j] != p {
					match = false
					break
				}
			}
			if match {
				seen[key] = true
				refs = append(refs, ObjectRef{Value: o.Value, Type: o.Type, Via: "named again"})
				break
			}
		}
	}
	return refs
}
