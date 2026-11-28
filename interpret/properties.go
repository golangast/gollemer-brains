package interpret

import (
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Concept properties: the small declared fact-set a
// concept carries ALONGSIDE the examples that define
// it — its default relations, and the type of the
// object it acts on. Identity stays example-defined:
// the matcher still matches the description and
// examples, the classifier still reads the rows. The
// properties are additions a host (or the relations
// layer) can reason over — what kind of thing this
// meaning moves, cancels, or acts on — without
// re-deriving them from prose every time.
//
// A property is stated when the declaration states it,
// and derived when it does not: from a declaration's
// ActsOn hint and slot shape, or from a label's own
// factorized name (delete_file acts on a file; its verb
// is a targets edge). Derivation is deterministic and
// uses the same vocabularies as everything else.

// ConceptProperties is one concept's property set.
type ConceptProperties struct {
	// ObjectType is the type of the thing the concept
	// acts on (file, folder, person…), "" when nothing
	// states or implies one.
	ObjectType string
	// Relations is the concept's default relation set:
	// stated relations first, then the ones its shape
	// implies (cmdmatch.DeriveRelations for a
	// declaration; the verb factor for a label).
	Relations []cmdmatch.Relation
}

// objectWordType maps the object words of factorized
// label names and descriptions to value types. Small
// and explicit, like the cousin families: a word not
// here implies no type.
var objectWordType = map[string]string{
	"file": "file", "files": "file",
	"photo": "file", "photos": "file", "picture": "file", "pictures": "file",
	"log": "file", "logs": "file",
	"folder": "folder", "folders": "folder", "directory": "folder",
}

// verbRelations maps a label's action prefix to the
// relations a concept with that verb stands in by
// default. The slot is empty: a label's relations are
// about its object as a whole, not about a seat.
var verbRelations = map[string][]string{
	"move":   {"moves_from", "moves_to"},
	"copy":   {"moves_to"},
	"delete": {"targets"},
	"remove": {"targets"},
	"create": {"targets"},
	"make":   {"targets"},
	"cancel": {"cancels"},
	"stop":   {"cancels"},
}

// ConceptProperties returns the property set of one
// concept — a declaration's name or a classifier
// label's — and whether the interpreter knows the
// concept at all.
func (in *Interpreter) ConceptProperties(name string) (ConceptProperties, bool) {
	if in.reg != nil {
		if decl, ok := in.reg.Find(name); ok {
			return declarationProperties(decl), true
		}
	}
	if in.factors != nil {
		if f, ok := in.factors.ByLabel[name]; ok {
			return labelProperties(name, f), true
		}
	}
	return ConceptProperties{}, false
}

// DeclarationProperties is declarationProperties for
// hosts that hold a declaration but no interpreter —
// teach-time inference in examples/customcmd uses it to
// write the inferred object type into the registry next
// to the inferred relations.
func DeclarationProperties(decl cmdmatch.Declaration) ConceptProperties {
	return declarationProperties(decl)
}

// declarationProperties assembles a declaration's
// properties: the stated object type wins; otherwise
// the ActsOn hint types it when the hint IS a type (or
// a known plural of one); otherwise a single distinct
// slot family type stands for the whole.
func declarationProperties(decl cmdmatch.Declaration) ConceptProperties {
	p := ConceptProperties{Relations: cmdmatch.DeriveRelations(decl)}
	if decl.ObjectType != "" {
		p.ObjectType = decl.ObjectType
		return p
	}
	if t := objectWordType[strings.ToLower(decl.ActsOn)]; t != "" {
		p.ObjectType = t
		return p
	}
	// The description's own object words ("copy files
	// from one folder to another" acts on files — the
	// first typed object word it names).
	for _, w := range strings.Fields(strings.ToLower(decl.Description)) {
		if t := objectWordType[strings.Trim(w, ".,;:!?\"'")]; t != "" {
			p.ObjectType = t
			return p
		}
	}
	seen := map[string]bool{}
	for _, slot := range decl.Slots {
		if t := ValueType(decl, slot, "", nil); t != "" {
			seen[t] = true
		}
	}
	if len(seen) == 1 {
		for t := range seen {
			p.ObjectType = t
		}
	}
	return p
}

// labelProperties assembles a label's properties from
// its factorized name: the object tokens type it (the
// LAST object token is the head noun — buffered_channel
// is a channel), and the action prefix names its
// default relations.
func labelProperties(label string, f LabelFactor) ConceptProperties {
	var p ConceptProperties
	for _, obj := range f.Objects {
		if t := objectWordType[obj]; t != "" {
			p.ObjectType = t
		}
	}
	if i := strings.Index(label, "_"); i > 0 {
		for _, rel := range verbRelations[label[:i]] {
			p.Relations = append(p.Relations, cmdmatch.Relation{Name: rel})
		}
	}
	return p
}
