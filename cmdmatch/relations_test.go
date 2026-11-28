package cmdmatch

import "testing"

// The relations layer's vocabulary and derivation,
// pinned: inverse pairs declared once, the obvious
// relations named from a declaration's own shape, and
// the registry refusing relation data it cannot honour.

func TestRelationInversePairs(t *testing.T) {
	pairs := map[string]string{
		"moves_from": "moves_to",
		"moves_to":   "moves_from",
		"contains":   "part_of",
		"part_of":    "contains",
		"cancels":    "", // a polarity, not a two-sided edge
		"targets":    "",
		"nope":       "",
	}
	for name, want := range pairs {
		if got := InverseRelation(name); got != want {
			t.Errorf("InverseRelation(%q) = %q, want %q", name, got, want)
		}
	}
}

func TestDeriveRelationsNamesTheSlots(t *testing.T) {
	d := Declaration{
		Name: "migrate-database", Command: "./m.sh",
		Description: "migrates a database from one folder to another",
		Slots:       []string{"from", "to"},
		Needs:       []string{"backup-photos"},
	}
	got := DeriveRelations(d)
	want := []Relation{
		{Name: "moves_from", Slot: "from"},
		{Name: "moves_to", Slot: "to"},
		{Name: "requires", Target: "backup-photos"},
	}
	if len(got) != len(want) {
		t.Fatalf("DeriveRelations = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("relation %d = %v, want %v", i, got[i], want[i])
		}
	}
	// A stated relation wins its key and is not doubled
	// by the derivation.
	d.Relations = []Relation{{Name: "moves_from", Slot: "from"}}
	if got := DeriveRelations(d); len(got) != 3 {
		t.Errorf("with moves_from stated, DeriveRelations = %v, want 3 unique", got)
	}
}

func TestHasRelationReadsBothDirections(t *testing.T) {
	d := Declaration{
		Name: "mover", Command: "./m.sh", Description: "moves things",
		Slots:     []string{"from"},
		Relations: []Relation{{Name: "moves_from", Slot: "from"}},
	}
	if !HasRelation(d, "moves_from") || !HasRelation(d, "moves_to") {
		t.Errorf("a stated moves_from must answer to both names")
	}
	if HasRelation(d, "cancels") {
		t.Errorf("an unstated, underivable relation must not appear")
	}
}

func TestTypeSatisfiesLattice(t *testing.T) {
	cases := []struct {
		got, want string
		ok        bool
	}{
		{"folder", "place", true},  // a folder is a place
		{"folder", "folder", true}, // equality
		{"file", "object", true},   // everything concrete is an object
		{"person", "object", true},
		{"file", "place", false}, // the violation shape
		{"folder", "file", false},
		{"", "place", false}, // unknown proves nothing
		{"file", "", false},
	}
	for _, c := range cases {
		if ok := TypeSatisfies(c.got, c.want); ok != c.ok {
			t.Errorf("TypeSatisfies(%q, %q) = %v, want %v", c.got, c.want, ok, c.ok)
		}
	}
}

func TestRegistryValidatesRelations(t *testing.T) {
	base := Declaration{
		Name: "mover", Command: "./m.sh", Description: "moves things",
		Slots: []string{"from", "to"},
	}
	bad := []struct {
		name string
		mut  func(*Declaration)
	}{
		{"unknown relation", func(d *Declaration) { d.Relations = []Relation{{Name: "teleports", Slot: "from"}} }},
		{"relation on undeclared slot", func(d *Declaration) { d.Relations = []Relation{{Name: "moves_from", Slot: "where"}} }},
		{"type for undeclared slot", func(d *Declaration) { d.SlotTypes = map[string]string{"where": "place"} }},
		{"unknown value type", func(d *Declaration) { d.SlotTypes = map[string]string{"from": "galaxy"} }},
	}
	for _, b := range bad {
		d := base
		b.mut(&d)
		reg := &Registry{Scripts: []Declaration{d}}
		if err := reg.Validate(); err == nil {
			t.Errorf("%s: Validate passed, want an error", b.name)
		}
	}
	good := base
	good.Relations = []Relation{{Name: "moves_from", Slot: "from"}, {Name: "moves_to", Slot: "to"}}
	good.SlotTypes = map[string]string{"from": "folder", "to": "folder"}
	if err := (&Registry{Scripts: []Declaration{good}}).Validate(); err != nil {
		t.Errorf("a well-formed relational declaration must validate: %v", err)
	}
}
