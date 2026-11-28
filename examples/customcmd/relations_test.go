package main

import (
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Piece 1, teach-time: a declaration taught in one
// sentence lands in the registry with its obvious
// relations NAMED — the from/to the sentence already
// parsed become moves_from/moves_to — so the relations
// layer has stated data to validate and trace against
// from the next line on.
func TestDeclarationFlowInfersRelations(t *testing.T) {
	d, _ := stubDemo(t, nil, nil)
	d.declarationFlow(mustParse(t, johnDeclaration))
	decl, ok := d.reg.Find("copy")
	if !ok {
		t.Fatal("the declaration was not learned")
	}
	want := []cmdmatch.Relation{
		{Name: "moves_from", Slot: "from"},
		{Name: "moves_to", Slot: "to"},
	}
	if len(decl.Relations) != len(want) {
		t.Fatalf("taught relations = %v, want %v", decl.Relations, want)
	}
	for i := range want {
		if decl.Relations[i] != want[i] {
			t.Errorf("relation %d = %v, want %v", i, decl.Relations[i], want[i])
		}
	}
	// The concept property rides along: the sentence
	// speaks of copying FILES, so the entry's object
	// type is inferred as file.
	if decl.ObjectType != "file" {
		t.Errorf("taught object type = %q, want file (inferred from the description)", decl.ObjectType)
	}
	// And the entry still matches the way it always did.
	if _, _, ok := d.matcher.Match("copy files from jim to jake"); !ok {
		t.Errorf("the taught command no longer matches its own shape")
	}
}

func mustParse(t *testing.T, line string) parsedDeclaration {
	t.Helper()
	p, ok := parseDeclaration(line)
	if !ok {
		t.Fatalf("line %q did not parse as a declaration", line)
	}
	return p
}
