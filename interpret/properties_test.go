package interpret_test

import (
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/intent"
	"github.com/golangast/gollemer-brains/interpret"
)

// Piece 4 probe set: concept properties. A concept's
// property set — its object type and default relations —
// is readable as data: stated on a declaration, derived
// from its shape when unstated, and derived for a
// classifier label from its factorized name. No earlier
// mechanism exposes any of this; the assertions are the
// property values themselves.

func TestConceptPropertiesProbes(t *testing.T) {
	reg := &cmdmatch.Registry{Scripts: []cmdmatch.Declaration{
		{
			Name: "typed-mover", Command: "./m.sh",
			Description: "moves things between places",
			Slots:       []string{"from", "to"},
			ObjectType:  "folder", // stated: wins over every derivation
		},
		{
			Name: "photo-copy", Command: "./c.sh",
			Description: "copies files from one folder to another",
			Slots:       []string{"from", "to"},
		},
	}}
	if err := reg.Validate(); err != nil {
		t.Fatal(err)
	}
	var pairs []data.Pair
	add := func(label, kind string, lines ...string) {
		for _, line := range lines {
			pairs = append(pairs, data.Pair{Input: line, Intent: label, Kind: kind})
		}
	}
	add("delete_file", "remove",
		"delete the file", "remove the file please", "erase this file",
		"delete a file for me", "get rid of the file", "delete that file")
	add("play_music", "play",
		"play some music", "start the music", "music please",
		"play a tune", "put on music", "play songs")
	clf, err := intent.Train(pairs, intent.DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	in := interpret.New(reg, clf)

	right, wrong := 0, 0
	check := func(name string, ok bool, detail string) {
		if ok {
			right++
		} else {
			wrong++
			t.Errorf("probe %s: %s", name, detail)
		}
	}

	// Stated property, read back verbatim.
	p, ok := in.ConceptProperties("typed-mover")
	check("stated", ok && p.ObjectType == "folder", "a stated object_type must be the property")

	// Derived from the description's object words, with
	// the slot relations named.
	p, ok = in.ConceptProperties("photo-copy")
	rels := map[string]bool{}
	for _, r := range p.Relations {
		rels[r.Name] = true
	}
	check("derived", ok && p.ObjectType == "file" && rels["moves_from"] && rels["moves_to"],
		"photo-copy should derive object type file and the move relations from its shape")

	// A label's properties come from its factorized
	// name: delete_file acts on a file, via targets.
	p, ok = in.ConceptProperties("delete_file")
	rels = map[string]bool{}
	for _, r := range p.Relations {
		rels[r.Name] = true
	}
	check("label", ok && p.ObjectType == "file" && rels["targets"],
		"delete_file should carry object type file and a targets relation")

	// An unknown concept has no properties.
	_, ok = in.ConceptProperties("nope")
	check("unknown", !ok, "an unknown concept must report no properties")

	t.Logf("piece 4 probe set: %d right, %d wrong, 0 abstain (of %d)", right, wrong, right+wrong)
	if wrong > 0 {
		t.Fatalf("probe set: %d wrong", wrong)
	}
}
