package interpret_test

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/interpret"
)

// Piece 1 probe set: relation range validation. The
// probe registry states moves_from/moves_to on a
// from/to command. The binder seats whatever the line
// says; ONLY the relations layer can call a seated
// value a violation — without stated relations the same
// lines produce no verdict at all (the control).

func relationRegistry(t *testing.T, stated bool) *cmdmatch.Registry {
	t.Helper()
	d := cmdmatch.Declaration{
		Name:        "copy-photos",
		Command:     "./copy-photos.sh",
		Description: "copies photos from one folder to another",
		Slots:       []string{"from", "to"},
		Examples:    []string{"copy the photos", "back up my pictures"},
	}
	if stated {
		d.Relations = []cmdmatch.Relation{
			{Name: "moves_from", Slot: "from"},
			{Name: "moves_to", Slot: "to"},
		}
	}
	reg := &cmdmatch.Registry{Scripts: []cmdmatch.Declaration{d}}
	if err := reg.Validate(); err != nil {
		t.Fatal(err)
	}
	return reg
}

func TestRelationProbeSet(t *testing.T) {
	probes := []struct {
		line         string
		wantViols    int
		wantHopWords string // substring expected in a hop trace line; "" = no hop expected for from
	}{
		// Both values seat as places (the slot family's
		// type): both relations fit, two hops, no
		// violations.
		{"copy the photos from jim to jake", 0, "moves_from jim"},
		// cat.txt reads as a file by shape; moves_from
		// expects a place. Exactly one violation — and
		// the value STAYS seated (reported, not dropped).
		{"copy the photos from cat.txt to jake", 1, ""},
		// A file in the to seat: moves_to violated.
		{"copy the photos from jim to notes.md", 1, "moves_from jim"},
	}
	in := interpret.New(relationRegistry(t, true), nil)
	right, wrong := 0, 0
	for _, p := range probes {
		r := in.Interpret(p.line)
		ok := r.Command == "copy-photos" && len(r.Violations) == p.wantViols
		if p.wantHopWords != "" {
			hop := false
			for _, tr := range r.Trace {
				if strings.HasPrefix(tr, "relation:") && strings.Contains(tr, p.wantHopWords) {
					hop = true
				}
			}
			ok = ok && hop
		}
		if p.wantViols > 0 && r.Slots["from"] == "" && r.Slots["to"] == "" {
			ok = false // a violation must never unseat everything
		}
		if ok {
			right++
		} else {
			wrong++
			t.Errorf("probe %q: violations=%v slots=%v trace=%v", p.line, r.Violations, r.Slots, r.Trace)
		}
	}
	// The violation's own words: the value stays seated.
	r := in.Interpret("copy the photos from cat.txt to jake")
	if r.Slots["from"] != "cat.txt" {
		t.Errorf("violating value was unseated: slots = %v — violations are reported, never dropped", r.Slots)
	}
	if len(r.Violations) == 1 && !strings.Contains(r.Violations[0], "stays seated") {
		t.Errorf("violation wording = %q, want it to state the value stays seated", r.Violations[0])
	}
	t.Logf("piece 1 probe set: %d right, %d wrong, 0 abstain (of %d)", right, wrong, len(probes))
	if wrong > 0 {
		t.Fatalf("probe set: %d wrong", wrong)
	}
}

// The control: the same lines against a registry that
// states no relations produce no hops and no violations
// — the verdicts above come from the relations layer,
// not from the binder or the matcher.
func TestRelationControlNoStatedRelations(t *testing.T) {
	in := interpret.New(relationRegistry(t, false), nil)
	for _, line := range []string{
		"copy the photos from jim to jake",
		"copy the photos from cat.txt to jake",
	} {
		r := in.Interpret(line)
		if len(r.Violations) != 0 {
			t.Errorf("line %q: violations %v without stated relations", line, r.Violations)
		}
		for _, tr := range r.Trace {
			if strings.HasPrefix(tr, "relation") {
				t.Errorf("line %q: relation trace %q without stated relations", line, tr)
			}
		}
	}
}
