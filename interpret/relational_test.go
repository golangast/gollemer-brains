package interpret_test

import (
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/interpret"
)

// Piece 5 probe set: relation-aware inference. Two
// mechanisms, each the only path to its verdict:
//
//   - The relational composition adopts a command from
//     relation cues + object words on a line the matcher
//     abstains on — no stated relations (the control),
//     no adoption.
//   - Relation disjointness withholds a matched move
//     read when the same line cancels the very value it
//     moves — without the relation layer the line reads
//     as a plain move.

func relationalRegistry(t *testing.T, stated bool) *cmdmatch.Registry {
	t.Helper()
	archive := cmdmatch.Declaration{
		Name:        "archive-photos",
		Command:     "./archive.sh",
		Description: "puts the photos into the archive folder",
		Slots:       []string{"to"},
		Examples:    []string{"archive the photos"},
	}
	cancel := cmdmatch.Declaration{
		Name:        "cancel-archive",
		Command:     "./cancel-archive.sh",
		Description: "cancels the archive job",
		Examples:    []string{"cancel the archive job"},
	}
	if stated {
		archive.Relations = []cmdmatch.Relation{{Name: "moves_to", Slot: "to"}}
		cancel.Relations = []cmdmatch.Relation{{Name: "cancels"}}
	}
	reg := &cmdmatch.Registry{Scripts: []cmdmatch.Declaration{archive, cancel}}
	if err := reg.Validate(); err != nil {
		t.Fatal(err)
	}
	return reg
}

// moverRegistry is the disjointness probe's registry:
// the mover alone, so the matcher has no rival to claim
// a torn line first (matcher-first precedence is a
// standing, separate wart — this probe set is about the
// relation check, and isolates it).
func moverRegistry(t *testing.T) *cmdmatch.Registry {
	t.Helper()
	mover := cmdmatch.Declaration{
		Name:        "move-photos",
		Command:     "./move.sh",
		Description: "moves photos from one folder to another",
		Slots:       []string{"from", "to"},
		Examples:    []string{"move the photos"},
		Relations: []cmdmatch.Relation{
			{Name: "moves_from", Slot: "from"},
			{Name: "moves_to", Slot: "to"},
		},
	}
	reg := &cmdmatch.Registry{Scripts: []cmdmatch.Declaration{mover}}
	if err := reg.Validate(); err != nil {
		t.Fatal(err)
	}
	return reg
}

func TestRelationalCompositionProbeSet(t *testing.T) {
	in := interpret.New(relationalRegistry(t, true), nil)
	right, wrong, abstain := 0, 0, 0

	// The line is nearly all relation + object words:
	// one matched content word (folder) is below the
	// matcher's two-word bar (the premise, asserted),
	// but its cue and object compose exactly one
	// stated-relations declaration.
	const line = "into the folder"
	if _, _, ok := cmdmatch.NewMatcher(relationalRegistry(t, true)).Match(line); ok {
		t.Fatalf("premise broken: the matcher itself reads %q — the probe would prove nothing", line)
	}
	r := in.Interpret(line)
	switch {
	case r.Command == "archive-photos" && r.Relational:
		right++
	case r.Command == "":
		abstain++
		t.Errorf("probe %q abstained, want archive-photos via the relational composition (trace %v)", line, r.Trace)
	default:
		wrong++
		t.Errorf("probe %q read %q (relational=%v), want archive-photos", line, r.Command, r.Relational)
	}

	// A line whose cues name TWO relations no single
	// declaration covers composes nothing.
	if _, _, ok := in.ComposeRelational("cancel the photos into archive"); ok {
		wrong++
		t.Errorf("two-relation line composed a read; want abstention — no declaration covers cancels+moves_to")
	} else {
		right++
	}

	// Control: without stated relations the probe line
	// adopts nothing.
	ctrl := interpret.New(relationalRegistry(t, false), nil)
	if rc := ctrl.Interpret(line); rc.Command != "" {
		wrong++
		t.Errorf("control read %q without stated relations — the composition is not the path", rc.Command)
	} else {
		right++
	}

	t.Logf("piece 5 composition probes: %d right, %d wrong, %d abstain (of %d)", right, wrong, abstain, right+wrong+abstain)
	if wrong > 0 || abstain > 0 {
		t.Fatalf("probe set: %d wrong, %d abstain", wrong, abstain)
	}
}

func TestRelationDisjointnessProbeSet(t *testing.T) {
	in := interpret.New(moverRegistry(t), nil)
	probes := []struct {
		line        string
		wantCommand string
		wantWithheld bool
	}{
		// A plain move: the read stands.
		{"move the photos from jim to jake", "move-photos", false},
		// The same move, with the moved value cancelled
		// in the same line: torn — withheld, not blended.
		{"move the photos from jim to jake and cancel jim", "", true},
		// A cancel aimed elsewhere does not tear the move.
		{"move the photos from jim to jake and cancel the job", "move-photos", false},
	}
	right, wrong := 0, 0
	for _, p := range probes {
		r := in.Interpret(p.line)
		ok := r.Command == p.wantCommand && (r.WithheldCommand == "move-photos") == p.wantWithheld
		if ok {
			right++
		} else {
			wrong++
			t.Errorf("probe %q: command=%q withheld=%q, want command=%q withheld=%v (trace %v)",
				p.line, r.Command, r.WithheldCommand, p.wantCommand, p.wantWithheld, r.Trace)
		}
	}
	t.Logf("piece 5 disjointness probes: %d right, %d wrong, 0 abstain (of %d)", right, wrong, len(probes))
	if wrong > 0 {
		t.Fatalf("probe set: %d wrong", wrong)
	}
}
