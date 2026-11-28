package interpret_test

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/interpret"
)

// Piece 2 probe set: session objects. A value seated in
// a typed seat is remembered with its type; later lines
// refer back to it — by name, or as "the same folder" —
// and a later seating under a conflicting type is
// reported, not forced. Every verdict below needs the
// object table: a fresh interpreter (the control) knows
// none of these things.

func objectRegistry(t *testing.T) *cmdmatch.Registry {
	t.Helper()
	reg := &cmdmatch.Registry{Scripts: []cmdmatch.Declaration{
		{
			Name:        "copy-photos",
			Command:     "./copy-photos.sh",
			Description: "copies photos from one folder to another",
			Slots:       []string{"from", "to"},
			Examples:    []string{"copy the photos", "back up my pictures"},
			SlotTypes:   map[string]string{"from": "folder", "to": "folder"},
		},
		{
			Name:        "read-file",
			Command:     "./read-file.sh",
			Description: "reads a file aloud",
			Slots:       []string{"file"},
			Examples:    []string{"read the file", "read a file to me"},
			SlotTypes:   map[string]string{"file": "file"},
		},
	}}
	if err := reg.Validate(); err != nil {
		t.Fatal(err)
	}
	return reg
}

func refFor(r interpret.Reading, value string) (interpret.ObjectRef, bool) {
	for _, ref := range r.References {
		if ref.Value == value {
			return ref, true
		}
	}
	return interpret.ObjectRef{}, false
}

func TestObjectProbeSet(t *testing.T) {
	in := interpret.New(objectRegistry(t), nil)
	right, wrong := 0, 0
	check := func(name string, ok bool, detail string) {
		if ok {
			right++
		} else {
			wrong++
			t.Errorf("probe %s failed: %s", name, detail)
		}
	}

	// 1. First seating: jim and jake enter the table as
	// folders; nothing is a reference yet.
	r1 := in.Interpret("copy the photos from jim to jake")
	check("seat", r1.Command == "copy-photos" && r1.SlotTypes["from"] == "folder" && len(r1.References) == 0,
		"first line should seat typed values with no references")
	if o, ok := in.Objects().Lookup("jim"); !ok || o.Type != "folder" || o.Command != "copy-photos" {
		check("table", false, "jim should be remembered as a folder seated by copy-photos")
	} else {
		check("table", true, "")
	}

	// 2. Naming jake again refers back to the SAME
	// object, with its remembered type.
	r2 := in.Interpret("copy the photos from jake to backup")
	ref, ok := refFor(r2, "jake")
	check("named-again", ok && ref.Type == "folder" && ref.Via == "named again",
		"jake should be referenced as the known folder")

	// 3. "the same folder" resolves to the most recent
	// folder the session introduced — backup, seated by
	// the previous line.
	r3 := in.Interpret("copy the photos to the same folder")
	ref, ok = refFor(r3, "backup")
	check("same-folder", ok && ref.Via == "the same folder",
		"the same folder should resolve to backup, the most recent folder")

	// 4. Seating jim — a known folder — as a file is a
	// reported mismatch, and the value stays seated.
	r4 := in.Interpret("read the file jim")
	mismatch := false
	for _, v := range r4.Violations {
		if strings.Contains(v, "object type mismatch") && strings.Contains(v, "stays seated") {
			mismatch = true
		}
	}
	check("mismatch-reported", mismatch && r4.Slots["file"] == "jim",
		"jim seated as a file should be reported, not forced, and stay seated")
	if o, _ := in.Objects().Lookup("jim"); o.Type != "folder" {
		check("first-type-kept", false, "the table must keep jim's first type")
	} else {
		check("first-type-kept", true, "")
	}

	// Control: a fresh interpreter knows nothing — the
	// same lines produce no references and no mismatch.
	fresh := interpret.New(objectRegistry(t), nil)
	rc := fresh.Interpret("read the file jim")
	check("control", len(rc.References) == 0 && len(rc.Violations) == 0,
		"a fresh session must not report what it never learned")

	t.Logf("piece 2 probe set: %d right, %d wrong, 0 abstain (of %d checks)", right, wrong, right+wrong)
	if wrong > 0 {
		t.Fatalf("probe set: %d wrong", wrong)
	}
}
