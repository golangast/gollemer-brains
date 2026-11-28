package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Piece 1 probe set: session objects in the one prompt.
// A value seated in one presentation is remembered with
// its type; a later line referring back — "the same
// folder", or the bare name — resolves against the
// table; a seat under a conflicting type is reported,
// never forced. Every check below needs the table: the
// extractor stub hears NOTHING, so the span binder and
// the object layer are the only seating paths, and the
// fresh-session controls show the same lines resolving
// nothing without the memory.

func objectProbeDemo(t *testing.T) (*demo, *bytes.Buffer) {
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
			Name:        "rename-photos",
			Command:     "./rename-photos.sh",
			Description: "renames the photos in a folder",
			Slots:       []string{"folder"},
			Examples:    []string{"rename the photos"},
			SlotTypes:   map[string]string{"folder": "folder"},
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
	path := filepath.Join(t.TempDir(), "commands.json")
	if err := reg.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	d := &demo{
		registryPath: path,
		reg:          loaded,
		matcher:      cmdmatch.NewMatcher(loaded),
		extract:      func(line string) (string, error) { return "", nil },
		out:          &out,
	}
	return d, &out
}

func TestSessionObjectProbeSet(t *testing.T) {
	right, wrong := 0, 0
	check := func(name string, ok bool, detail string) {
		if ok {
			right++
		} else {
			wrong++
			t.Errorf("probe %s failed: %s", name, detail)
		}
	}

	d, out := objectProbeDemo(t)

	// 1. First seating: jim and jake enter the table as
	// folders; nothing is remembered yet, nothing marked.
	d.handle("copy the photos from jim to jake")
	got := out.String()
	check("seat", strings.Contains(got, "arguments: from=jim, to=jake") && !strings.Contains(got, "(remembered)"),
		"first line should seat plainly:\n"+got)
	first := got

	// 2. Saying the first line again is byte-identical:
	// its values are named again but already seated, so
	// the layer changes nothing.
	out.Reset()
	d.handle("copy the photos from jim to jake")
	check("repeat-identical", out.String() == first,
		"a repeated line must present identically:\n"+out.String())

	// 3. "the same folder": the binder lifts only the
	// leftover "same" into the to seat — the remembered
	// jake (the most recent folder) is the only way the
	// seat gets a value.
	out.Reset()
	d.handle("copy the photos from backup to the same folder")
	got = out.String()
	check("same-folder", strings.Contains(got, "to=jake (remembered)") &&
		strings.Contains(got, "would run: ./copy-photos.sh backup jake"),
		"the same folder should resolve to jake:\n"+got)

	// 4. The bare name: holidays is seated by an anchored
	// line, then named again in a line with no anchor —
	// the table is the only way it seats.
	out.Reset()
	d.handle("rename the photos in holidays")
	if !strings.Contains(out.String(), "arguments: folder=holidays") {
		check("name-seat", false, "anchored line should seat holidays:\n"+out.String())
	} else {
		check("name-seat", true, "")
	}
	out.Reset()
	d.handle("rename the holidays photos")
	got = out.String()
	check("named-again", strings.Contains(got, "folder=holidays (remembered)"),
		"the bare name should resolve from the table:\n"+got)

	// 5. Conflict in the open: jim is a folder this
	// session; seating it as a file is reported, and the
	// value stays seated.
	out.Reset()
	d.handle("read the file jim")
	got = out.String()
	check("conflict", strings.Contains(got, "object type mismatch") && strings.Contains(got, `"jim"`) &&
		strings.Contains(got, "arguments: file=jim") && strings.Contains(got, "would run: ./read-file.sh jim"),
		"the mismatch should be reported with the value seated:\n"+got)

	// Controls: a fresh session knows nothing — the same
	// referring lines resolve nothing and report nothing.
	fresh, fout := objectProbeDemo(t)
	fresh.handle("copy the photos from backup to the same folder")
	// Without the memory the binder can only lift the
	// seat word itself — "same" is seated as the value.
	// The remembered object is what the table alone
	// supplies.
	check("control-same", strings.Contains(fout.String(), "to=same") && !strings.Contains(fout.String(), "jake"),
		"fresh session must not resolve the same folder:\n"+fout.String())
	fout.Reset()
	fresh.handle("rename the holidays photos")
	check("control-named", strings.Contains(fout.String(), "folder=(not heard)"),
		"fresh session must not seat the bare name:\n"+fout.String())
	fout.Reset()
	fresh.handle("read the file jim")
	check("control-conflict", !strings.Contains(fout.String(), "object type mismatch"),
		"fresh session must not report a mismatch it never learned:\n"+fout.String())

	t.Logf("piece 1 probe set: %d right, %d wrong, 0 abstain (of %d checks)", right, wrong, right+wrong)
	if wrong > 0 {
		t.Fatalf("probe set: %d wrong", wrong)
	}
}
