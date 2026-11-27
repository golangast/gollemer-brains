package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Tests for the concept-graph build: Needs relations and
// place-family inheritance (Piece 1), near-tie
// disambiguation (Piece 2), corrections that stick
// (Piece 3), and named routines (Piece 4). The measured
// sessions behind them are declared in the build's
// scratch NOTES.

// graphDemo is planDemo over a scratch registry built
// from the given declarations (planDemo's extractor
// answers "jim jake" for lines naming jim).
func graphDemo(t *testing.T, scripts []cmdmatch.Declaration, answers []string) (*demo, *bytes.Buffer) {
	t.Helper()
	reg := &cmdmatch.Registry{Scripts: scripts}
	dir := t.TempDir()
	path := filepath.Join(dir, "commands.json")
	if err := reg.Save(path); err != nil {
		t.Fatal(err)
	}
	reg, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	return planDemo(t, path, reg, answers)
}

// shippedPlusCopy loads the shipped registry into a temp
// file, adds the taught-style copy entry the tie case
// needs (exactly what the declaration door writes for
// John's sentence), and returns a demo over it.
func shippedPlusCopy(t *testing.T, answers []string) (*demo, *bytes.Buffer, string) {
	t.Helper()
	path, reg := tempShippedRegistry(t)
	err := reg.Add(cmdmatch.Declaration{
		Name: "copy", Command: "./copy.sh",
		Description: "copy files from one folder to another",
		Slots:       []string{"from", "to"},
		Examples:    []string{"copy files from one folder to another"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := reg.Save(path); err != nil {
		t.Fatal(err)
	}
	reg, err = cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	d, out := planDemo(t, path, reg, answers)
	return d, out, path
}

// --- Piece 1a: Needs notes ---

// TestNeedsNoteAbsent: a plan containing migrate but no
// backup earns the absence note (shipped registry, where
// migrate-database declares Needs [backup-photos]).
func TestNeedsNoteAbsent(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := planDemo(t, path, reg, nil)
	d.handle("migrate the database from jim to jake and compress the logs")
	if !strings.Contains(out.String(), "note: migrate-database usually needs backup-photos first — it is not in this plan.") {
		t.Errorf("missing-need note not printed:\n%s", out.String())
	}
}

// TestNeedsNoteLater: the need is in the plan but sits
// after the step that needs it (same stage, typed order
// survives the sort) — the later note fires.
func TestNeedsNoteLater(t *testing.T) {
	migrate := cmdmatch.Declaration{
		Name: "migrate-database", Command: "./scripts/migrate-db.sh",
		Description: "migrates a database from one folder to another",
		Slots:       []string{"from", "to"}, Stage: "execute",
		Examples: []string{"migrate the database from jim to jake"},
		Needs:    []string{"backup-photos"},
	}
	backup := cmdmatch.Declaration{
		Name: "backup-photos", Command: "./scripts/backup-photos.sh",
		Description: "backs up my photo library to the external drive",
		Stage:       "execute", // same rank as migrate: typed order stands
		Examples:    []string{"back up my photos"},
	}
	d, out := graphDemo(t, []cmdmatch.Declaration{migrate, backup}, nil)
	d.handle("migrate the database from jim to jake and back up my photos")
	if !strings.Contains(out.String(), "note: migrate-database usually needs backup-photos first — it comes later in this plan.") {
		t.Errorf("later-need note not printed:\n%s", out.String())
	}
}

// TestNeedsSatisfiedNoNote: backup runs before migrate
// (its prepare stage sorts it first) — no note.
func TestNeedsSatisfiedNoNote(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := planDemo(t, path, reg, nil)
	d.handle("back up my photos and migrate the database from jim to jake")
	if !strings.Contains(out.String(), "plan: 2 steps") {
		t.Fatalf("expected a plan:\n%s", out.String())
	}
	if strings.Contains(out.String(), "usually needs") {
		t.Errorf("satisfied need earned a note:\n%s", out.String())
	}
}

// TestNeedsCoveredByAfter: an After edge encoding the
// same relation owns it — the After absence note fires,
// the Needs note does not.
func TestNeedsCoveredByAfter(t *testing.T) {
	migrate := cmdmatch.Declaration{
		Name: "migrate-database", Command: "./scripts/migrate-db.sh",
		Description: "migrates a database from one folder to another",
		Slots:       []string{"from", "to"}, Stage: "execute",
		Examples: []string{"migrate the database from jim to jake"},
		Needs:    []string{"backup-photos"},
		After:    []string{"backup-photos"},
	}
	compress := cmdmatch.Declaration{
		Name: "compress-logs", Command: "./scripts/compress-logs.sh",
		Description: "compresses old log files into a single archive",
		Stage:       "cleanup",
		Examples:    []string{"compress the logs"},
	}
	d, out := graphDemo(t, []cmdmatch.Declaration{migrate, compress}, nil)
	d.handle("migrate the database from jim to jake and compress the logs")
	if !strings.Contains(out.String(), "is declared after backup-photos") {
		t.Errorf("after note not printed:\n%s", out.String())
	}
	if strings.Contains(out.String(), "usually needs") {
		t.Errorf("need already encoded by After earned a second note:\n%s", out.String())
	}
}

// --- Piece 1b: place-family inheritance ---

var renameDirScript = cmdmatch.Declaration{
	Name: "rename-photos", Command: "./scripts/rename-photos.sh",
	Description: "renames the photos in a directory from camera names to dates",
	Slots:       []string{"directory"}, Stage: "execute",
	Examples: []string{"rename the photos"},
}

// TestPlaceFamilyInheritance: step 2's slot is
// "directory", a place name no earlier step uses; step 1
// bound "folder" to trips. The place family carries
// trips across the names, marked "(from step 1)", and
// the extractor's ungrounded favourite for step 2
// ("school") is displaced.
func TestPlaceFamilyInheritance(t *testing.T) {
	d, out := stubDemo(t, []cmdmatch.Declaration{sortScript, renameDirScript}, nil)
	d.handle("sort the files in trips and rename the photos")
	got := out.String()
	if !strings.Contains(got, "directory=trips (from step 1)") {
		t.Errorf("place-family inheritance missing:\n%s", got)
	}
	if strings.Contains(got, "school") {
		t.Errorf("ungrounded favourite survived the family inheritance:\n%s", got)
	}
}

// TestPlaceFamilyControlNonPlace: a non-place slot
// (env) with no value anywhere does NOT inherit the
// place value step 1 bound — the family is places only.
func TestPlaceFamilyControlNonPlace(t *testing.T) {
	seed := cmdmatch.Declaration{
		Name: "seed-test-data", Command: "./scripts/seed-test-data.sh",
		Description: "fills a test database with fake seed data",
		Slots:       []string{"env"}, Stage: "prepare",
		Examples: []string{"seed the test data"},
	}
	d, out := stubDemo(t, []cmdmatch.Declaration{sortScript, seed}, nil)
	d.handle("sort the files in trips and seed the test data")
	got := out.String()
	if !strings.Contains(got, "env: (not heard)") {
		t.Errorf("non-place slot should stay unheard:\n%s", got)
	}
	if strings.Contains(got, "env=trips") {
		t.Errorf("non-place slot cross-inherited a place value:\n%s", got)
	}
}

// --- Piece 2: near-tie disambiguation ---

// TestTieQuestionResolves: the measured tie case. On the
// shipped registry plus the taught copy entry, "copy
// files from jim to jake" margin-abstains between
// migrate-database (0.631) and copy (0.553); the question
// is offered in ranked order, and choosing a candidate —
// by number or by name — presents it with the slots
// extracted from the original line.
func TestTieQuestionResolves(t *testing.T) {
	const question = "did you mean migrate-database or copy? (1/2, or enter for neither)"

	d, out, _ := shippedPlusCopy(t, []string{"1"})
	d.handle("copy files from jim to jake")
	got := out.String()
	if !strings.Contains(got, question) {
		t.Fatalf("tie question not offered:\n%s", got)
	}
	if !strings.Contains(got, "understood: migrate-database") ||
		!strings.Contains(got, "would run: ./scripts/migrate-db.sh jim jake") {
		t.Errorf("answer 1 did not present migrate-database with jim/jake:\n%s", got)
	}

	d, out, _ = shippedPlusCopy(t, []string{"copy"})
	d.handle("copy files from jim to jake")
	got = out.String()
	if !strings.Contains(got, "understood: copy") ||
		!strings.Contains(got, "would run: ./copy.sh jim jake") {
		t.Errorf("answering by name did not present copy with jim/jake:\n%s", got)
	}

	// Declining falls through to the old abstain flow:
	// the teach offer arrives, is skipped, nothing is
	// presented and nothing is learned.
	d, out, path := shippedPlusCopy(t, []string{"", ""})
	d.handle("copy files from jim to jake")
	got = out.String()
	if !strings.Contains(got, question) {
		t.Fatalf("tie question not offered on the decline path:\n%s", got)
	}
	if strings.Contains(got, "understood:") {
		t.Errorf("declining the tie still presented a script:\n%s", got)
	}
	reg, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(reg.Scripts) != 12 {
		t.Errorf("declining the tie changed the registry: %d scripts", len(reg.Scripts))
	}
}

// TestTieNotAskedForOneWordTie: "create a folder" ties
// half the registry on the single word "folder" — an
// abstention on the two-words bar, not the margin — so
// no question is asked and the teach loop closes as
// before.
func TestTieNotAskedForOneWordTie(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := queueDemo(t, path, reg, []string{"mkdir {folder}"})
	d.handle("create a folder")
	got := out.String()
	if strings.Contains(got, "did you mean") {
		t.Errorf("one-word tie earned a question:\n%s", got)
	}
	if !strings.Contains(got, "learned: when you say \"create a folder\"") {
		t.Errorf("teach loop did not close:\n%s", got)
	}
}

// --- Piece 3: corrections ---

// TestCorrectionSticks: the full loop on the shipped
// registry. "clean the photo library" presents
// backup-photos; the user meant clean-downloads (#3).
// The correction appends the line to clean-downloads'
// examples, writes corrections.jsonl, re-presents
// clean-downloads — and a fresh process over the same
// registry matches the line correctly with no correction.
func TestCorrectionSticks(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := planDemo(t, path, reg, []string{"3"})
	d.handle("clean the photo library")
	if !strings.Contains(out.String(), "understood: backup-photos") {
		t.Fatalf("setup: line did not present backup-photos:\n%s", out.String())
	}
	out.Reset()
	d.handle("no")
	got := out.String()
	if !strings.Contains(got, "which script did you mean?") ||
		!strings.Contains(got, "3. clean-downloads") {
		t.Fatalf("correction did not ask with the numbered list:\n%s", got)
	}
	if !strings.Contains(got, "understood: clean-downloads") {
		t.Errorf("correction did not re-present clean-downloads:\n%s", got)
	}

	back, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	decl, ok := back.Find("clean-downloads")
	if !ok {
		t.Fatal("clean-downloads vanished")
	}
	found := false
	for _, ex := range decl.Examples {
		if ex == "clean the photo library" {
			found = true
		}
	}
	if !found {
		t.Errorf("corrected line not in clean-downloads' examples: %v", decl.Examples)
	}

	raw, err := os.ReadFile(filepath.Join(filepath.Dir(path), "corrections.jsonl"))
	if err != nil {
		t.Fatalf("corrections.jsonl not written: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 1 {
		t.Fatalf("corrections.jsonl has %d lines, want 1: %q", len(lines), raw)
	}
	var rec map[string]string
	if err := json.Unmarshal([]byte(lines[0]), &rec); err != nil {
		t.Fatalf("correction record is not JSON: %v", err)
	}
	if rec["input"] != "clean the photo library" || rec["chose"] != "clean-downloads" || rec["rejected"] != "backup-photos" {
		t.Errorf("correction record wrong: %v", rec)
	}

	// Restart persistence: a fresh demo over the saved
	// registry matches the line to clean-downloads.
	d2, out2 := planDemo(t, path, back, nil)
	d2.handle("clean the photo library")
	if !strings.Contains(out2.String(), "understood: clean-downloads") {
		t.Errorf("after restart the line does not match the chosen script:\n%s", out2.String())
	}
}

// TestCorrectionAloneIsNothing: a correction word with
// no presentation before it is an ordinary line — on an
// unwired demo, silence.
func TestCorrectionAloneIsNothing(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := queueDemo(t, path, reg, nil)
	for _, line := range []string{"no", "wrong", "not that one", "no, not that"} {
		out.Reset()
		d.handle(line)
		if out.String() != "" {
			t.Errorf("bare correction %q produced output:\n%s", line, out.String())
		}
	}
}

// TestIsCorrection pins the trigger set, small and exact.
func TestIsCorrection(t *testing.T) {
	for _, line := range []string{"no", "No", "no, not that one", "no i meant the backup", "wrong", "wrong one", "not that one"} {
		if !isCorrection(line) {
			t.Errorf("%q should be a correction", line)
		}
	}
	for _, line := range []string{"nope", "north", "clean the photo library", "no way to know", "notebook"} {
		if isCorrection(line) {
			t.Errorf("%q should not be a correction", line)
		}
	}
}

// --- Piece 4: routines ---

// TestRoutineDeclarationRefusal: a routine segment no
// command knows refuses the whole declaration, naming
// the segment, and saves nothing.
func TestRoutineDeclarationRefusal(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := queueDemo(t, path, reg, nil)
	d.handle("i have a routine named broken and it will back up my photos and polish the trophies in the cabinet")
	got := out.String()
	if !strings.Contains(got, "i don't know 'polish the trophies in the cabinet' yet — teach me that command first") {
		t.Errorf("refusal line wrong:\n%s", got)
	}
	back, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := back.Find("broken"); ok {
		t.Error("refused routine was saved")
	}
	if len(back.Scripts) != len(reg.Scripts) {
		t.Errorf("registry changed on refusal: %d scripts, want %d", len(back.Scripts), len(reg.Scripts))
	}
}

// TestRoutineDeclarationAndInvoke: a routine declared in
// one sentence is stored as steps with no command, and a
// later want that matches it expands into its plan.
func TestRoutineDeclarationAndInvoke(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := queueDemo(t, path, reg, nil)
	d.handle("i have a routine named nightly and it will back up my photos and migrate the database and compress the logs")
	got := out.String()
	if !strings.Contains(got, "learned: nightly") ||
		!strings.Contains(got, "routine: 3 steps (backup-photos, migrate-database, compress-logs)") {
		t.Fatalf("routine declaration not learned as declared:\n%s", got)
	}
	back, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	decl, ok := back.Find("nightly")
	if !ok {
		t.Fatal("nightly not in the registry")
	}
	if decl.Command != "" || len(decl.Steps) != 3 {
		t.Errorf("nightly stored wrong: command %q, steps %v", decl.Command, decl.Steps)
	}

	d2, out2 := planDemo(t, path, back, nil)
	d2.handle("nightly photos database logs")
	got = out2.String()
	if !strings.Contains(got, "routine: nightly") ||
		!strings.Contains(got, "plan: 3 steps (3 ready, 0 gaps)") ||
		!strings.Contains(got, "nothing has run yet.") {
		t.Errorf("routine invocation did not expand into its plan:\n%s", got)
	}
}

// TestShippedRoutineInvoke: the shipped full-backup
// routine, invoked by its example phrasing, renders its
// three steps as a plan under its own header.
func TestShippedRoutineInvoke(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := planDemo(t, path, reg, nil)
	d.handle("run the full backup routine")
	got := out.String()
	for _, want := range []string{
		"routine: full-backup",
		"plan: 3 steps (3 ready, 0 gaps)",
		"concept: backup-photos",
		"concept: migrate-database",
		"concept: compress-logs",
		"nothing has run yet.",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("shipped routine output missing %q:\n%s", want, got)
		}
	}
}

// TestDeclaredRoutineDoesNotGapShipped: the live find —
// after a routine whose description is its doing-clause
// is declared, the SHIPPED routine's steps must still
// resolve to their commands. Segment resolution is
// two-phase (commands before routines) precisely so one
// routine's clause cannot tie another routine's steps
// into gaps.
func TestDeclaredRoutineDoesNotGapShipped(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, _ := queueDemo(t, path, reg, nil)
	d.handle("i have a routine named nightly and it will back up my photos and migrate the database and compress the logs")
	back, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	d2, out2 := planDemo(t, path, back, nil)
	d2.handle("run the full backup routine")
	if !strings.Contains(out2.String(), "plan: 3 steps (3 ready, 0 gaps)") {
		t.Errorf("declared routine gapped the shipped routine's steps:\n%s", out2.String())
	}
}

// TestRoutineInPlanExpands: a plan segment that names a
// routine expands in place into the routine's steps.
func TestRoutineInPlanExpands(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := planDemo(t, path, reg, nil)
	d.handle("back up my photos and run the full backup routine")
	if !strings.Contains(out.String(), "plan: 4 steps (4 ready, 0 gaps)") {
		t.Errorf("routine segment did not expand inside the plan:\n%s", out.String())
	}
}

// TestRoutineTooDeep: routines nest one level; a third
// level renders as a gap with the too-deep note, and the
// leaf command is never reached.
func TestRoutineTooDeep(t *testing.T) {
	leaf := cmdmatch.Declaration{
		Name: "backup-photos", Command: "./b.sh",
		Description: "backs up my photo library to the external drive",
		Examples:    []string{"back up my photos"},
	}
	r2 := cmdmatch.Declaration{
		Name: "r2", Description: "the second routine",
		Examples: []string{"run the r2"},
		Steps:    []string{"back up my photos"},
	}
	r1 := cmdmatch.Declaration{
		Name: "r1", Description: "the first routine",
		Examples: []string{"run the r1"},
		Steps:    []string{"run the r2"},
	}
	r0 := cmdmatch.Declaration{
		Name: "r0", Description: "the outer routine",
		Examples: []string{"run the r0"},
		Steps:    []string{"run the r1"},
	}
	d, out := graphDemo(t, []cmdmatch.Declaration{leaf, r2, r1, r0}, nil)
	d.handle("run the r0")
	got := out.String()
	if !strings.Contains(got, "routine: r0") {
		t.Fatalf("outer routine did not open:\n%s", got)
	}
	if !strings.Contains(got, "routine too deep") {
		t.Errorf("depth cap did not fire:\n%s", got)
	}
	if strings.Contains(got, "concept: backup-photos") {
		t.Errorf("expansion ran past the depth cap:\n%s", got)
	}
}
