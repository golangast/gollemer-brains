package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// queueDemo builds a demo over a registry at path whose
// nextLine serves the given answers in order, and whose
// extractor answers "receipts" for lines naming receipts
// and nothing otherwise.
func queueDemo(t *testing.T, path string, reg *cmdmatch.Registry, answers []string) (*demo, *bytes.Buffer) {
	t.Helper()
	var out bytes.Buffer
	d := &demo{
		registryPath: path,
		reg:          reg,
		matcher:      cmdmatch.NewMatcher(reg),
		extract: func(line string) (string, error) {
			if strings.Contains(line, "receipts") {
				return "receipts", nil
			}
			return "", nil
		},
		nextLine: func(prompt string) (string, bool) {
			if len(answers) == 0 {
				return "", false
			}
			a := answers[0]
			answers = answers[1:]
			return a, true
		},
		out: &out,
	}
	return d, &out
}

func tempShippedRegistry(t *testing.T) (string, *cmdmatch.Registry) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "commands.json")
	reg, err := cmdmatch.LoadRegistry(filepath.Join("..", "commands.json"))
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	if err := reg.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	reg, err = cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry(temp): %v", err)
	}
	return path, reg
}

// TestTeachOnAbstainClosesLoop is the teach flow end to
// end: a want that matches nothing earns an offer; the
// answer names the command; the entry written is the
// want-line itself as description with the command's
// placeholders as slots, plus the teach-time synthesized
// variants of the want as extra examples (synthesize.go);
// the same breath re-runs the want and it lands; and a
// later phrasing with a value chooses the taught command
// and binds the value.
func TestTeachOnAbstainClosesLoop(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := queueDemo(t, path, reg, []string{"mkdir {folder}"})

	d.handle("create a folder")
	got := out.String()
	for _, want := range []string{
		"learned: when you say \"create a folder\", i'll run mkdir {folder} (saved as mkdir)",
		"understood: mkdir", // the same-breath re-run
	} {
		if !strings.Contains(got, want) {
			t.Errorf("teach output missing %q:\n%s", want, got)
		}
	}

	back, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry(after teach): %v", err)
	}
	decl, ok := back.Find("mkdir")
	if !ok {
		t.Fatalf("taught entry not in the registry file:\n%s", got)
	}
	if decl.Description != "create a folder" || decl.Command != "mkdir {folder}" ||
		len(decl.Slots) != 1 || decl.Slots[0] != "folder" {
		t.Errorf("taught entry wrong: %+v", decl)
	}
	// The want line itself first, then its synthesized
	// variants: create's cousins, then folder's (the slot
	// word still substitutes in a teach want — its anchors
	// are the command's placeholders, not prose anchors;
	// no value is seated in the line, so no value swaps).
	wantExamples := []string{
		"create a folder",
		"build a folder",
		"generate a folder",
		"create a directory",
		"create a dir",
	}
	if strings.Join(decl.Examples, "\n") != strings.Join(wantExamples, "\n") {
		t.Errorf("taught examples = %v, want %v", decl.Examples, wantExamples)
	}

	out.Reset()
	d.handle("create a folder called receipts")
	got = out.String()
	for _, want := range []string{
		"understood: mkdir",
		"arguments: folder=receipts",
		"would run: mkdir receipts",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("later phrasing missing %q:\n%s", want, got)
		}
	}
}

// TestTeachSkipWritesNothing: the offer answered with an
// empty line leaves the registry file byte-identical.
func TestTeachSkipWritesNothing(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	d, out := queueDemo(t, path, reg, []string{""})
	d.handle("create a folder")
	if strings.Contains(out.String(), "learned:") {
		t.Errorf("skip should learn nothing:\n%s", out.String())
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Error("registry file changed after a skipped teach offer")
	}
}

// TestTalkGetsNoTeachOffer: small talk has no closest
// script and no action word, so the offer is never made.
func TestTalkGetsNoTeachOffer(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	asked := false
	var out bytes.Buffer
	d := &demo{
		registryPath: path,
		reg:          reg,
		matcher:      cmdmatch.NewMatcher(reg),
		extract:      func(line string) (string, error) { return "", nil },
		nextLine: func(prompt string) (string, bool) {
			asked = true
			return "", true
		},
		out: &out,
	}
	for _, line := range []string{"how are you", "good morning everyone", "the weather is nice this week"} {
		out.Reset()
		d.handle(line)
		if out.String() != "" {
			t.Errorf("talk line %q should get silence:\n%s", line, out.String())
		}
	}
	if asked {
		t.Error("a teach offer was made for small talk")
	}
}

// TestWouldRunSlotsOnly pins the binding fix: values
// heard beyond the declared slots are reported but never
// reach the would-run line.
func TestWouldRunSlotsOnly(t *testing.T) {
	d, out := newTestDemo(t, "jim owen")
	d.handle("rename the photos in jim")
	got := out.String()
	if !strings.Contains(got, "(also heard, with no slot to bind: owen)") {
		t.Errorf("extra value should be reported:\n%s", got)
	}
	for _, line := range strings.Split(got, "\n") {
		if strings.HasPrefix(line, "would run:") && strings.Contains(line, "owen") {
			t.Errorf("unbound value reached the command line: %q", line)
		}
	}
	if !strings.Contains(got, "would run: ./scripts/rename-photos.sh jim\n") {
		t.Errorf("would-run should carry the bound slot only:\n%s", got)
	}
}

// TestWouldRunForms covers both command shapes: inline
// placeholders are substituted (and left visible when no
// value was heard); appended values stop at the slots.
func TestWouldRunForms(t *testing.T) {
	taught := cmdmatch.Declaration{Command: "mkdir {folder}", Slots: []string{"folder"}}
	if got := wouldRun(taught, []string{"trips"}); got != "mkdir trips" {
		t.Errorf("placeholder substitution: got %q", got)
	}
	if got := wouldRun(taught, nil); got != "mkdir {folder}" {
		t.Errorf("unbound placeholder should stay visible: got %q", got)
	}
	if got := wouldRun(taught, []string{"trips", "extra"}); got != "mkdir trips" {
		t.Errorf("extra values must not leak into a placeholder command: got %q", got)
	}
	appended := cmdmatch.Declaration{Command: "./x.sh", Slots: []string{"a"}}
	if got := wouldRun(appended, []string{"1", "2"}); got != "./x.sh 1" {
		t.Errorf("append should stop at the slots: got %q", got)
	}
	slotless := cmdmatch.Declaration{Command: "./x.sh"}
	if got := wouldRun(slotless, []string{"1"}); got != "./x.sh" {
		t.Errorf("slotless command takes no values: got %q", got)
	}
}

// TestParseTeachAnswerGuard pins the teach-answer
// guard's two sides: a declaration in disguise (a
// sentence carrying a file marker) is recognised for
// routing, with name and description parsed out of it;
// a bare command — the answer the prompt actually asks
// for — is not.
func TestParseTeachAnswerGuard(t *testing.T) {
	p, ok := parseTeachAnswer(johnDeleteDeclaration)
	if !ok {
		t.Fatal("John's declaration as a teach answer not recognised")
	}
	if p.name != "del" || p.file != "del.sh" || p.description != "deletes the first line of a file" {
		t.Errorf("guard parse = %+v, want del/del.sh/clean description", p)
	}
	p, ok = parseTeachAnswer("the command is in file del.sh")
	if !ok || p.name != "del" || p.file != "del.sh" || p.description != "" {
		t.Errorf("file-only answer = %+v, %v; want del/del.sh/no description yet", p, ok)
	}
	for _, bare := range []string{"./wipe.sh", "wipe.sh --force", "mkdir {folder}", "./scripts/make-folder.sh"} {
		if p, ok := parseTeachAnswer(bare); ok {
			t.Errorf("bare command %q routed as a declaration: %+v", bare, p)
		}
	}
}

// TestTeachAnswerDeclarationNotStoredRaw is the defect
// itself, end to end: a want earns the teach offer, the
// answer is a declaration sentence, and what lands in
// the registry is the DECLARED entry — command ./del.sh,
// slot file — never the sentence stored as a command
// with a filename baked in.
func TestTeachAnswerDeclarationNotStoredRaw(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := queueDemo(t, path, reg, []string{johnDeleteDeclaration})

	d.handle("create a folder")
	got := out.String()
	if !strings.Contains(got, "learned: del — deletes the first line of a file") {
		t.Errorf("teach answer should be learned as the del declaration:\n%s", got)
	}
	if strings.Contains(got, "i'll run i have a command") {
		t.Errorf("declaration sentence stored raw as a command:\n%s", got)
	}
	back, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry(after teach): %v", err)
	}
	decl, ok := back.Find("del")
	if !ok || decl.Command != "./del.sh" || len(decl.Slots) != 1 || decl.Slots[0] != "file" {
		t.Errorf("registry entry wrong: %+v (found %v)", decl, ok)
	}
	if _, ok := back.Find("mkdir"); ok {
		t.Error("the folder want itself should not have been taught — its answer declared a different command")
	}
}

func TestDeriveNameAndPlaceholders(t *testing.T) {
	for command, want := range map[string]string{
		"mkdir {folder}":           "mkdir",
		"./scripts/make-folder.sh": "make-folder",
		"backup.sh --full":         "backup",
		"{folder}":                 "folder",
		"":                         "my-command",
	} {
		if got := deriveName(command); got != want {
			t.Errorf("deriveName(%q) = %q, want %q", command, got, want)
		}
	}
	if got := placeholdersOf("cp {from} {to} {from}"); len(got) != 2 || got[0] != "from" || got[1] != "to" {
		t.Errorf("placeholdersOf dedup/order wrong: %v", got)
	}
	if got := placeholdersOf("./x.sh"); len(got) != 0 {
		t.Errorf("placeholdersOf of a plain command: %v", got)
	}
}
