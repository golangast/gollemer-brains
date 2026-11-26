package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/data"
)

// TestGenericDataSeparation pins the design's guarantee:
// the extractor's training file is GENERIC. No registry
// script's name, none of the registry's distinctive
// content words, and no registry description fragment may
// appear in it — if a future edit taught the extractor
// about a real registered script, the example's central
// claim (describe a new script, no retraining) would
// silently stop being measured honestly. The held-out
// unseen names are pinned out for the same reason.
func TestGenericDataSeparation(t *testing.T) {
	reg, err := cmdmatch.LoadRegistry(filepath.Join("..", "commands.json"))
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	pairs, err := data.LoadPairs(filepath.Join("..", "genericcommands.jsonl"))
	if err != nil {
		t.Fatalf("LoadPairs: %v", err)
	}
	if len(pairs) < 500 {
		t.Errorf("only %d generic rows; the extractor needs the shape taught in bulk", len(pairs))
	}
	distinctive := regexp.MustCompile(`\b(database|photos?|downloads?|mirrors?|notes?|logs?|website|server|migrate|deploy|rename|compress|sync)\b`)
	fragments := []string{
		"external drive", "photo library", "production server",
		"development server", "cloud backup", "camera names",
		"mirror servers", "seed data",
	}
	seen := map[string]bool{}
	for _, p := range pairs {
		if seen[p.Input] {
			t.Errorf("duplicate generic input: %q", p.Input)
		}
		seen[p.Input] = true
		text := p.Input + " " + p.Output
		for _, s := range reg.Scripts {
			if strings.Contains(text, s.Name) {
				t.Errorf("registry script %q leaked into generic row %q", s.Name, p.Input)
			}
		}
		if m := distinctive.FindString(text); m != "" {
			t.Errorf("generic row %q contains registry-distinctive word %q", p.Input, m)
		}
		for _, f := range fragments {
			if strings.Contains(text, f) {
				t.Errorf("generic row %q contains registry description fragment %q", p.Input, f)
			}
		}
		for _, unseen := range []string{"rudy", "petra"} {
			if strings.Contains(text, unseen) {
				t.Errorf("held-out unseen name %q leaked into generic row %q", unseen, p.Input)
			}
		}
		// An extractor row must be an honest extraction:
		// every value it teaches appears in its sentence.
		for _, v := range strings.Fields(p.Output) {
			if !strings.Contains(p.Input, v) {
				t.Errorf("generic row output value %q does not appear in its input %q", v, p.Input)
			}
		}
		if p.Intent != "" {
			t.Errorf("generic row %q carries intent %q; the generic file trains the extractor only", p.Input, p.Intent)
		}
	}
}

// newTestDemo builds a demo over the shipped registry with
// a stub extractor that echoes a fixed answer.
func newTestDemo(t *testing.T, extractAnswer string) (*demo, *bytes.Buffer) {
	t.Helper()
	reg, err := cmdmatch.LoadRegistry(filepath.Join("..", "commands.json"))
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	var out bytes.Buffer
	d := &demo{
		registryPath: filepath.Join("..", "commands.json"),
		reg:          reg,
		matcher:      cmdmatch.NewMatcher(reg),
		extract:      func(line string) (string, error) { return extractAnswer, nil },
		out:          &out,
	}
	return d, &out
}

func TestHandleUnderstands(t *testing.T) {
	d, out := newTestDemo(t, "jim jake")
	d.handle("migrate a database from folder jim to jake")
	got := out.String()
	for _, want := range []string{
		"understood: migrate-database",
		"arguments: from=jim, to=jake",
		"would run: ./scripts/migrate-db.sh jim jake",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q:\n%s", want, got)
		}
	}

	// A slotless script: no extractor consult, the bare
	// command is the would-run line.
	out.Reset()
	d.extract = func(line string) (string, error) {
		t.Error("extractor consulted for a slotless script")
		return "", nil
	}
	d.handle("back up my photos")
	if !strings.Contains(out.String(), "would run: ./scripts/backup-photos.sh\n") {
		t.Errorf("slotless match should print the bare command:\n%s", out.String())
	}

	// Talk lands on the abstention, not on a script.
	out.Reset()
	d.handle("how are you doing today")
	if !strings.Contains(out.String(), "i don't know which script you mean") {
		t.Errorf("talk line should abstain:\n%s", out.String())
	}
}

// TestAddFlowZeroRetrain is the design's headline property
// at unit level: a script that does not exist cannot be
// matched; after `add` declares it by description alone,
// the same kind of line matches it — same matcher rebuild
// the REPL performs on any registry edit, no training
// anywhere in the loop.
func TestAddFlowZeroRetrain(t *testing.T) {
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
	var out bytes.Buffer
	answers := []string{
		"", // skip the teach offer the first line now earns
		"water-plants",
		"./scripts/water-plants.sh",
		"logs watering the greenhouse plants and notes the time",
		"", // no slots
		"", // no examples — finish immediately
	}
	d := &demo{
		registryPath: path,
		reg:          reg,
		matcher:      cmdmatch.NewMatcher(reg),
		extract:      func(line string) (string, error) { return "", nil },
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

	d.handle("the script that logs watering the greenhouse")
	if !strings.Contains(out.String(), "i don't know which script you mean") {
		t.Fatalf("unregistered script should not match:\n%s", out.String())
	}
	out.Reset()
	d.handle("add")
	if !strings.Contains(out.String(), "added water-plants") {
		t.Fatalf("add flow did not confirm:\n%s", out.String())
	}
	out.Reset()
	d.handle("water the greenhouse plants and log it")
	if !strings.Contains(out.String(), "understood: water-plants") ||
		!strings.Contains(out.String(), "would run: ./scripts/water-plants.sh") {
		t.Errorf("freshly declared script did not match:\n%s", out.String())
	}
	// And the addition persisted to the registry file.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var back cmdmatch.Registry
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if _, ok := back.Find("water-plants"); !ok {
		t.Error("registry file does not contain water-plants after add")
	}
}

func TestCleanValues(t *testing.T) {
	for raw, want := range map[string]bool{
		"jim jake":    true,
		"holidays":    true,
		"":            false,
		"../etc":      false,
		"a/b":         false,
		`a\b`:         false,
		"two words":   true,
		"jim .. jake": false,
	} {
		if got := cleanValues(raw); got != want {
			t.Errorf("cleanValues(%q) = %v, want %v", raw, got, want)
		}
	}
}
