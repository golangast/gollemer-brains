package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

const johnDeclaration = "i have a command named copy and it will copy files from one folder to another and it is in file copy.sh"

const johnDeleteDeclaration = "i have a command that deletes the first line of a file and the command is in file del.sh"

// TestParseDeclarationUnnamedJohn pins John's delete
// sentence: no named/called marker at all, yet it IS a
// declaration — "i have a command that …" plus a file.
// The name is the file's stem, the description is cut
// at the file-clause introducer ("and the command is"),
// not at the file token, and the file rule gives the
// entry its one slot.
func TestParseDeclarationUnnamedJohn(t *testing.T) {
	p, ok := parseDeclaration(johnDeleteDeclaration)
	if !ok {
		t.Fatal("John's unnamed declaration not recognised")
	}
	if p.name != "del" {
		t.Errorf("name = %q, want del (the file's stem)", p.name)
	}
	if p.file != "del.sh" {
		t.Errorf("file = %q, want del.sh", p.file)
	}
	if p.description != "deletes the first line of a file" {
		t.Errorf("description = %q, want %q", p.description, "deletes the first line of a file")
	}
	slots := declarationSlots(p.description, "./del.sh")
	if len(slots) != 1 || slots[0] != "file" {
		t.Errorf("slots = %v, want [file]", slots)
	}
}

// TestParseDeclarationNamedDelete: the same sentence
// with "named delete" inserted parses through the named
// door to the SAME clean description — the tail
// fragment "and the command is" must not survive the
// cut — and the same file slot.
func TestParseDeclarationNamedDelete(t *testing.T) {
	line := "i have a command named delete that deletes the first line of a file and the command is in file del.sh"
	p, ok := parseDeclaration(line)
	if !ok {
		t.Fatal("named delete declaration not recognised")
	}
	if p.name != "delete" {
		t.Errorf("name = %q, want delete", p.name)
	}
	if p.description != "deletes the first line of a file" {
		t.Errorf("description = %q, want %q", p.description, "deletes the first line of a file")
	}
	if slots := declarationSlots(p.description, "./del.sh"); len(slots) != 1 || slots[0] != "file" {
		t.Errorf("slots = %v, want [file]", slots)
	}
}

// TestDeclarationSlotsFileRule pins the slot rules'
// order: placeholders first, then the from/to pair,
// then the single file slot for a description that
// mentions a file; a description mentioning neither
// gets no slots.
func TestDeclarationSlotsFileRule(t *testing.T) {
	cases := []struct {
		desc, command string
		want          []string
	}{
		{"deletes the first line of a file", "./del.sh", []string{"file"}},
		{"shows the file", "./show.sh", []string{"file"}},
		{"copies a file from one folder to another", "./copy.sh", []string{"from", "to"}},
		{"backs up my photo library", "./backup.sh", nil},
		{"makes a folder", "mkdir {folder}", []string{"folder"}},
	}
	for _, c := range cases {
		got := declarationSlots(c.desc, c.command)
		if len(got) != len(c.want) {
			t.Errorf("declarationSlots(%q, %q) = %v, want %v", c.desc, c.command, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("declarationSlots(%q, %q) = %v, want %v", c.desc, c.command, got, c.want)
				break
			}
		}
	}
}

// TestJohnDeleteEndToEnd is John's handtest, verbatim:
// the declaration line, then the same want with two
// different filenames. The filenames must BIND — the
// second want's dog.txt replaces the first's cat.txt in
// the would-run line, which is exactly what the broken
// build could not do.
func TestJohnDeleteEndToEnd(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "commands.json")
	reg := &cmdmatch.Registry{}
	if err := reg.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	reg, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	var out bytes.Buffer
	d := &demo{
		registryPath: path,
		reg:          reg,
		matcher:      cmdmatch.NewMatcher(reg),
		extract: func(line string) (string, error) {
			switch {
			case strings.Contains(line, "cat.txt"):
				return "cat.txt", nil
			case strings.Contains(line, "dog.txt"):
				return "dog.txt", nil
			}
			return "", nil
		},
		nextLine: func(prompt string) (string, bool) {
			t.Errorf("no follow-up or teach prompt expected; got %q", prompt)
			return "", false
		},
		out: &out,
	}

	d.handle(johnDeleteDeclaration)
	got := out.String()
	for _, want := range []string{
		"learned: del — deletes the first line of a file",
		"command: ./del.sh (slots: file)",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("declaration output missing %q:\n%s", want, got)
		}
	}

	for _, tc := range []struct{ line, file string }{
		{"delete the first line of the file cat.txt", "cat.txt"},
		{"delete the first line of the file dog.txt", "dog.txt"},
	} {
		out.Reset()
		d.handle(tc.line)
		got = out.String()
		for _, want := range []string{
			"understood: del",
			"arguments: file=" + tc.file,
			// The preview: the filename worked into the
			// description's own sentence, the caution a
			// remove-family description earns, and the
			// runs line agreeing with the would-run line.
			"preview: deletes the first line of the file " + tc.file,
			"  runs: ./del.sh " + tc.file,
			"  nothing has run yet.",
			"  careful: this command removes or changes things.",
			"would run: ./del.sh " + tc.file,
		} {
			if !strings.Contains(got, want) {
				t.Errorf("want %q missing %q:\n%s", tc.line, want, got)
			}
		}
	}

	back, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry(after): %v", err)
	}
	decl, ok := back.Find("del")
	if !ok || decl.Command != "./del.sh" || decl.Description != "deletes the first line of a file" ||
		len(decl.Slots) != 1 || decl.Slots[0] != "file" {
		t.Errorf("persisted entry wrong: %+v (found %v)", decl, ok)
	}
}

// TestParseDeclarationJohn pins the parse of John's
// sentence: name, file, and the doing-clause isolated
// from between "it will" and the file clause.
func TestParseDeclarationJohn(t *testing.T) {
	p, ok := parseDeclaration(johnDeclaration)
	if !ok {
		t.Fatal("John's sentence not recognised as a declaration")
	}
	if p.name != "copy" {
		t.Errorf("name = %q, want copy", p.name)
	}
	if p.file != "copy.sh" {
		t.Errorf("file = %q, want copy.sh", p.file)
	}
	if p.description != "copy files from one folder to another" {
		t.Errorf("description = %q, want %q", p.description, "copy files from one folder to another")
	}
	if got := runnableCommand(p.file); got != "./copy.sh" {
		t.Errorf("runnableCommand = %q, want ./copy.sh", got)
	}
	slots := declarationSlots(p.description, "./copy.sh")
	if len(slots) != 2 || slots[0] != "from" || slots[1] != "to" {
		t.Errorf("slots = %v, want [from to]", slots)
	}
}

// TestParseDeclarationShapes covers the other trigger
// shapes and the paths that must NOT fire: an ordinary
// want has no markers, small talk none either, and "my
// script <verb>…" without a file or doing-marker is a
// statement, not a declaration.
func TestParseDeclarationShapes(t *testing.T) {
	fires := map[string]string{ // line -> expected name
		"there is a command called zap that zaps the cache": "zap",
		"add a script named tidy that tidies the desk":      "tidy",
		"my command copy is in file copy.sh":                "copy",
	}
	for line, want := range fires {
		p, ok := parseDeclaration(line)
		if !ok || p.name != want {
			t.Errorf("parseDeclaration(%q) = %+v, %v; want name %q", line, p, ok, want)
		}
	}
	silent := []string{
		"copy files from jim to jake",      // a want, not a declaration
		"how are you",                      // small talk
		"my script backs up the photos",    // statement: no file, no marker
		"my command is broken",             // "is" is not a name
		"run the command that lists files", // no named/called marker
	}
	for _, line := range silent {
		if p, ok := parseDeclaration(line); ok {
			t.Errorf("parseDeclaration(%q) fired with %+v; want no declaration", line, p)
		}
	}
}

// TestDeclarationFlowEndToEnd: on a fresh registry,
// John's sentence is learned whole — no follow-up
// questions needed — and the want then chooses the
// declared command with its values bound.
func TestDeclarationFlowEndToEnd(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "commands.json")
	reg := &cmdmatch.Registry{}
	if err := reg.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	reg, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	var out bytes.Buffer
	d := &demo{
		registryPath: path,
		reg:          reg,
		matcher:      cmdmatch.NewMatcher(reg),
		extract: func(line string) (string, error) {
			if strings.Contains(line, "jim") {
				return "jim jake", nil
			}
			return "", nil
		},
		nextLine: func(prompt string) (string, bool) {
			t.Errorf("no follow-up expected for a complete declaration; got %q", prompt)
			return "", false
		},
		out: &out,
	}

	d.handle(johnDeclaration)
	got := out.String()
	for _, want := range []string{
		"learned: copy — copy files from one folder to another",
		"command: ./copy.sh (slots: from, to)",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("declaration output missing %q:\n%s", want, got)
		}
	}

	out.Reset()
	d.handle("copy files from jim to jake")
	got = out.String()
	for _, want := range []string{
		"understood: copy",
		"arguments: from=jim, to=jake",
		"would run: ./copy.sh jim jake",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("want after declaration missing %q:\n%s", want, got)
		}
	}

	back, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry(after declaration): %v", err)
	}
	decl, ok := back.Find("copy")
	if !ok || decl.Command != "./copy.sh" || len(decl.Slots) != 2 {
		t.Errorf("persisted declaration wrong: %+v (found %v)", decl, ok)
	}
}

// TestDeclarationAsksWhatItDoes: a declaration whose
// doing-clause cannot be isolated asks exactly one
// follow-up and uses the answer as the description.
func TestDeclarationAsksWhatItDoes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "commands.json")
	reg := &cmdmatch.Registry{}
	if err := reg.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	reg, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	d, out := queueDemo(t, path, reg, []string{"zaps the cache clean"})
	d.handle("i have a command named zap and it is in file zap.sh")
	got := out.String()
	if !strings.Contains(got, "learned: zap — zaps the cache clean") {
		t.Errorf("follow-up answer should become the description:\n%s", got)
	}
	if !strings.Contains(got, "command: ./zap.sh (no slots)") {
		t.Errorf("confirmation should name the command and slots:\n%s", got)
	}
}

// TestWantCreatesNothing: a want typed at a fresh
// registry is not a declaration — it goes down the want
// path (here: abstain, teach offer, skipped) and the
// registry stays empty.
func TestWantCreatesNothing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "commands.json")
	reg := &cmdmatch.Registry{}
	if err := reg.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	reg, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	d, _ := queueDemo(t, path, reg, []string{""}) // skip any teach offer
	d.handle("copy files from jim to jake")
	back, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry(after want): %v", err)
	}
	if len(back.Scripts) != 0 {
		t.Errorf("a want created registry entries: %+v", back.Scripts)
	}
}
