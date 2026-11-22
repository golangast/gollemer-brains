package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/intent"
)

func TestSanitizeName(t *testing.T) {
	good := map[string]string{
		"notes.txt":    "notes.txt",
		"  todo.md  ":  "todo.md",
		"my-file_2.md": "my-file_2.md",
	}
	for raw, want := range good {
		got, err := sanitizeName(raw)
		if err != nil || got != want {
			t.Errorf("sanitizeName(%q) = %q, %v; want %q, nil", raw, got, err, want)
		}
	}
	bad := []string{
		"../evil.txt",     // climbs out of the sandbox
		"..",              // the parent itself
		"a/b.txt",         // a path, not a name
		`a\b.txt`,         // a path on the other separator
		"a..txt",          // contains ".."
		"",                // nothing
		"notes.txt extra", // two tokens: not a bare filename
		"make",            // a word, no extension
		".hidden",         // no stem before the dot
		"notes.",          // no extension
		"/etc/passwd",     // absolute path
	}
	for _, raw := range bad {
		if got, err := sanitizeName(raw); err == nil {
			t.Errorf("sanitizeName(%q) = %q, nil; want an error", raw, got)
		}
	}
}

func TestSandboxPathConfinement(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "demo-files")
	p, err := sandboxPath(dir, "notes.txt")
	if err != nil {
		t.Fatalf("sandboxPath: %v", err)
	}
	if p != filepath.Join(dir, "notes.txt") {
		t.Errorf("sandboxPath = %q, want the name directly inside the sandbox", p)
	}
	// Even a name that somehow slipped past sanitizeName
	// must not resolve outside the sandbox.
	if _, err := sandboxPath(dir, filepath.Join("..", "evil.txt")); err == nil {
		t.Error("sandboxPath accepted a name that escapes the sandbox")
	}
}

// TestDemoDrivesCreateAndDelete is the integration test: a
// REAL intent classifier, trained on the shipped
// filecommands.jsonl, predicts the meanings; a stub
// extractor stands in for the seq2seq brain (training one
// inside a unit test is slow and seed-sensitive — the
// README's proof run exercises the real extractor end to
// end). Everything else — sanitizing, sandboxing, the
// delete gate and confirmation — is the real host code.
func TestDemoDrivesCreateAndDelete(t *testing.T) {
	pairs, err := data.LoadPairs(filepath.Join("..", "filecommands.jsonl"))
	if err != nil {
		t.Fatalf("LoadPairs: %v", err)
	}
	clf, err := intent.Train(pairs, intent.DefaultThreshold)
	if err != nil {
		t.Fatalf("intent.Train: %v", err)
	}
	dir := filepath.Join(t.TempDir(), "demo-files")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	d := &demo{
		dir:     dir,
		predict: clf.Predict,
		extract: func(line string) (string, error) { return "notes.txt", nil },
		out:     &out,
	}
	d.confirmFn = func(prompt string) bool { return true }

	// An unseen phrasing of a trained meaning: the
	// classifier — not any parsing in this program — has
	// to land it on create_file.
	if !d.handle("could you make a file called notes.txt") {
		t.Fatal("handle returned false for a normal line")
	}
	if _, err := os.Stat(filepath.Join(dir, "notes.txt")); err != nil {
		t.Fatalf("create did not produce the file: %v\noutput:\n%s", err, out.String())
	}
	if !strings.Contains(out.String(), "intent: create_file") {
		t.Errorf("output does not report the predicted intent:\n%s", out.String())
	}

	// A line with no meaning in the set — and no words in
	// common with it — must do nothing.
	out.Reset()
	d.handle("good morning friend")
	if !strings.Contains(out.String(), "nothing done") {
		t.Errorf("no-meaning line should do nothing, got:\n%s", out.String())
	}

	// Delete, confirmed: the file goes.
	out.Reset()
	d.handle("get rid of notes.txt")
	if _, err := os.Stat(filepath.Join(dir, "notes.txt")); !os.IsNotExist(err) {
		t.Errorf("confirmed delete left the file behind (stat err = %v)\noutput:\n%s", err, out.String())
	}

	// Delete, refused at the prompt: the file stays.
	d.handle("could you make a file called notes.txt")
	d.confirmFn = func(prompt string) bool { return false }
	out.Reset()
	d.handle("get rid of notes.txt")
	if _, err := os.Stat(filepath.Join(dir, "notes.txt")); err != nil {
		t.Errorf("unconfirmed delete removed the file\noutput:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "left notes.txt alone") {
		t.Errorf("unconfirmed delete should say the file was left alone:\n%s", out.String())
	}
}
