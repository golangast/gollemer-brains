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

// TestRoutingFileAndChat is the integration test for the
// one thing this example adds: the routing. A REAL intent
// classifier, trained on the shipped filecommands.jsonl,
// decides which half of the assistant each line goes to;
// a stub extractor stands in for the file brain's seq2seq
// and a stub stands in for the chat brain (training real
// seq2seq brains inside a unit test is slow and
// seed-sensitive — the README's measured session exercises
// both real brains end to end). Everything else — the
// routing itself, sanitizing, sandboxing, the delete gate
// and confirmation — is the real host code.
func TestRoutingFileAndChat(t *testing.T) {
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
	chatCalls := 0
	a := &assistant{
		dir:     dir,
		predict: clf.Predict,
		extract: func(line string) (string, error) { return "notes.txt", nil },
		chat: func(line string) (string, error) {
			chatCalls++
			return "i am well, thank you.", nil
		},
		out: &out,
	}
	a.confirmFn = func(prompt string) bool { return true }

	// A file meaning, in an unseen phrasing: the
	// classifier — not any parsing in this program — has
	// to route it to the file half, and the chat brain
	// must not be consulted.
	if !a.handle("could you make a file called notes.txt") {
		t.Fatal("handle returned false for a normal line")
	}
	if _, err := os.Stat(filepath.Join(dir, "notes.txt")); err != nil {
		t.Fatalf("routed create did not produce the file: %v\noutput:\n%s", err, out.String())
	}
	if !strings.Contains(out.String(), "intent: create_file") {
		t.Errorf("file line should report the predicted intent:\n%s", out.String())
	}
	if chatCalls != 0 {
		t.Errorf("chat brain was consulted for a file line (%d calls)", chatCalls)
	}

	// A line with no file meaning: it falls through to
	// the chat brain, gets its reply printed — and
	// nothing else. No intent line, no file action.
	out.Reset()
	a.handle("good morning friend")
	if chatCalls != 1 {
		t.Fatalf("chat line did not reach the chat brain (calls = %d)\noutput:\n%s", chatCalls, out.String())
	}
	if !strings.Contains(out.String(), "i am well, thank you.") {
		t.Errorf("chat line should print the chat reply:\n%s", out.String())
	}
	if strings.Contains(out.String(), "intent:") {
		t.Errorf("chat line should print nothing but the reply:\n%s", out.String())
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "notes.txt" {
		t.Errorf("chat line changed the sandbox: %v entries", len(entries))
	}

	// Delete, confirmed: routed to the file half, the
	// file goes, chat stays out of it.
	out.Reset()
	a.handle("get rid of notes.txt")
	if _, err := os.Stat(filepath.Join(dir, "notes.txt")); !os.IsNotExist(err) {
		t.Errorf("confirmed delete left the file behind (stat err = %v)\noutput:\n%s", err, out.String())
	}
	if chatCalls != 1 {
		t.Errorf("delete line reached the chat brain (calls = %d)", chatCalls)
	}
}

// TestChatLabelFallsThroughToChat pins the other half of
// the routing rule: when the file classifier confidently
// reads a line as "chat" — the fourth class the file data
// now teaches, so small talk has somewhere to land besides
// a file intent — the line is conversation. The chat brain
// answers it, no intent line is printed, the delete
// refusal wording never appears, and the sandbox is
// untouched.
func TestChatLabelFallsThroughToChat(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "demo-files")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	chatCalls := 0
	a := &assistant{
		dir: dir,
		predict: func(line string) (string, float64) {
			return "chat", 0.97
		},
		extract: func(line string) (string, error) { return "notes.txt", nil },
		chat: func(line string) (string, error) {
			chatCalls++
			return "i am well, thank you.", nil
		},
		confirmFn: func(prompt string) bool { return true },
		out:       &out,
	}
	a.handle("well, hello")
	if chatCalls != 1 {
		t.Fatalf("chat-labelled line did not reach the chat brain (calls = %d)\noutput:\n%s", chatCalls, out.String())
	}
	if !strings.Contains(out.String(), "i am well, thank you.") {
		t.Errorf("chat-labelled line should print the chat reply:\n%s", out.String())
	}
	if strings.Contains(out.String(), "intent:") {
		t.Errorf("chat-labelled line should print nothing but the reply:\n%s", out.String())
	}
	if strings.Contains(out.String(), "not confident enough to delete") {
		t.Errorf("chat-labelled line printed the delete-refusal wording:\n%s", out.String())
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Errorf("chat-labelled line changed the sandbox: %d entries", len(entries))
	}
}

// TestDeleteBelowGateRefusesInsteadOfChatting pins the
// routing rule that is easiest to get wrong: a line the
// classifier reads as delete_file, but not confidently
// enough to act on, is REFUSED as a command. It must not
// fall through to the chat brain and turn into small
// talk — a refused command stays refused.
func TestDeleteBelowGateRefusesInsteadOfChatting(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "demo-files")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	chatCalls := 0
	a := &assistant{
		dir: dir,
		predict: func(line string) (string, float64) {
			return "delete_file", deleteMinConfidence - 0.2
		},
		extract: func(line string) (string, error) { return "notes.txt", nil },
		chat: func(line string) (string, error) {
			chatCalls++
			return "i am well, thank you.", nil
		},
		confirmFn: func(prompt string) bool { return true },
		out:       &out,
	}
	a.handle("delete notes.txt")
	if !strings.Contains(out.String(), "not confident enough to delete") {
		t.Errorf("below-gate delete should print the refusal wording:\n%s", out.String())
	}
	if chatCalls != 0 {
		t.Errorf("refused delete fell through to the chat brain (%d calls)", chatCalls)
	}
	if _, err := os.Stat(filepath.Join(dir, "notes.txt")); err != nil {
		t.Errorf("refused delete removed the file: %v", err)
	}
}
