package main

import (
	"strings"
	"testing"
)

// The inspect mode (inspect.go): with inspect on, a
// handled line's output gains [inspect] provenance
// lines after its ordinary output; with inspect off,
// the output is byte-identical to the un-inspected
// program — the mode must be the only difference.

func TestInspectOffByDefault(t *testing.T) {
	d, out := newTestDemo(t, "jim jake")
	d.handle("migrate a database from folder jim to jake")
	d.handle("how are you doing today")
	if strings.Contains(out.String(), "[inspect]") {
		t.Errorf("inspect lines printed with inspect off:\n%s", out.String())
	}
	if len(d.trace) != 0 {
		t.Errorf("trace collected with inspect off: %v", d.trace)
	}
}

func TestInspectMatcherTrace(t *testing.T) {
	d, out := newTestDemo(t, "jim jake")
	d.inspect = true
	d.handle("migrate a database from folder jim to jake")
	got := out.String()
	if !strings.Contains(got, "understood: migrate-database") {
		t.Fatalf("ordinary presentation missing:\n%s", got)
	}
	if !strings.Contains(got, "[inspect] matcher: migrate-database (1.27)") {
		t.Errorf("matcher trace missing or wrong:\n%s", got)
	}
	// The trace names how words reached the profile:
	// "database" is a defining word of the description.
	if !strings.Contains(got, "databas: defining") && !strings.Contains(got, "database: defining") {
		t.Errorf("matcher trace lacks word provenance:\n%s", got)
	}
	// The trace comes AFTER the ordinary output.
	if strings.Index(got, "[inspect]") < strings.Index(got, "would run:") {
		t.Errorf("trace printed before the ordinary output:\n%s", got)
	}
}

func TestInspectFallbackTrace(t *testing.T) {
	d, out := brainsDemo(t, nil)
	d.inspect = true
	d.clf = wireClf()
	d.factorTbl = nil
	d.chatGists = map[string]string{"explain_widget": "a widget is a small useful thing."}
	d.handle("what is a widget")
	got := out.String()
	if !strings.Contains(got, "chat brain: a widget is a small useful thing.") {
		t.Fatalf("ordinary gist answer missing:\n%s", got)
	}
	// (No matcher trace: brainsDemo's registry is empty,
	// so there is no best script to report.)
	for _, want := range []string{
		"[inspect] intent: abstained",
		"[inspect] fallback: kind guess find + the line's objects compose explain_widget (widget verbatim)",
		"[inspect] route: chat brain (label explain_widget, kind find)",
		"[inspect] chat: the taught gist for explain_widget answers",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("inspect output missing %q:\n%s", want, got)
		}
	}
}

func TestInspectParentWalkTrace(t *testing.T) {
	// A kind guess of talk names a neighbourhood whose
	// only child here (tell_story) composes nothing for
	// this line, so the fallback walks: the trace must
	// show the kind-tree step it took.
	d, _ := brainsDemo(t, nil)
	d.inspect = true
	d.clf = wireClf()
	d.factorTbl = nil
	if _, ok := d.fallbackRead("what is a widget", "talk"); !ok {
		t.Fatal("parent walk did not adopt")
	}
	joined := strings.Join(d.trace, "\n")
	if !strings.Contains(joined, "parent walk — top label explain_widget") ||
		!strings.Contains(joined, "parent kind find") {
		t.Errorf("walk trace missing:\n%s", joined)
	}
}
