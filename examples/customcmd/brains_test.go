package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/intent"
)

// The brains tests wire the routing to a REAL intent
// classifier — trained in the test on a handful of rows
// with unmistakable vocabularies — and stub answers for
// the chat and file brains, so the ROUTING is what is
// under test: which brain a line reaches, and what the
// user is shown about the choice they never had to make.

func brainsClf(t *testing.T) *intent.Classifier {
	t.Helper()
	row := func(in, label, kind string) data.Pair {
		return data.Pair{Input: in, Intent: label, Kind: kind}
	}
	pairs := []data.Pair{
		row("hello there friend", "greeting", "talk"),
		row("hello good morning friend", "greeting", "talk"),
		row("hello there, how is your day friend", "greeting", "talk"),
		row("what is a widget", "explain_widget", "find"),
		row("what does a widget do", "explain_widget", "find"),
		row("explain the widget to me", "explain_widget", "find"),
		row("create a file named x", "create_file", "make"),
		row("make a file named y", "create_file", "make"),
		row("create a new file please", "create_file", "make"),
		row("remind me to call mother", "set_reminder", "schedule"),
		row("remind me about the dentist", "set_reminder", "schedule"),
		row("set a reminder for noon", "set_reminder", "schedule"),
	}
	clf, err := intent.Train(pairs, intent.DefaultThreshold)
	if err != nil {
		t.Fatalf("intent.Train: %v", err)
	}
	return clf
}

// brainsDemo builds a demo over an EMPTY registry (the
// matcher abstains on everything, so every line reaches
// the routing) with the test classifier and stub brains.
func brainsDemo(t *testing.T, answers []string) (*demo, *bytes.Buffer) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "commands.json")
	reg := &cmdmatch.Registry{}
	if err := reg.Save(path); err != nil {
		t.Fatalf("Save empty registry: %v", err)
	}
	d, out := queueDemo(t, path, reg, answers)
	d.clf = brainsClf(t)
	d.chatAnswer = func(line string) (string, error) {
		return "stub chat reply", nil
	}
	d.filePredict = func(line string) (string, float64) {
		if strings.Contains(line, "file") {
			return "create_file", 0.9
		}
		return "", 0
	}
	d.fileExtract = func(line string) (string, error) {
		return "trips.txt", nil
	}
	return d, out
}

// TestBrainsChatRoute: a greeting the matcher cannot
// claim is answered by the chat brain, labelled — the
// answer alone, no intent annotation under it (John's
// rule) — and no teach prompt is served.
func TestBrainsChatRoute(t *testing.T) {
	d, out := brainsDemo(t, nil)
	d.handle("hello there friend")
	got := out.String()
	if !strings.Contains(got, "chat brain: stub chat reply") {
		t.Errorf("chat route output missing the answer:\n%s", got)
	}
	if strings.Contains(got, "(intent:") {
		t.Errorf("chat answer still carries an intent annotation:\n%s", got)
	}
	if strings.Contains(got, "want to teach me") {
		t.Errorf("small talk earned a teach offer:\n%s", got)
	}
}

// TestBrainsQuestionRoute: a find-kind question goes to
// the chat brain too — questions it can answer are its
// job, not the teach loop's.
func TestBrainsQuestionRoute(t *testing.T) {
	d, out := brainsDemo(t, nil)
	d.handle("what is a widget")
	got := out.String()
	if !strings.Contains(got, "chat brain: stub chat reply") {
		t.Errorf("question did not take the chat route:\n%s", got)
	}
	if strings.Contains(got, "(intent:") {
		t.Errorf("question answer still carries an intent annotation:\n%s", got)
	}
}

// TestBrainsFileRoute: a file want the matcher cannot
// claim is read by the file brain — label and filename
// shown, understanding only.
func TestBrainsFileRoute(t *testing.T) {
	d, out := brainsDemo(t, nil)
	d.handle("create a file named trips.txt")
	got := out.String()
	for _, want := range []string{
		"file brain: reads this as create_file — file trips.txt",
		"understanding only",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("file route output missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "want to teach me") {
		t.Errorf("a file want the file brain read earned a teach offer:\n%s", got)
	}
}

// TestBrainsTeachNoRead: a want-kind meaning with no
// script (set_reminder) reaches the teach offer, and the
// offer stands alone — no intent read printed before it
// (John's rule).
func TestBrainsTeachNoRead(t *testing.T) {
	d, out := brainsDemo(t, []string{""}) // skip the offer
	d.handle("remind me to call mother")
	got := out.String()
	if strings.Contains(got, "this reads as:") {
		t.Errorf("teach offer still shows the intent read:\n%s", got)
	}
	if strings.Contains(got, "chat brain:") || strings.Contains(got, "file brain:") {
		t.Errorf("a reminder want was routed to a brain:\n%s", got)
	}
}

// TestBrainsUnwiredIsTheOldBehaviour: with no brains
// wired, a greeting is the plain don't-know it always
// was — the routing's nil-safety is the regression fence
// the whole existing suite stands on.
func TestBrainsUnwiredIsTheOldBehaviour(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "commands.json")
	reg := &cmdmatch.Registry{}
	if err := reg.Save(path); err != nil {
		t.Fatalf("Save empty registry: %v", err)
	}
	d, out := queueDemo(t, path, reg, nil)
	d.handle("hello there friend")
	got := out.String()
	if got != "" {
		t.Errorf("unwired greeting should get silence (no brain, no want — just the next prompt):\n%q", got)
	}
}

// TestModalGists: the gist is the label's most frequent
// output; ties keep the earliest in file order; unlabelled
// and empty-output rows never contribute.
func TestModalGists(t *testing.T) {
	pairs := []data.Pair{
		{Input: "a", Output: "gist one", Intent: "explain_x"},
		{Input: "b", Output: "gist two", Intent: "explain_x"},
		{Input: "c", Output: "gist one", Intent: "explain_x"},
		{Input: "d", Output: "first", Intent: "explain_y"},
		{Input: "e", Output: "second", Intent: "explain_y"},
		{Input: "f", Output: "no label here"},
		{Input: "g", Output: "", Intent: "explain_z"},
	}
	g := modalGists(pairs)
	if g["explain_x"] != "gist one" {
		t.Errorf("explain_x gist = %q, want %q", g["explain_x"], "gist one")
	}
	if g["explain_y"] != "first" {
		t.Errorf("explain_y gist = %q, want the tie's earliest %q", g["explain_y"], "first")
	}
	if _, ok := g["explain_z"]; ok {
		t.Errorf("explain_z has only an empty output and must have no gist")
	}
	if len(g) != 2 {
		t.Errorf("gists = %v, want exactly explain_x and explain_y", g)
	}
}

// TestLooksLikeQuestion: punctuation or an opener marks
// the shape; plain statements do not.
func TestLooksLikeQuestion(t *testing.T) {
	for _, q := range []string{
		"what is a widget", "what is a widget?", "how do you run it",
		"tell me about widgets", "explain the widget", "is it ready",
		"can you explain the widget",
	} {
		if !looksLikeQuestion(q) {
			t.Errorf("looksLikeQuestion(%q) = false, want true", q)
		}
	}
	for _, s := range []string{
		"this error message is confusing me", "hello there friend",
		"i do not understand this error message",
	} {
		if looksLikeQuestion(s) {
			t.Errorf("looksLikeQuestion(%q) = true, want false", s)
		}
	}
}

// TestBrainsGistRoute: a confident label on a
// question-shaped line is answered with the concept's
// taught gist, not the brain's generation; a statement
// with the same label keeps the generation path.
func TestBrainsGistRoute(t *testing.T) {
	d, out := brainsDemo(t, nil)
	d.chatGists = map[string]string{"explain_widget": "a widget is the taught gist."}
	d.handle("what is a widget")
	if !strings.Contains(out.String(), "chat brain: a widget is the taught gist.") {
		t.Errorf("question line did not get the gist answer:\n%s", out.String())
	}

	d2, out2 := brainsDemo(t, nil)
	d2.chatGists = map[string]string{"explain_widget": "a widget is the taught gist."}
	ans, err := d2.answerChat("this error message is confusing me", intentRead{label: "explain_widget"})
	if err != nil || ans != "stub chat reply" {
		t.Errorf("statement answerChat = %q, %v; want the generated stub reply", ans, err)
	}
	_ = out2
}
