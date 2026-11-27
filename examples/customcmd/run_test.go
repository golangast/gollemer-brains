package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// The gated-execution tests (run.go). Every script here
// is harmless and created by the test in its own temp
// dir: an echo, a file-maker, an argv recorder, and a
// script that exits 3. Demos are interactive with a
// queued nextLine, so the gate can be answered; the
// gate counter proves how many times a confirmation was
// actually asked for — a step that was never offered is
// a step whose prompt never appears.

// writeTestScript drops a script with the given body and
// permissions into dir and returns its path.
func writeTestScript(t *testing.T, dir, name, body string, perm os.FileMode) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(body), perm); err != nil {
		t.Fatalf("WriteFile %s: %v", name, err)
	}
	return p
}

// gateQueue is a nextLine that serves answers in order
// and counts how many were asked for.
type gateQueue struct {
	answers []string
	asked   int
}

func (g *gateQueue) next(prompt string) (string, bool) {
	g.asked++
	if len(g.answers) == 0 {
		return "", false
	}
	a := g.answers[0]
	g.answers = g.answers[1:]
	return a, true
}

// execTestDemo builds an interactive demo over a temp
// registry holding the given declarations, with the
// extractor stubbed to answer extractAnswer for any line
// and the gate fed by answers.
func execTestDemo(t *testing.T, scripts []cmdmatch.Declaration, extractAnswer string, answers []string) (*demo, *bytes.Buffer, *gateQueue) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "commands.json")
	reg := &cmdmatch.Registry{Scripts: scripts}
	if err := reg.Save(path); err != nil {
		t.Fatalf("Save registry: %v", err)
	}
	loaded, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	g := &gateQueue{answers: answers}
	var out bytes.Buffer
	d := &demo{
		registryPath: path,
		reg:          loaded,
		matcher:      cmdmatch.NewMatcher(loaded),
		extract:      func(string) (string, error) { return extractAnswer, nil },
		nextLine:     g.next,
		out:          &out,
		interactive:  true,
	}
	return d, &out, g
}

// presentPlan presents hand-built steps the way
// handlePlan does: context shared, then printed.
func presentPlan(d *demo, steps []planStep) {
	shareContext(steps)
	d.printPlan(steps, nil)
}

func TestRunExecutesPresentedCommand(t *testing.T) {
	dir := t.TempDir()
	echo := writeTestScript(t, dir, "echo.sh", "#!/bin/sh\necho hello from the script\n", 0o755)
	decl := cmdmatch.Declaration{
		Name: "say-hello", Command: echo,
		Description: "prints hello to the screen",
		Examples:    []string{"prints hello to the screen"},
	}
	d, out, g := execTestDemo(t, []cmdmatch.Declaration{decl}, "", []string{"y"})
	d.handle("prints hello to the screen")
	out.Reset()
	d.handle("run")
	got := out.String()
	for _, want := range []string{
		"step 1: " + echo,
		"hello from the script",
		"done: ran 1 of 1 step.",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("run output missing %q:\n%s", want, got)
		}
	}
	if g.asked != 1 {
		t.Errorf("gate asked %d times, want 1", g.asked)
	}
}

func TestRunItAlsoTriggers(t *testing.T) {
	dir := t.TempDir()
	echo := writeTestScript(t, dir, "echo.sh", "#!/bin/sh\necho hello from the script\n", 0o755)
	decl := cmdmatch.Declaration{
		Name: "say-hello", Command: echo,
		Description: "prints hello to the screen",
		Examples:    []string{"prints hello to the screen"},
	}
	d, out, _ := execTestDemo(t, []cmdmatch.Declaration{decl}, "", []string{"yes"})
	d.handle("prints hello to the screen")
	out.Reset()
	d.handle("run it")
	if !strings.Contains(out.String(), "done: ran 1 of 1 step.") {
		t.Errorf("'run it' did not run the presentation:\n%s", out.String())
	}
}

func TestRunAloneIsSilence(t *testing.T) {
	d, out, g := execTestDemo(t, nil, "", nil)
	d.handle("run")
	d.handle("run it")
	if out.String() != "" {
		t.Errorf("bare run with nothing presented printed %q, want silence", out.String())
	}
	if g.asked != 0 {
		t.Errorf("gate asked %d times with nothing presented", g.asked)
	}
}

func TestRunNeedsTheNextLine(t *testing.T) {
	dir := t.TempDir()
	echo := writeTestScript(t, dir, "echo.sh", "#!/bin/sh\necho hello from the script\n", 0o755)
	decl := cmdmatch.Declaration{
		Name: "say-hello", Command: echo,
		Description: "prints hello to the screen",
		Examples:    []string{"prints hello to the screen"},
	}
	d, out, _ := execTestDemo(t, []cmdmatch.Declaration{decl}, "", []string{"y"})
	d.handle("prints hello to the screen")
	d.handle("how are you") // any other line consumes the presentation
	out.Reset()
	d.handle("run")
	if out.String() != "" {
		t.Errorf("run after an intervening line printed %q, want silence", out.String())
	}
}

func TestRunDeclineStopsPlan(t *testing.T) {
	dir := t.TempDir()
	echo := writeTestScript(t, dir, "echo.sh", "#!/bin/sh\necho hello from the script\n", 0o755)
	made := filepath.Join(dir, "made.txt")
	touch := writeTestScript(t, dir, "touch.sh", "#!/bin/sh\ntouch "+made+"\n", 0o755)
	echoDecl := cmdmatch.Declaration{Name: "say-hello", Command: echo, Description: "prints hello to the screen"}
	touchDecl := cmdmatch.Declaration{Name: "make-file", Command: touch, Description: "makes the file"}
	d, out, _ := execTestDemo(t, []cmdmatch.Declaration{echoDecl, touchDecl}, "", []string{"y", "n"})
	presentPlan(d, []planStep{
		{segment: "prints hello to the screen", decl: echoDecl, matched: true},
		{segment: "makes the file", decl: touchDecl, matched: true},
	})
	out.Reset()
	d.handle("run")
	got := out.String()
	if !strings.Contains(got, "hello from the script") {
		t.Errorf("first step did not run:\n%s", got)
	}
	if _, err := os.Stat(made); !os.IsNotExist(err) {
		t.Errorf("declined step ran anyway: %s exists", made)
	}
	if !strings.Contains(got, "stopped: ran 1 of 2 steps — step 2 was not confirmed.") {
		t.Errorf("summary missing or wrong:\n%s", got)
	}
}

func TestRunFailureStopsPlan(t *testing.T) {
	dir := t.TempDir()
	fail := writeTestScript(t, dir, "fail.sh", "#!/bin/sh\necho something broke\nexit 3\n", 0o755)
	echo := writeTestScript(t, dir, "echo.sh", "#!/bin/sh\necho hello from the script\n", 0o755)
	failDecl := cmdmatch.Declaration{Name: "break-it", Command: fail, Description: "breaks on purpose"}
	echoDecl := cmdmatch.Declaration{Name: "say-hello", Command: echo, Description: "prints hello to the screen"}
	d, out, g := execTestDemo(t, []cmdmatch.Declaration{failDecl, echoDecl}, "", []string{"y", "y"})
	presentPlan(d, []planStep{
		{segment: "breaks on purpose", decl: failDecl, matched: true},
		{segment: "prints hello to the screen", decl: echoDecl, matched: true},
	})
	out.Reset()
	d.handle("run")
	got := out.String()
	for _, want := range []string{
		"something broke",
		"step 1 failed: exit status 3",
		"stopped: ran 0 of 2 steps — step 1 failed with exit status 3.",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("failure output missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "hello from the script") {
		t.Errorf("step after the failure ran:\n%s", got)
	}
	if g.asked != 1 {
		t.Errorf("gate asked %d times, want 1 (the failed step's)", g.asked)
	}
}

func TestRunRefusesNonExecutable(t *testing.T) {
	dir := t.TempDir()
	noexec := writeTestScript(t, dir, "noexec.sh", "#!/bin/sh\necho should never print\n", 0o644)
	decl := cmdmatch.Declaration{Name: "say-hello", Command: noexec, Description: "prints hello to the screen"}
	d, out, g := execTestDemo(t, []cmdmatch.Declaration{decl}, "", []string{"y"})
	presentPlan(d, []planStep{{segment: "prints hello to the screen", decl: decl, matched: true}})
	out.Reset()
	d.handle("run")
	got := out.String()
	if !strings.Contains(got, "cannot run step 1: "+noexec+" is not executable.") {
		t.Errorf("missing refusal:\n%s", got)
	}
	if !strings.Contains(got, "stopped: ran 0 of 1 step — step 1 cannot run.") {
		t.Errorf("missing summary:\n%s", got)
	}
	if strings.Contains(got, "should never print") {
		t.Errorf("non-executable script ran:\n%s", got)
	}
	if g.asked != 0 {
		t.Errorf("gate asked %d times for a refused step, want 0", g.asked)
	}
}

func TestRunRefusesMissingFile(t *testing.T) {
	dir := t.TempDir()
	ghost := filepath.Join(dir, "ghost.sh")
	decl := cmdmatch.Declaration{Name: "say-hello", Command: ghost, Description: "prints hello to the screen"}
	d, out, _ := execTestDemo(t, []cmdmatch.Declaration{decl}, "", []string{"y"})
	presentPlan(d, []planStep{{segment: "prints hello to the screen", decl: decl, matched: true}})
	out.Reset()
	d.handle("run")
	if !strings.Contains(out.String(), "cannot run step 1: "+ghost+" does not exist.") {
		t.Errorf("missing refusal:\n%s", out.String())
	}
}

func TestRunRefusedWhenNotInteractive(t *testing.T) {
	dir := t.TempDir()
	echo := writeTestScript(t, dir, "echo.sh", "#!/bin/sh\necho hello from the script\n", 0o755)
	decl := cmdmatch.Declaration{
		Name: "say-hello", Command: echo,
		Description: "prints hello to the screen",
		Examples:    []string{"prints hello to the screen"},
	}
	d, out, g := execTestDemo(t, []cmdmatch.Declaration{decl}, "", []string{"y"})
	d.interactive = false
	d.handle("prints hello to the screen")
	out.Reset()
	d.handle("run")
	got := out.String()
	if got != "execution needs the interactive prompt — nothing ran.\n" {
		t.Errorf("non-interactive run printed %q, want the one plain line", got)
	}
	if g.asked != 0 {
		t.Errorf("gate asked %d times in a non-interactive session", g.asked)
	}
}

func TestRunArgvCarriesSeatedValues(t *testing.T) {
	dir := t.TempDir()
	record := filepath.Join(dir, "argv.txt")
	rec := writeTestScript(t, dir, "rec.sh", "#!/bin/sh\nprintf '%s\\n' \"$@\" > "+record+"\n", 0o755)
	decl := cmdmatch.Declaration{
		Name: "move-thing", Command: rec,
		Description: "moves a thing from one place to another",
		Slots:       []string{"from", "to"},
		Examples:    []string{"move the thing from one place to another"},
	}
	d, _, _ := execTestDemo(t, []cmdmatch.Declaration{decl}, "jim jake", []string{"y"})
	d.handle("move the thing from jim to jake")
	d.handle("run")
	raw, err := os.ReadFile(record)
	if err != nil {
		t.Fatalf("recorder script did not run: %v", err)
	}
	if string(raw) != "jim\njake\n" {
		t.Errorf("argv = %q, want the seated values in order (jim, jake)", raw)
	}
}

func TestRunGapStopsPlan(t *testing.T) {
	dir := t.TempDir()
	echo := writeTestScript(t, dir, "echo.sh", "#!/bin/sh\necho hello from the script\n", 0o755)
	made := filepath.Join(dir, "made.txt")
	touch := writeTestScript(t, dir, "touch.sh", "#!/bin/sh\ntouch "+made+"\n", 0o755)
	echoDecl := cmdmatch.Declaration{Name: "say-hello", Command: echo, Description: "prints hello to the screen"}
	touchDecl := cmdmatch.Declaration{Name: "make-file", Command: touch, Description: "makes the file"}
	d, out, g := execTestDemo(t, []cmdmatch.Declaration{echoDecl, touchDecl}, "", []string{"y"})
	presentPlan(d, []planStep{
		{segment: "prints hello to the screen", decl: echoDecl, matched: true},
		{segment: "polish the trophies"},
		{segment: "makes the file", decl: touchDecl, matched: true},
	})
	out.Reset()
	d.handle("run")
	got := out.String()
	if !strings.Contains(got, "hello from the script") {
		t.Errorf("step before the gap did not run:\n%s", got)
	}
	if !strings.Contains(got, "step 2 has no command yet: 'polish the trophies'") {
		t.Errorf("gap line missing:\n%s", got)
	}
	if !strings.Contains(got, "stopped: ran 1 of 3 steps — step 2 is a gap.") {
		t.Errorf("summary missing or wrong:\n%s", got)
	}
	if _, err := os.Stat(made); !os.IsNotExist(err) {
		t.Errorf("step after the gap ran anyway: %s exists", made)
	}
	if g.asked != 1 {
		t.Errorf("gate asked %d times, want 1 (only the step before the gap)", g.asked)
	}
}

func TestRunRepeatsCautionForRemoveFamily(t *testing.T) {
	dir := t.TempDir()
	echo := writeTestScript(t, dir, "echo.sh", "#!/bin/sh\necho cleaned\n", 0o755)
	decl := cmdmatch.Declaration{
		Name: "delete-thing", Command: echo,
		Description: "deletes the thing from the disk",
		Examples:    []string{"deletes the thing from the disk"},
	}
	d, out, _ := execTestDemo(t, []cmdmatch.Declaration{decl}, "", []string{"y"})
	d.handle("deletes the thing from the disk")
	presentCount := strings.Count(out.String(), "careful: this command removes or changes things.")
	out.Reset()
	d.handle("run")
	got := out.String()
	if presentCount != 1 {
		t.Fatalf("presentation carried the caution %d times, want 1:\n%s", presentCount, got)
	}
	if !strings.Contains(got, "careful: this command removes or changes things.") {
		t.Errorf("gate did not repeat the caution:\n%s", got)
	}
	if !strings.Contains(got, "done: ran 1 of 1 step.") {
		t.Errorf("confirmed remove-family step did not run:\n%s", got)
	}
}

func TestRunOutputIsCapped(t *testing.T) {
	dir := t.TempDir()
	loud := writeTestScript(t, dir, "loud.sh", "#!/bin/sh\ni=0\nwhile [ $i -lt 5000 ]; do echo xxxxxxxxxx; i=$((i+1)); done\n", 0o755)
	decl := cmdmatch.Declaration{Name: "be-loud", Command: loud, Description: "prints a great deal"}
	d, out, _ := execTestDemo(t, []cmdmatch.Declaration{decl}, "", []string{"y"})
	presentPlan(d, []planStep{{segment: "prints a great deal", decl: decl, matched: true}})
	out.Reset()
	d.handle("run")
	got := out.String()
	if !strings.Contains(got, "(output truncated — ") {
		t.Errorf("loud output was not reported truncated (len %d)", len(got))
	}
	if len(got) > maxStepOutput+4096 {
		t.Errorf("printed output not bounded: %d bytes", len(got))
	}
}

func TestRunAfterExtractorFailureIsSilent(t *testing.T) {
	dir := t.TempDir()
	echo := writeTestScript(t, dir, "echo.sh", "#!/bin/sh\necho hello from the script\n", 0o755)
	decl := cmdmatch.Declaration{
		Name: "move-thing", Command: echo,
		Description: "moves a thing from one place to another",
		Slots:       []string{"from", "to"},
		Examples:    []string{"move the thing from one place to another"},
	}
	d, out, g := execTestDemo(t, []cmdmatch.Declaration{decl}, "", []string{"y"})
	d.extract = func(string) (string, error) { return "", errStub }
	d.handle("move the thing from jim to jake")
	if !strings.Contains(out.String(), "(arguments missing)") {
		t.Fatalf("expected the extractor-failure presentation:\n%s", out.String())
	}
	out.Reset()
	d.handle("run")
	if out.String() != "" {
		t.Errorf("run after an arguments-missing presentation printed %q, want silence", out.String())
	}
	if g.asked != 0 {
		t.Errorf("gate asked %d times, want 0", g.asked)
	}
}

var errStub = &stubError{}

type stubError struct{}

func (*stubError) Error() string { return "stub extractor failure" }

// TestRoutineInvocationIsNotTheTrigger pins live session
// 5 at the unit level: a line NAMING a routine is
// answered with its presentation, never with the
// executor — only the bare words run / run it trigger.
func TestRoutineInvocationIsNotTheTrigger(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := queueDemo(t, path, reg, nil)
	d.interactive = true
	d.handle("run the full backup routine")
	got := out.String()
	if !strings.Contains(got, "routine: full-backup") {
		t.Errorf("routine was not presented:\n%s", got)
	}
	if strings.Contains(got, "ran 0 of") || strings.Contains(got, "done: ran") || strings.Contains(got, "execution needs") {
		t.Errorf("the executor fired on a routine invocation line:\n%s", got)
	}
}
