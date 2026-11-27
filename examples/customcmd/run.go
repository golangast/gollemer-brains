package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Gated execution: the one place this program runs
// anything, and the gate is the whole design. A command
// or plan the REPL has just PRESENTED — the understood
// block, or the plan checklist — can be run by typing
// the bare line `run` (or `run it`) as the very next
// line, the same adjacency discipline the corrections
// use: the trigger refers to the presentation before it
// and to nothing else, and any other line consumes the
// presentation untouched. "run the nightly routine" is
// not the trigger — it names a routine and is answered
// with its presentation, as always.
//
// What can run is only what was presented, and a
// presentation is built only from registry entries: the
// registry is the allowlist, and there is no path from
// a typed line to an executed file that does not pass
// through a presentation and this gate. Each step is
// then confirmed on its own — the exact would-run line
// is shown again and only a `y` (or `yes`) runs it;
// anything else stops the whole run at that step, later
// steps un-offered, because a plan's later steps may
// depend on its earlier ones. Before its gate, each
// step's file must exist and be executable; a file that
// fails that check is refused in plain words and stops
// the run there. A step that runs and fails (nonzero
// exit) stops the run the same way. A gap step and a
// per-file step cannot run at all — a gap has no
// command, and a per-file step names a set, not one
// command — and reaching one stops the run there too.
//
// There is NO SHELL anywhere in this: a step runs by
// exec.Command(file, args...) with argv split from the
// presented would-run line — the file the registry
// declared, then the seated slot values in order. The
// child's stdin is the null device, so a script can
// never eat the REPL's own input lines. And execution
// needs the interactive prompt: with stdin not a TTY
// the trigger prints one plain line and nothing runs,
// so pipes and tests can never execute anything.

// maxStepOutput bounds how much of a step's combined
// output is kept and printed; the rest is counted and
// reported as not shown.
const maxStepOutput = 32 * 1024

// execStep is one step of a runnable presentation: the
// declaration it came from, the would-run line exactly
// as presented (the gate shows it again), and the argv
// split from that line — file first, then the seated
// values in order. A gap step has no command at all; an
// eachFile step names a set of files rather than one
// command. Neither can run.
type execStep struct {
	decl     cmdmatch.Declaration
	line     string
	argv     []string
	remove   bool // remove-family: the caution repeats at the gate
	eachFile bool
	gap      bool
	segment  string // a gap step's segment text
}

// runnable is a presentation the trigger word can act
// on: the steps it showed, in the order it showed them.
// handle consumes it on the very next line, run or not —
// only the NEXT line can run a presentation.
type runnable struct {
	steps []execStep
}

// isRunTrigger reports whether a line is the execution
// trigger: exactly "run" or "run it", case-folded. The
// set is small and exact by the same reasoning as the
// correction triggers — anything broader would eat
// ordinary wants ("run the nightly routine" above all).
func isRunTrigger(line string) bool {
	l := strings.ToLower(strings.TrimSpace(line))
	return l == "run" || l == "run it"
}

// removeFamily reports whether a declaration is a
// remove-family command — the same test the preview's
// caution line uses.
func removeFamily(decl cmdmatch.Declaration) bool {
	return cmdmatch.HasRemoveWord(decl.Name) || cmdmatch.HasRemoveWord(decl.Description)
}

// singleRunnable builds the one-step runnable a single
// presented command leaves behind.
func singleRunnable(decl cmdmatch.Declaration, line string) *runnable {
	return &runnable{steps: []execStep{{
		decl:   decl,
		line:   line,
		argv:   strings.Fields(line),
		remove: removeFamily(decl),
	}}}
}

// planRunnable builds the runnable a printed plan (or
// routine expansion) leaves behind: its steps in the
// order they were printed, gaps included — a run that
// reaches a gap stops there.
func planRunnable(steps []planStep) *runnable {
	r := &runnable{}
	for _, s := range steps {
		if !s.matched {
			r.steps = append(r.steps, execStep{gap: true, segment: s.segment})
			continue
		}
		decl, vals := s.stepRender()
		r.steps = append(r.steps, execStep{
			decl:     s.decl,
			line:     s.wouldRunText(),
			argv:     strings.Fields(wouldRun(decl, vals)),
			remove:   removeFamily(s.decl),
			eachFile: s.eachFile,
		})
	}
	return r
}

// isYes reads a gate answer: only "y" or "yes",
// case-folded, confirms. Everything else — enter, EOF,
// "n", a mistyped word — declines.
func isYes(ans string) bool {
	a := strings.ToLower(strings.TrimSpace(ans))
	return a == "y" || a == "yes"
}

// checkExecutable is the allowlist's file check, run
// before a step's gate: the step's file must exist and
// be executable. A file named with a path is statted
// directly; a bare name must resolve on the path. It
// returns "" when the file may run, or the plain reason
// it may not.
func checkExecutable(file string) string {
	if file == "" {
		return "the step has no command"
	}
	if strings.Contains(file, "/") {
		fi, err := os.Stat(file)
		if err != nil {
			return file + " does not exist"
		}
		if fi.IsDir() {
			return file + " is a folder, not a script"
		}
		if fi.Mode()&0o111 == 0 {
			return file + " is not executable"
		}
		return ""
	}
	if _, err := exec.LookPath(file); err != nil {
		return file + " was not found on the path"
	}
	return ""
}

// capBuffer is an output sink that keeps at most limit
// bytes and counts everything written to it — a step
// that floods the terminal is reported as truncated,
// not obeyed.
type capBuffer struct {
	buf   bytes.Buffer
	limit int
	total int
}

func (c *capBuffer) Write(p []byte) (int, error) {
	n := len(p)
	c.total += n
	if room := c.limit - c.buf.Len(); room > 0 {
		if len(p) > room {
			p = p[:room]
		}
		c.buf.Write(p)
	}
	return n, nil
}

// runArgv runs one step's argv directly — no shell, no
// command string — and returns its combined stdout and
// stderr (capped, with a truncation note when the cap
// cut) and the run's error, if any. The child's stdin
// is left nil: the null device, never the REPL's input.
func runArgv(argv []string) (string, error) {
	cmd := exec.Command(argv[0], argv[1:]...)
	buf := &capBuffer{limit: maxStepOutput}
	cmd.Stdout = buf
	cmd.Stderr = buf
	err := cmd.Run()
	out := buf.buf.String()
	if buf.total > buf.limit {
		out += fmt.Sprintf("(output truncated — %d bytes not shown)\n", buf.total-buf.limit)
	}
	return out, err
}

// runFlow runs a presented command or plan behind the
// gate, step by step, and ends with a one-line summary:
// how many steps ran, and — when the run stopped early —
// where and why. Reaching the end of the steps is the
// only way to the "done" line.
func (d *demo) runFlow(r *runnable) {
	if !d.interactive || d.nextLine == nil {
		fmt.Fprintln(d.out, "execution needs the interactive prompt — nothing ran.")
		return
	}
	total := len(r.steps)
	ran := 0
	stop := func(reason string) {
		fmt.Fprintf(d.out, "stopped: ran %d of %d %s — %s.\n", ran, total, plural(total, "step"), reason)
	}
	for i, step := range r.steps {
		n := i + 1
		if step.gap {
			fmt.Fprintf(d.out, "step %d has no command yet: '%s'\n", n, step.segment)
			stop(fmt.Sprintf("step %d is a gap", n))
			return
		}
		if step.eachFile {
			fmt.Fprintf(d.out, "step %d (%s) applies to each file — running a per-file step is not supported.\n", n, step.line)
			stop(fmt.Sprintf("step %d applies to each file", n))
			return
		}
		if len(step.argv) == 0 {
			fmt.Fprintf(d.out, "cannot run step %d: the step has no command.\n", n)
			stop(fmt.Sprintf("step %d cannot run", n))
			return
		}
		if problem := checkExecutable(step.argv[0]); problem != "" {
			fmt.Fprintf(d.out, "cannot run step %d: %s.\n", n, problem)
			stop(fmt.Sprintf("step %d cannot run", n))
			return
		}
		fmt.Fprintf(d.out, "step %d: %s\n", n, step.line)
		if step.remove {
			fmt.Fprintln(d.out, d.paint(ansiWarn, "  careful: this command removes or changes things."))
		}
		ans, ok := d.nextLine("run this step? (y/n) ")
		if !ok || !isYes(ans) {
			stop(fmt.Sprintf("step %d was not confirmed", n))
			return
		}
		out, err := runArgv(step.argv)
		if out != "" {
			fmt.Fprint(d.out, out)
			if !strings.HasSuffix(out, "\n") {
				fmt.Fprintln(d.out)
			}
		}
		if err != nil {
			if exitErr, isExit := err.(*exec.ExitError); isExit {
				fmt.Fprintf(d.out, "step %d failed: %v\n", n, exitErr)
				stop(fmt.Sprintf("step %d failed with %v", n, exitErr))
			} else {
				fmt.Fprintf(d.out, "step %d could not start: %v\n", n, err)
				stop(fmt.Sprintf("step %d could not start", n))
			}
			return
		}
		ran++
	}
	fmt.Fprintf(d.out, "done: ran %d of %d %s.\n", ran, total, plural(total, "step"))
}
