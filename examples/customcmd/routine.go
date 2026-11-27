package main

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Routines: named bundles of steps, declared once and
// invoked by description like any script — except that
// where a script's match renders its one command, a
// routine's match EXPANDS: each of its segment texts is
// matched in turn, and the result is rendered through
// the plan machinery, headered "routine: <name>". A
// routine is a macro over concepts the registry already
// knows, never a new capability: every segment must
// resolve to a known command (the declaration door
// refuses one that does not, naming the segment), and a
// segment that is itself a routine expands one more
// level — depth two — beyond which the step renders as
// a gap carrying a "routine too deep" note rather than
// expanding forever.
//
// Steps are bound the way plan steps are: the extractor
// reads each segment, and a segment that yields no
// values leaves its slots to the ORIGINAL line — the
// invocation is where the job's arguments were actually
// spoken ("run the full backup from jim to jake": the
// migration step's from and to live in that line, not in
// its segment). Ordered, context-shared, and rendered by
// the same code as a typed plan, footer included: a
// routine is printed, never run.

// routineDepthCap is how deep routine expansion goes:
// the invoked routine is level one, a routine among its
// steps is level two, and a third level is a gap.
const routineDepthCap = 2

// handleRoutine answers a want that matched a routine:
// expand its steps, order and context-share them as a
// plan, and render under the routine's name.
func (d *demo) handleRoutine(decl cmdmatch.Declaration, line string) {
	steps := d.expandRoutineSteps(decl.Steps, line, 1)
	ordered, cycle := orderSteps(steps)
	shareContext(ordered)
	fmt.Fprintf(d.out, "%s %s\n", d.paint(ansiLabel, "routine:"), decl.Name)
	d.printPlan(ordered, planNotes(ordered, cycle))
}

// matchSegment resolves one routine segment to the entry
// it names, in two phases: COMMANDS first, then
// ROUTINES. The phases are not a preference, they are a
// necessity: a declared routine's description is its own
// doing-clause — the concatenation of its segments — so
// its matcher profile contains every segment's words,
// and in one mixed matcher a bare segment ties between
// its command and any routine whose clause contains it,
// abstains, and lands as a gap in the very routine it
// belongs to. (Measured live: with a "nightly" routine
// declared, the shipped full-backup's steps all gapped.)
// Separated, a segment that names a command is a command
// step, full stop; only a segment no command claims can
// name a routine. The declaration door resolves its
// segments the same way.
func (d *demo) matchSegment(seg string) (cmdmatch.Declaration, float64, bool) {
	commands, routines := d.segmentMatchers()
	if decl, score, ok := commands.Match(seg); ok {
		return decl, score, true
	}
	return routines.Match(seg)
}

// segmentMatchers builds the two phase matchers of
// matchSegment: the registry's commands in one, its
// routines in the other. Registries are small and
// expansion is shallow, so building them per call is
// cheap — and always current against the registry the
// REPL holds.
func (d *demo) segmentMatchers() (commands, routines *cmdmatch.Matcher) {
	var cmds, routs []cmdmatch.Declaration
	for _, s := range d.reg.Scripts {
		if len(s.Steps) > 0 {
			routs = append(routs, s)
		} else {
			cmds = append(cmds, s)
		}
	}
	return cmdmatch.NewMatcher(&cmdmatch.Registry{Scripts: cmds}),
		cmdmatch.NewMatcher(&cmdmatch.Registry{Scripts: routs})
}

// expandRoutineSteps turns a routine's segment texts
// into plan steps, depth-first: a segment naming an
// ordinary command becomes its step (values read from
// the segment, or from the invocation line when the
// segment offers none); a segment naming a routine
// expands in place while depth allows; a segment naming
// nothing is a gap, as in any plan.
func (d *demo) expandRoutineSteps(segments []string, line string, depth int) []planStep {
	var steps []planStep
	for _, seg := range segments {
		decl, score, ok := d.matchSegment(seg)
		if !ok {
			steps = append(steps, planStep{segment: seg})
			continue
		}
		if len(decl.Steps) > 0 {
			if depth >= routineDepthCap {
				steps = append(steps, planStep{
					segment: seg,
					note:    fmt.Sprintf("note: routine too deep — '%s' is a routine, and routines expand at most %d levels.", seg, routineDepthCap),
				})
				continue
			}
			steps = append(steps, d.expandRoutineSteps(decl.Steps, line, depth+1)...)
			continue
		}
		step := planStep{segment: seg, decl: decl, score: score, matched: true}
		if len(decl.Slots) > 0 {
			if raw, err := d.extract(seg); err == nil {
				step.values = strings.Fields(raw)
			}
			if len(step.values) == 0 && line != seg {
				if raw, err := d.extract(line); err == nil {
					step.values = strings.Fields(raw)
				}
			}
		}
		steps = append(steps, step)
	}
	return steps
}
