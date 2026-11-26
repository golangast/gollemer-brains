package main

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Plans: several wants in one line, laid out as an
// ordered checklist. A single want is answered by the
// matcher alone; a line that splits into two or more
// segments — "back up the config, update the script,
// and run the checks" — is a small JOB, and answering it
// with only the single best-matching script would drop
// the rest of the job on the floor. So a multi-segment
// line where at least one segment matches a command is
// treated as a plan: each segment is matched on its own,
// the steps are ordered by their declared stages and
// prerequisites rather than by typing order, the
// segments no command knows are shown as gaps (with one
// offer to teach the first of them), and the result is a
// numbered checklist whose every step carries the same
// preview the single-line path prints — the gate any
// future execution step would confirm against, step by
// step. A plan is printed, never run: execution stays
// the separate, gated step it has always been here.

// splitSegments splits a line at the punctuation-level
// joins a multi-step request is typed with: commas,
// "and", "then", "and then". The pieces are trimmed of
// spaces and a trailing full stop; empty pieces (a
// leading "then", a trailing comma) are dropped. The
// split is deliberately dumb — it knows nothing about
// the words — because the matcher, not the splitter,
// decides what each piece means.
var rePlanSplit = regexp.MustCompile(`(?i)\s*(?:,|\band\s+then\b|\bthen\b|\band\b)\s*`)

func splitSegments(line string) []string {
	var out []string
	for _, piece := range rePlanSplit.Split(line, -1) {
		piece = strings.Trim(strings.TrimSpace(piece), ".")
		if piece = strings.TrimSpace(piece); piece != "" {
			out = append(out, piece)
		}
	}
	return out
}

// planStep is one segment of a plan: the text as typed,
// and — when a command matched it — the command, the
// score, and the argument values the extractor heard in
// the segment. A step with matched false is a GAP: no
// command knows that step yet.
type planStep struct {
	segment string
	decl    cmdmatch.Declaration
	score   float64
	values  []string
	matched bool
}

// stageRank orders the stages a plan is sorted by:
// inspect, then prepare, then execute, then verify,
// then cleanup. An unstaged command — and a gap, which
// has no command at all — ranks as execute: the middle
// of the job, and (by the sort's stability) near where
// it was typed relative to its same-rank neighbours. An
// unrecognised stage word ranks the same: the tag is
// the user's to spell, and a rough order is the point.
func stageRank(stage string) int {
	switch strings.ToLower(strings.TrimSpace(stage)) {
	case "inspect":
		return 0
	case "prepare":
		return 1
	case "verify":
		return 3
	case "cleanup":
		return 4
	default: // execute, unstaged, unknown
		return 2
	}
}

// buildPlan matches each segment of a line on its own.
// It reports false when the line is not a plan: fewer
// than two segments, or no segment matching any command
// — in which case the caller falls back to the ordinary
// single-line handling, whole line and all.
func (d *demo) buildPlan(line string) ([]planStep, bool) {
	segments := splitSegments(line)
	if len(segments) < 2 {
		return nil, false
	}
	steps := make([]planStep, 0, len(segments))
	matched := 0
	for _, seg := range segments {
		step := planStep{segment: seg}
		if decl, score, ok := d.matcher.Match(seg); ok {
			step.decl, step.score, step.matched = decl, score, true
			matched++
			if len(decl.Slots) > 0 {
				if raw, err := d.extract(seg); err == nil {
					step.values = strings.Fields(raw)
				}
			}
		}
		steps = append(steps, step)
	}
	if matched == 0 {
		return nil, false
	}
	return steps, true
}

// orderSteps sorts steps into the order they would run:
// by stage rank first (stable, so equal ranks keep
// their typed order), then by a stable topological pass
// honouring the After edges among the plan's own steps —
// a step whose prerequisites are in the plan comes
// after them, even across stage ranks and even when it
// was typed first. An After naming a step that is not in
// the plan constrains nothing here (it is reported as a
// missing link by planNotes). If the After edges among
// the remaining steps form a cycle, nothing more can be
// freed: the rest are emitted in their stage order and
// cycle is reported true — a plan is printed, never
// looped on.
func orderSteps(steps []planStep) (ordered []planStep, cycle bool) {
	sorted := append([]planStep(nil), steps...)
	sort.SliceStable(sorted, func(i, j int) bool {
		return stageRank(stepStage(sorted[i])) < stageRank(stepStage(sorted[j]))
	})
	present := map[string]bool{}
	for _, s := range sorted {
		if s.matched {
			present[s.decl.Name] = true
		}
	}
	emitted := map[string]bool{}
	remaining := sorted
	for len(remaining) > 0 {
		freed := false
		for i, s := range remaining {
			if !afterSatisfied(s, present, emitted) {
				continue
			}
			ordered = append(ordered, s)
			if s.matched {
				emitted[s.decl.Name] = true
			}
			remaining = append(remaining[:i], remaining[i+1:]...)
			freed = true
			break // restart: always free the earliest available step
		}
		if !freed {
			ordered = append(ordered, remaining...)
			return ordered, true
		}
	}
	return ordered, false
}

// stepStage is the stage a step sorts by: its command's
// declared stage, or "" for a gap.
func stepStage(s planStep) string {
	if !s.matched {
		return ""
	}
	return s.decl.Stage
}

// afterSatisfied reports whether every After prerequisite
// of a step that is present in the plan has already been
// emitted. Gaps have no prerequisites.
func afterSatisfied(s planStep, present, emitted map[string]bool) bool {
	if !s.matched {
		return true
	}
	for _, before := range s.decl.After {
		if present[before] && !emitted[before] {
			return false
		}
	}
	return true
}

// planNotes collects the missing-link notes a plan
// earns, in step order: an After prerequisite no step in
// the plan provides, the verify-with-no-execute gap, and
// (when orderSteps found one) the cycle note.
func planNotes(steps []planStep, cycle bool) []string {
	var notes []string
	present := map[string]bool{}
	hasVerify, hasExecute := false, false
	for _, s := range steps {
		if !s.matched {
			continue
		}
		present[s.decl.Name] = true
		if strings.EqualFold(strings.TrimSpace(s.decl.Stage), "verify") {
			hasVerify = true
		}
		if stageRank(s.decl.Stage) == stageRank("execute") {
			hasExecute = true
		}
	}
	for _, s := range steps {
		if !s.matched {
			continue
		}
		for _, before := range s.decl.After {
			if !present[before] {
				notes = append(notes, fmt.Sprintf("note: %s is declared after %s, but no step in this plan is %s.", s.decl.Name, before, before))
			}
		}
	}
	if hasVerify && !hasExecute {
		notes = append(notes, "verify with no execute step — nothing to check the result of.")
	}
	if cycle {
		notes = append(notes, "note: the after-links between these steps form a cycle — shown in stage order.")
	}
	return notes
}

// handlePlan prints a line's plan and then makes the
// one teaching offer a gapped plan earns: teach the
// FIRST gap segment, through the same learnWant core
// the single-line teach offer uses. When the offer is
// taken, the plan is rebuilt and printed again —
// complete, if the teaching landed. Enter skips, and
// the plan stands as printed.
func (d *demo) handlePlan(line string) {
	steps, _ := d.buildPlan(line)
	ordered, cycle := orderSteps(steps)
	d.printPlan(ordered, planNotes(ordered, cycle))

	var firstGap *planStep
	for i := range ordered {
		if !ordered[i].matched {
			firstGap = &ordered[i]
			break
		}
	}
	if firstGap == nil {
		return
	}
	before := len(d.reg.Scripts)
	d.learnWant(firstGap.segment)
	if len(d.reg.Scripts) > before {
		if steps, ok := d.buildPlan(line); ok {
			ordered, cycle := orderSteps(steps)
			d.printPlan(ordered, planNotes(ordered, cycle))
		}
	}
}

// printPlan renders the checklist: a header counting
// steps, ready steps, and gaps; one numbered entry per
// step — a matched step with its command's name, its
// stage and acts-on hints when declared, its preview
// line (the description with the segment's values
// seated, exactly as the single-line path renders it)
// and its runs line; a gap step with the segment nobody
// knows; then the missing-link notes; and the standing
// reminder that nothing has run.
func (d *demo) printPlan(steps []planStep, notes []string) {
	ready, gaps := 0, 0
	for _, s := range steps {
		if s.matched {
			ready++
		} else {
			gaps++
		}
	}
	fmt.Fprintf(d.out, "plan: %d %s (%d ready, %d %s)\n", len(steps), plural(len(steps), "step"), ready, gaps, plural(gaps, "gap"))
	for i, s := range steps {
		if !s.matched {
			fmt.Fprintf(d.out, "%d. ? — no command knows this step yet: '%s'\n", i+1, s.segment)
			continue
		}
		fmt.Fprintf(d.out, "%d. %s%s — %s\n", i+1, s.decl.Name, stepMeta(s.decl), previewText(s.decl, s.values))
		fmt.Fprintf(d.out, "   runs: %s\n", wouldRun(s.decl, s.values))
	}
	for _, n := range notes {
		fmt.Fprintln(d.out, n)
	}
	fmt.Fprintln(d.out, "nothing has run yet.")
}

// stepMeta renders a step's declared hints — its stage,
// and what it acts on — as " [execute · on logs]", or ""
// when the command declares neither.
func stepMeta(decl cmdmatch.Declaration) string {
	var parts []string
	if decl.Stage != "" {
		parts = append(parts, decl.Stage)
	}
	if decl.ActsOn != "" {
		parts = append(parts, "on "+decl.ActsOn)
	}
	if len(parts) == 0 {
		return ""
	}
	return " [" + strings.Join(parts, " · ") + "]"
}

// plural returns the word, or the word plus "s" — the
// header's small grammar.
func plural(n int, word string) string {
	if n == 1 {
		return word
	}
	return word + "s"
}
