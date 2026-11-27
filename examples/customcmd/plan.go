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
// checklist whose every step carries the same preview
// the single-line path prints — the gate any future
// execution step would confirm against, step by step.
// A plan is printed, never run: execution stays the
// separate, gated step it has always been here.
//
// Steps in a plan also SHARE CONTEXT, the way the steps
// of a job do: a step that is missing a value inherits
// the value an earlier step bound to the same slot name
// ("sort the files in trips … and rename the photos in
// the folder" — the rename step's folder is trips,
// inherited and marked "(from step 1)"), and a step
// whose segment speaks of files in the plural, in a
// plan that has a folder in play, applies to EACH file
// of that folder rather than to one file it never named.

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
//
// The remaining fields are filled in by shareContext
// once the steps are ordered: eff holds the step's
// effective slot values (its own, plus any inherited
// from earlier steps), fromStep records which step an
// inherited value came from, grounded records whether a
// value was actually named in the segment that first
// stated it (as opposed to offered by the extractor for
// a segment that named none), and eachFile/eachFolder
// carry the per-file reading of the step.
type planStep struct {
	segment string
	decl    cmdmatch.Declaration
	score   float64
	values  []string
	matched bool

	eff          map[string]string
	fromStep     map[string]int
	grounded     map[string]bool
	eachFile     bool
	eachFolder   string
	eachFolderIn int // step the each-file folder came from; 0 = this step or none

	note string // a note the step itself carries into planNotes (routine expansion)
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
			if len(decl.Steps) > 0 {
				// A segment that names a routine expands
				// into the routine's own steps, in place —
				// the same expansion a single-line match
				// gets (routine.go), one level from here.
				for _, sub := range d.expandRoutineSteps(decl.Steps, seg, 1) {
					if sub.matched {
						matched++
					}
					steps = append(steps, sub)
				}
				continue
			}
			step.decl, step.score, step.matched = decl, score, true
			matched++
			if len(decl.Slots) > 0 {
				if raw, err := d.extract(seg); err == nil {
					step.values = bindSlotValues(decl, seg, strings.Fields(raw))
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
// the plan provides, a Needs relation (Piece 1 of the
// concept-graph build) the plan does not satisfy — the
// needed concept absent, or present but later in the
// order — the verify-with-no-execute gap, and (when
// orderSteps found one) the cycle note. A Needs entry
// that an After edge on the same step already encodes
// earns no note: the After machinery owns that relation
// (it orders by it, and reports its absence above).
func planNotes(steps []planStep, cycle bool) []string {
	var notes []string
	present := map[string]bool{}
	position := map[string]int{}
	hasVerify, hasExecute := false, false
	for i, s := range steps {
		if !s.matched {
			continue
		}
		present[s.decl.Name] = true
		position[s.decl.Name] = i
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
		for _, need := range s.decl.Needs {
			if containsString(s.decl.After, need) {
				continue
			}
			if !present[need] {
				notes = append(notes, fmt.Sprintf("note: %s usually needs %s first — it is not in this plan.", s.decl.Name, need))
			} else if position[need] > position[s.decl.Name] {
				notes = append(notes, fmt.Sprintf("note: %s usually needs %s first — it comes later in this plan.", s.decl.Name, need))
			}
		}
	}
	for _, s := range steps {
		if s.note != "" {
			notes = append(notes, s.note)
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
	shareContext(ordered)
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
			shareContext(ordered)
			d.printPlan(ordered, planNotes(ordered, cycle))
		}
	}
}

// shareContext works out what the ordered steps share,
// in two passes over the steps in the order they would
// run.
//
// First, INHERITANCE: a step missing a value for one of
// its slots inherits the value an earlier step bound to
// the SAME slot name. One exception, the place family:
// the slots that name a place (from, to, folder,
// directory, dir) are one kind of thing, so a place slot
// no earlier step holds any value for may take the
// nearest earlier grounded place value across names —
// a folder is not a file, but a folder IS a directory.
// The earliest step holding a same-name value is its
// source, and the step remembers it (fromStep) so the
// value can be marked "(from step N)" wherever it is
// shown. A value a step
// inherited can itself be inherited onward; the mark
// still names the step that first bound it. And a value
// the extractor merely OFFERED a step — a favourite its
// segment never named — loses to a grounded value from
// an earlier step: what the user actually said in step
// 1 outranks what the extractor guessed in step 2.
//
// Second, the EACH-FILE reading: a step with a file
// slot its own segment left unnamed applies to each
// file when its segment speaks of files in the plural
// ("the files", "each file", "all files", "every file")
// or when the plan's context supplies a folder — an
// earlier step's folder, or its own. The folder it
// applies in is that folder. A step whose segment named
// one concrete file is NOT each-file: a single named
// file is a single target, and defeats the reading.
//
// "Named" is meant literally, and it matters: the
// extractor may offer a favourite value for a segment
// that stated none ("delete the first line in the
// files" can come back with a file the user never
// said). Such a value is not a named target, and does
// not defeat the each-file reading; likewise only a
// folder some segment actually named counts as context
// for the per-file set. A value is grounded when it
// appears in the segment that first stated it, and
// groundedness travels with the value when it is
// inherited.
func shareContext(steps []planStep) {
	for i := range steps {
		s := &steps[i]
		if !s.matched {
			continue
		}
		s.eff = map[string]string{}
		s.fromStep = map[string]int{}
		s.grounded = map[string]bool{}
		for j, slot := range s.decl.Slots {
			own := ""
			if j < len(s.values) {
				own = s.values[j]
			}
			if own != "" && segmentMentions(s.segment, own) {
				s.eff[slot] = own // stated by this very segment: grounded
				s.grounded[slot] = true
				continue
			}
			// The step stated nothing (or only an unnamed
			// favourite): a grounded value from an earlier
			// step wins over the favourite.
			filled := false
			for k := 0; k < i && !filled; k++ {
				if !steps[k].matched {
					continue
				}
				if v, ok := steps[k].eff[slot]; ok && steps[k].grounded[slot] {
					s.eff[slot] = v
					s.grounded[slot] = true
					if n := steps[k].fromStep[slot]; n > 0 {
						s.fromStep[slot] = n
					} else {
						s.fromStep[slot] = k + 1
					}
					filled = true
				}
			}
			if filled {
				continue
			}
			// Place-family inheritance: the place slots
			// {from, to, folder, directory, dir} are one
			// family. A place slot that no earlier step
			// holds a value for — under its own name or
			// any other — may take the NEAREST earlier
			// grounded place value, bound under whatever
			// name that step used, marked "(from step N)"
			// like any inheritance. Same-name still wins
			// (both passes above run first, and the guard
			// below defers to any same-name value at all);
			// non-place slots never cross-inherit; and
			// only grounded values cross — a favourite an
			// extractor offered an earlier step stays home.
			if placeSlot(slot) && !earlierHasSlot(steps[:i], slot) {
				if v, from, ok := nearestPlaceValue(steps[:i]); ok {
					s.eff[slot] = v
					s.grounded[slot] = true
					s.fromStep[slot] = from
					continue
				}
			}
			if own != "" {
				s.eff[slot] = own // the favourite stands: nothing better exists
				continue
			}
			for k := 0; k < i; k++ {
				if !steps[k].matched {
					continue
				}
				if v, ok := steps[k].eff[slot]; ok {
					s.eff[slot] = v
					s.grounded[slot] = steps[k].grounded[slot]
					if n := steps[k].fromStep[slot]; n > 0 {
						s.fromStep[slot] = n
					} else {
						s.fromStep[slot] = k + 1
					}
					break
				}
			}
		}
	}
	for i := range steps {
		s := &steps[i]
		if !s.matched {
			continue
		}
		fileIdx := -1
		for j, slot := range s.decl.Slots {
			if slot == "file" {
				fileIdx = j
			}
		}
		if fileIdx < 0 {
			continue
		}
		ownNamed := fileIdx < len(s.values) && s.values[fileIdx] != "" &&
			segmentMentions(s.segment, s.values[fileIdx])
		if ownNamed || s.fromStep["file"] > 0 {
			continue // one concrete file is in play: a single target
		}
		folder, folderIn := "", 0
		if v, ok := s.eff["folder"]; ok && s.grounded["folder"] {
			folder = v
			if n := s.fromStep["folder"]; n > 0 {
				folderIn = n
			} else {
				folderIn = i + 1
			}
		} else {
			for k := 0; k < i; k++ {
				if !steps[k].matched {
					continue
				}
				if v, ok := steps[k].eff["folder"]; ok && steps[k].grounded["folder"] {
					folder, folderIn = v, k+1
					if n := steps[k].fromStep["folder"]; n > 0 {
						folderIn = n
					}
					break
				}
			}
		}
		if rePluralFiles.MatchString(s.segment) || folder != "" {
			s.eachFile = true
			s.eachFolder = folder
			if folder != "" && folderIn != i+1 {
				s.eachFolderIn = folderIn
			}
		}
	}
}

// placeSlots are the slot names that name a PLACE — one
// family for plan-context inheritance (see shareContext).
// A folder by any of these names is the same kind of
// thing: where the work happens.
var placeSlots = map[string]bool{
	"from": true, "to": true, "folder": true,
	"directory": true, "dir": true,
}

// placeSlot reports whether a slot name belongs to the
// place family.
func placeSlot(slot string) bool { return placeSlots[slot] }

// earlierHasSlot reports whether any earlier step holds
// an effective value under the given slot name, grounded
// or not — the guard that keeps place-family inheritance
// from ever overriding a same-name value.
func earlierHasSlot(earlier []planStep, slot string) bool {
	for _, s := range earlier {
		if !s.matched {
			continue
		}
		if _, ok := s.eff[slot]; ok {
			return true
		}
	}
	return false
}

// nearestPlaceValue finds the nearest earlier step's
// grounded place value: steps are searched nearest
// first, a step's own place slots in their declared
// order. It returns the value and the 1-based number of
// the step that first bound it (provenance travels, as
// with same-name inheritance).
func nearestPlaceValue(earlier []planStep) (string, int, bool) {
	for k := len(earlier) - 1; k >= 0; k-- {
		s := earlier[k]
		if !s.matched {
			continue
		}
		for _, slot := range s.decl.Slots {
			if !placeSlot(slot) {
				continue
			}
			if v, ok := s.eff[slot]; ok && s.grounded[slot] {
				if n := s.fromStep[slot]; n > 0 {
					return v, n, true
				}
				return v, k + 1, true
			}
		}
	}
	return "", 0, false
}

// segmentMentions reports whether a value appears, as a
// whole word, in a segment — the test of whether the
// segment actually NAMED the value, as opposed to the
// extractor offering it for a segment that stated none.
func segmentMentions(segment, value string) bool {
	if value == "" {
		return false
	}
	return regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(value) + `\b`).MatchString(segment)
}

// rePluralFiles is the plural-file marker a segment can
// carry: files spoken of as a set, not one named file.
var rePluralFiles = regexp.MustCompile(`(?i)\b(?:the|all)\s+files\b|\b(?:each|every)\s+files?\b`)

// posValues renders a step's effective values in slot
// order, for previewText and wouldRun: each slot's own
// or inherited value, positionally. Only positions up to
// the last valued slot are returned (a wholly unvalued
// step keeps the extractor's own answer), and a slot
// with no value before that point pads as "" — which
// previewText reads as not heard.
func (s planStep) posValues() []string {
	last := -1
	for j, slot := range s.decl.Slots {
		if _, ok := s.eff[slot]; ok {
			last = j
		}
	}
	if last < 0 {
		return s.values
	}
	out := make([]string, last+1)
	for j := 0; j <= last; j++ {
		out[j] = s.eff[s.decl.Slots[j]]
	}
	return out
}

// stageLabel renders a stage for the checklist: the
// declared stage, capitalized — and "Execute" for an
// unstaged command, which is where an untagged step
// sorts anyway.
func stageLabel(stage string) string {
	s := strings.ToLower(strings.TrimSpace(stage))
	if s == "" {
		return "Execute"
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// provMark is the "(from step N)" marker for a slot
// whose value the step inherited, or "" for its own.
func (s planStep) provMark(slot string) string {
	if n := s.fromStep[slot]; n > 0 {
		return fmt.Sprintf(" (from step %d)", n)
	}
	return ""
}

// targetText renders a step's Target line: what the
// step acts on, in words. An each-file step names the
// set it applies to; a step whose command declares an
// acts-on hint leads with it (a bound folder seats as
// "in <folder>", a bound file as the file itself, and
// other slots stay out of the way — "database" is
// target enough for a migration); a step with no hint
// lists its bound slots as words ("folder trips",
// "from jim to jake"). Inherited values carry their
// "(from step N)" mark. A step with nothing bound and
// no hint has no target to name: "—".
func (s planStep) targetText() string {
	if s.eachFile {
		t := "each file"
		if s.eachFolder != "" {
			t += " in " + s.eachFolder
			if s.eachFolderIn > 0 {
				t += fmt.Sprintf(" (from step %d)", s.eachFolderIn)
			}
		}
		return t
	}
	if s.decl.ActsOn != "" {
		t := s.decl.ActsOn
		if v, ok := s.eff["folder"]; ok {
			t += " in " + v + s.provMark("folder")
		} else if v, ok := s.eff["file"]; ok {
			t += " " + v + s.provMark("file")
		}
		return t
	}
	var words []string
	for _, slot := range s.decl.Slots {
		v, ok := s.eff[slot]
		if !ok {
			continue
		}
		switch slot {
		case "from":
			words = append(words, "from "+v+s.provMark("from"))
		case "to":
			words = append(words, "to "+v+s.provMark("to"))
		default:
			words = append(words, slot+" "+v+s.provMark(slot))
		}
	}
	if len(words) == 0 {
		return "—"
	}
	return strings.Join(words, ", ")
}

// stepRender returns the declaration and positional
// values a step renders its preview and Would run lines
// with. For an each-file step the file slot is dropped
// from both: the step applies to a set of files, not to
// any one file — least of all a favourite the extractor
// offered for a segment that named none, which must not
// ride into the rendered command.
func (s planStep) stepRender() (cmdmatch.Declaration, []string) {
	decl, vals := s.decl, s.posValues()
	if !s.eachFile {
		return decl, vals
	}
	var slots []string
	var out []string
	for j, slot := range decl.Slots {
		if slot == "file" {
			continue
		}
		slots = append(slots, slot)
		if j < len(vals) {
			out = append(out, vals[j])
		} else {
			out = append(out, "")
		}
	}
	for len(out) > 0 && out[len(out)-1] == "" {
		out = out[:len(out)-1]
	}
	decl.Slots = slots
	return decl, out
}

// wouldRunText renders a step's Would run line: the
// command with the step's effective values, exactly as
// if they had all been spoken in its own segment — and,
// for an each-file step, the per-file suffix naming the
// set it applies to.
func (s planStep) wouldRunText() string {
	decl, vals := s.stepRender()
	base := wouldRun(decl, vals)
	if !s.eachFile {
		return base
	}
	if s.eachFolder != "" {
		return base + " — on each file in " + s.eachFolder
	}
	return base + " — on each file"
}

// printPlan renders the checklist: a header counting
// steps, ready steps, and gaps; then one checkbox entry
// per step — its stage and concept, a Target line
// naming what it acts on (with inherited values marked
// "(from step N)"), its preview line (the description
// with the step's effective values seated, exactly as
// the single-line path renders it), and its Would run
// line. A step whose concept removes or changes things
// carries the caution, as the single-line preview does.
// A gap step is a single checkbox line with the segment
// nobody knows. Then the missing-link notes, and the
// standing reminder that nothing has run.
func (d *demo) printPlan(steps []planStep, notes []string) {
	ready, gaps := 0, 0
	for _, s := range steps {
		if s.matched {
			ready++
		} else {
			gaps++
		}
	}
	fmt.Fprintf(d.out, "%s %d %s (%d ready, %d %s)\n", d.paint(ansiLabel, "plan:"), len(steps), plural(len(steps), "step"), ready, gaps, plural(gaps, "gap"))
	for i, s := range steps {
		if !s.matched {
			fmt.Fprintf(d.out, "[ ] %d. ? — no command knows this step yet: '%s'\n", i+1, s.segment)
			continue
		}
		fmt.Fprintf(d.out, "[ ] %d. %s — %s %s\n", i+1, stageLabel(stepStage(s)), d.paint(ansiLabel, "concept:"), s.decl.Name)
		fmt.Fprintf(d.out, "    %s %s\n", d.paint(ansiLabel, "Target:"), s.targetText())
		decl, vals := s.stepRender()
		fmt.Fprintf(d.out, "    %s %s\n", d.paint(ansiLabel, "preview:"), previewTextProvenance(decl, vals, s.fromStep))
		fmt.Fprintf(d.out, "    %s %s\n", d.paint(ansiLabel, "Would run:"), s.wouldRunText())
		if cmdmatch.HasRemoveWord(s.decl.Name) || cmdmatch.HasRemoveWord(s.decl.Description) {
			fmt.Fprintln(d.out, d.paint(ansiWarn, "    careful: this command removes or changes things."))
		}
	}
	for _, n := range notes {
		fmt.Fprintln(d.out, n)
	}
	fmt.Fprintln(d.out, "nothing has run yet.")
}

// containsString reports whether xs holds x exactly.
func containsString(xs []string, x string) bool {
	for _, s := range xs {
		if s == x {
			return true
		}
	}
	return false
}

// plural returns the word, or the word plus "s" — the
// header's small grammar.
func plural(n int, word string) string {
	if n == 1 {
		return word
	}
	return word + "s"
}
