// Command customcmd is a runnable example of commands
// bound by DESCRIPTION instead of by training: you declare
// each of your own scripts once — a name, the command, and
// a plain description of what it does — in a registry file
// (examples/commands.json), and from then on you just say
// what you want. A lexical matcher (the cmdmatch package)
// reads your line against the registered descriptions and
// says which script you mean; a small trained brain, the
// extractor, reads the argument values out of your line.
//
// The extractor was trained ONCE, on generic data
// (examples/genericcommands.jsonl) about invented scripts
// that do not exist — it knows sentence shapes ("from X
// to Y", "in the X folder", "called X"), never your
// scripts. Your scripts live only in the registry, so
// adding one needs no retraining: edit the file (the REPL
// re-reads it as you type) or use the REPL's `add`
// command, and the new script matches from the next line
// on.
//
// And the extractor is no longer the only brain here
// (see brains.go): when the matcher abstains, an intent
// classifier trained in memory from examples/pairs.jsonl
// reads the line's meaning and routes it behind the
// scenes — conversation to the chat brain (brain.gob),
// file wants to the file brain (filebrain.gob, which
// only ever reports what it understood), and the rest to
// the teach offer, now showing the meaning it read. The
// user never chooses a brain: there is one prompt, and
// the .gob files are data it loads, not programs to pick
// between.
//
// The registry also grows IN CONVERSATION, two ways.
// When a line that looks like a want matches nothing, the
// REPL offers to learn it on the spot: you name the
// command that should run, and your want-line itself
// becomes the new entry's description — from then on the
// want (and its cousins) chooses that command. And a
// declaration — "i have a command named copy and it will
// copy files from one folder to another and it is in file
// copy.sh" — is parsed deterministically into the same
// kind of entry. Both doors write through the one
// registry save path `add` uses, so what is learned in a
// session is still there in the next one.
//
// A line carrying several wants at once is answered
// with a PLAN (plan.go): the line is split at its commas
// and ands, each segment is matched on its own, and the
// steps are laid out as a checklist ordered by the
// stages and prerequisites the registry declares — not
// by typing order — with the segments no command knows
// shown as gaps and offered for teaching. Steps share
// context as they go: a value one step binds can be
// inherited by a later step that needs it, and a step
// that speaks of files in the plural applies to each
// file of the plan's folder.
//
// The session also remembers what its presentations
// seated (objects.go): a value bound in one line is a
// typed OBJECT for the rest of the session, and a later
// line that refers back to it — "the same folder", or
// the bare name — gets the remembered value seated,
// marked "(remembered)" in its arguments. A later line
// that seats a known value under a conflicting type is
// told so, in the open; the value stays seated.
//
// A presentation on its own runs nothing: the program
// prints the script it understood, the arguments it
// heard, a plain-words preview of what the command
// would do (the description with the heard values
// worked into it), and the exact command line it WOULD
// run. Running it is the gated step in run.go: the bare
// line `run` (or `run it`) typed as the very next line
// after a presentation offers each step for its own
// explicit y/n confirmation; only registry scripts
// whose files exist and are executable can run; and a
// step runs by exec.Command with argv straight from
// the presented line — no shell anywhere. Without the
// trigger word, nothing ever runs.
//
// Every brain ships trained (custombrain.gob, brain.gob,
// and filebrain.gob at the repo root), so the example
// runs as-is — one command, no choices:
//
//	go run ./examples/customcmd
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/golangast/gollemer-brains/brain"
	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/intent"
	"github.com/golangast/gollemer-brains/interpret"
)

// demo is the REPL's state. extract is the trained
// extractor brain's answer to a line; nextLine feeds the
// `add` flow's prompts (in main it reads the same scanner
// as the main loop; in tests it serves a queue).
type demo struct {
	registryPath string
	reg          *cmdmatch.Registry
	matcher      *cmdmatch.Matcher
	regStamp     string // file stamp the loaded registry came from

	extract  func(line string) (raw string, err error)
	nextLine func(prompt string) (string, bool)
	out      io.Writer
	color    bool // colour the answers (color.go); tests leave it off

	// The other brains (brains.go). All nil-safe: a nil
	// field removes that brain's route, and with none of
	// them wired the REPL behaves exactly as it did
	// before they arrived.
	clf         *intent.Classifier // the in-memory intent read (pairs.jsonl)
	factorTbl   *factorTable       // the label factorization, derived lazily from clf (factor.go)
	chatAnswer  func(line string) (string, error)
	chatGists   map[string]string // per-label taught gist for confident question lines (brains.go)
	filePredict func(line string) (label string, conf float64)
	fileExtract func(line string) (raw string, err error)

	// objects is the session's object table (objects.go):
	// the typed values its presentations have seated,
	// remembered across lines so a later line can refer
	// back to them. Created lazily by objectTable().
	objects *interpret.ObjectTable

	// lastPresent is the single-line match most recently
	// presented — the only thing a correction ("no, …")
	// can refer to. handle consumes it on the very next
	// line, corrected or not: only the NEXT line can
	// correct a presentation (correct.go).
	lastPresent *presentedMatch

	// pendingRun is the presentation most recently made —
	// a single command, a plan, a routine — in runnable
	// form: the only thing the execution trigger ("run",
	// "run it") can act on. handle consumes it on the
	// very next line, run or not, with the same adjacency
	// discipline as lastPresent (run.go).
	pendingRun *runnable

	// interactive reports whether the REPL is talking to
	// a person at a terminal. Execution requires it: a
	// piped or scripted session can present all it likes,
	// but the gate's confirmation must come from the
	// interactive prompt (run.go).
	interactive bool

	// inspect turns on the provenance traces
	// (inspect.go): one [inspect] line per decision on
	// the handled line's path, printed after its
	// ordinary output. Off, nothing is collected or
	// printed and output is byte-identical.
	inspect bool
	trace   []string
}

// presentedMatch is a match the REPL just presented:
// the line that produced it and the script it chose.
type presentedMatch struct {
	line string
	decl cmdmatch.Declaration
}

// stampOf identifies a registry file's current contents
// cheaply (size + mtime): enough to notice an edit between
// two typed lines.
func stampOf(path string) string {
	fi, err := os.Stat(path)
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%d/%d", fi.Size(), fi.ModTime().UnixNano())
}

// reloadIfChanged re-reads the registry when the file has
// changed since the loaded copy — this is what makes a
// script added in another window (or by `add`) matchable
// on the very next line, with no restart and no retrain.
func (d *demo) reloadIfChanged() {
	stamp := stampOf(d.registryPath)
	if stamp == "" || stamp == d.regStamp {
		return
	}
	reg, err := cmdmatch.LoadRegistry(d.registryPath)
	if err != nil {
		fmt.Fprintf(d.out, "(registry %s changed but does not load: %v — keeping the previous registry)\n", d.registryPath, err)
		d.regStamp = stamp // do not nag on every line
		return
	}
	d.reg = reg
	d.matcher = cmdmatch.NewMatcher(reg)
	d.regStamp = stamp
}

// handle runs one input line. It returns false only for
// quit/exit.
func (d *demo) handle(line string) bool {
	// The inspect trace (inspect.go) is per handled
	// line: collected at the decision points below,
	// printed after the line's ordinary output. Off,
	// both the collection and the printing are no-ops.
	d.trace = nil
	defer d.flushInspect()
	line = strings.TrimSpace(line)
	if line == "" {
		return true
	}
	if line == "quit" || line == "exit" {
		return false
	}
	d.reloadIfChanged()

	// A correction ("no", "no i meant …", "wrong", "not
	// that one") is read against the match just
	// presented, if any. The presented state is consumed
	// by this line whatever it is: a correction anywhere
	// later than the next line has nothing to refer to,
	// and a correction word with no presentation before
	// it is an ordinary line.
	prev := d.lastPresent
	d.lastPresent = nil
	pending := d.pendingRun
	d.pendingRun = nil
	if isRunTrigger(line) {
		// The execution trigger (run.go): it acts on the
		// presentation before it and nothing else. With
		// no presentation before it the trigger word is
		// unclaimed — silence, like any unclaimed line.
		if pending != nil {
			d.runFlow(pending)
		}
		return true
	}
	if prev != nil && isCorrection(line) {
		d.correctionFlow(prev)
		return true
	}

	switch line {
	case "add":
		d.addFlow()
		return true
	case "list":
		for _, s := range d.reg.Scripts {
			fmt.Fprintf(d.out, "%s — %s\n", s.Name, s.Description)
		}
		return true
	}

	// A declaration ("i have a command named copy … it is
	// in file copy.sh") is teaching, not a want: it is
	// parsed and learned before matching is consulted.
	if p, isDecl := parseDeclaration(line); isDecl {
		d.declarationFlow(p)
		return true
	}

	// A line of several wants is a PLAN, not a single
	// want: it is split and each segment matched on its
	// own (see plan.go). Lines that do not split, or
	// whose segments match nothing, fall through to the
	// ordinary single-line handling below, unchanged.
	if _, isPlan := d.buildPlan(line); isPlan {
		d.handlePlan(line)
		return true
	}

	decl, score, ok := d.matcher.Match(line)
	if !ok {
		ranked := d.matcher.Rank(line)
		d.traceMatcherAbstain(ranked)
		// A near tie between two scripts is a question,
		// not a coin flip (tie.go): the one place the
		// abstain path speaks before the brains. Declining
		// — or no tie — falls through to the ordinary
		// abstain handling below, exactly as if the
		// question had never been asked.
		if d.disambiguate(line, ranked) {
			return true
		}
		// No script claims the line — the other brains
		// read it (brains.go): conversation goes to the
		// chat brain, file wants to the file brain's
		// understanding, behind the scenes. And when no
		// brain takes it and it is not a teachable want,
		// nothing is printed at all: the next prompt is
		// the whole answer. (John's rule, 2026-11-26:
		// no dead-end line — silence for unclaimed lines.)
		read, haveRead := d.readIntent(line)
		if d.routeAbstain(line, ranked, read, haveRead) {
			return true
		}
		// An unmatched WANT is a teaching moment: offer
		// to learn it on the spot. Small talk (no closest
		// script, no action word, no want read) gets no
		// offer — and talk the intent brain read as talk
		// was already taken by the chat route above.
		d.teachOffer(line, ranked, read, haveRead)
		return true
	}

	d.traceMatch(line, decl, score)
	d.presentMatch(decl, score, line)
	return true
}

// presentMatch presents one matched entry for a line: a
// script gets the understood block (present), a routine
// expands into its plan (routine.go). Every caller that
// answers a single-line match goes through here, so the
// two shapes can never be confused downstream.
func (d *demo) presentMatch(decl cmdmatch.Declaration, score float64, line string) {
	if len(decl.Steps) > 0 {
		d.handleRoutine(decl, line)
		return
	}
	d.present(decl, score, line)
}

// present prints the understood block for a matched
// declaration: the script and its score, the argument
// values the extractor heard bound to the declared
// slots, a plain-words PREVIEW of what the command
// would do, and the exact command line that WOULD run.
func (d *demo) present(decl cmdmatch.Declaration, score float64, line string) {
	d.lastPresent = &presentedMatch{line: line, decl: decl}
	fmt.Fprintf(d.out, "%s %s (%.2f) — %s\n", d.paint(ansiLabel, "understood:"), decl.Name, score, decl.Description)
	if len(decl.Slots) == 0 {
		d.printPreview(decl, nil)
		fmt.Fprintf(d.out, "%s %s\n", d.paint(ansiLabel, "would run:"), decl.Command)
		d.pendingRun = singleRunnable(decl, decl.Command)
		return
	}
	raw, err := d.extract(line)
	if err != nil {
		fmt.Fprintf(d.out, "but the extractor brain failed: %v — no arguments read.\n", err)
		d.printPreview(decl, nil)
		fmt.Fprintf(d.out, "%s %s (arguments missing)\n", d.paint(ansiLabel, "would run:"), decl.Command)
		// A presentation whose arguments were never read
		// is not runnable: the would-run line it showed
		// is not a command anyone should confirm.
		d.pendingRun = nil
		return
	}
	values := bindSlotValues(decl, line, strings.Fields(raw))
	// The object layer (objects.go): a line that refers
	// back to a value an earlier presentation seated —
	// "the same folder", or the bare name — gets that
	// value seated from the session's memory, marked as
	// remembered; a seat that contradicts the remembered
	// type is reported under the arguments. A line with
	// no reference back and no conflict is untouched by
	// the layer, byte for byte.
	values, remembered, conflicts := d.applySessionObjects(decl, line, values)
	var bound []string
	for i, slot := range decl.Slots {
		if i < len(values) && values[i] != "" {
			b := fmt.Sprintf("%s=%s", slot, values[i])
			if remembered[i] {
				b += " (remembered)"
			}
			bound = append(bound, b)
		} else {
			bound = append(bound, fmt.Sprintf("%s=(not heard)", slot))
		}
	}
	fmt.Fprintf(d.out, "%s %s\n", d.paint(ansiLabel, "arguments:"), strings.Join(bound, ", "))
	if len(values) > len(decl.Slots) {
		fmt.Fprintf(d.out, "(also heard, with no slot to bind: %s)\n", strings.Join(values[len(decl.Slots):], " "))
	}
	for _, conflict := range conflicts {
		fmt.Fprintln(d.out, d.paint(ansiWarn, conflict))
	}
	d.printPreview(decl, values)
	rendered := wouldRun(decl, values)
	fmt.Fprintf(d.out, "%s %s\n", d.paint(ansiLabel, "would run:"), rendered)
	d.pendingRun = singleRunnable(decl, rendered)
}

// printPreview prints the preview block between the
// understood line and the would-run line: the command's
// own description with the bound values worked into its
// seats, the exact line that would run (built by the
// same wouldRun as the would-run line, so the two can
// never disagree), and the standing reminder that
// nothing has run. A command whose name or description
// carries a remove-family word — it deletes, cleans,
// wipes, throws away — earns one more line: the caution.
// The preview is the gate the execution step (run.go)
// confirms against: this is what the user says yes to.
func (d *demo) printPreview(decl cmdmatch.Declaration, values []string) {
	fmt.Fprintf(d.out, "%s %s\n", d.paint(ansiLabel, "preview:"), previewText(decl, values))
	fmt.Fprintf(d.out, "  %s %s\n", d.paint(ansiLabel, "runs:"), wouldRun(decl, values))
	fmt.Fprintln(d.out, "  nothing has run yet.")
	if cmdmatch.HasRemoveWord(decl.Name) || cmdmatch.HasRemoveWord(decl.Description) {
		fmt.Fprintln(d.out, d.paint(ansiWarn, "  careful: this command removes or changes things."))
	}
}

var (
	// The from/to seat: "… from one folder to another".
	reFromToSeat = regexp.MustCompile(`(?i)\bfrom\s+[^,]*?\s+to\s+[^,]*`)
	// The file seat: "a file" / "the file".
	reFileSeat = regexp.MustCompile(`(?i)\b(?:a|the)\s+file\b`)
	// The folder seat: "a folder" / "the folder".
	reFolderSeat = regexp.MustCompile(`(?i)\b(?:a|the)\s+folder\b`)
)

// previewText works the bound values into the
// description's obvious seats, deterministically — the
// preview is the user's own description with their
// values in it, and no prose the description does not
// support is invented:
//
//   - a from/to pair replaces the description's
//     from-phrase and to-phrase ("migrates a database
//     from one folder to another" → "… from jim to
//     jake");
//   - a bound file slot turns the description's file
//     mention into the actual file ("deletes the first
//     line of a file" → "… of the file cat.txt");
//   - a bound folder slot does the same for its folder
//     mention ("in a folder" → "in the folder trips").
//
// A slot with no seat in the description is listed after
// it ("— with folder=trips"); a slot whose value was not
// heard is marked "(not heard)" in its seat or its
// listing, never hidden.
func previewText(decl cmdmatch.Declaration, values []string) string {
	return previewTextProvenance(decl, values, nil)
}

// previewTextProvenance is previewText with provenance:
// prov maps a slot name to the number of the plan step
// its value was inherited from (the plan layer's shared
// context, plan.go), and a value that arrived that way
// is marked "(from step N)" where it lands — in its seat
// or in its listing. The single-line path passes nil:
// nothing there is inherited, and its output is exactly
// previewText's. An empty string in values counts as not
// heard (the extractor's Fields never yields one; the
// plan layer's positional padding can).
func previewTextProvenance(decl cmdmatch.Declaration, values []string, prov map[string]int) string {
	mark := func(slot string) string {
		if n := prov[slot]; n > 0 {
			return fmt.Sprintf(" (from step %d)", n)
		}
		return ""
	}
	text := decl.Description
	bound := map[string]string{}
	for i, slot := range decl.Slots {
		if i < len(values) && values[i] != "" {
			bound[slot] = values[i]
		} else {
			bound[slot] = "(not heard)"
		}
	}
	seated := map[string]bool{}
	if from, hasFrom := bound["from"]; hasFrom {
		if to, hasTo := bound["to"]; hasTo {
			if loc := reFromToSeat.FindStringIndex(text); loc != nil {
				text = text[:loc[0]] + "from " + from + mark("from") + " to " + to + mark("to") + text[loc[1]:]
				seated["from"], seated["to"] = true, true
			}
		}
	}
	if v, ok := bound["file"]; ok {
		if loc := reFileSeat.FindStringIndex(text); loc != nil {
			text = text[:loc[0]] + "the file " + v + mark("file") + text[loc[1]:]
			seated["file"] = true
		}
	}
	if v, ok := bound["folder"]; ok {
		if loc := reFolderSeat.FindStringIndex(text); loc != nil {
			text = text[:loc[0]] + "the folder " + v + mark("folder") + text[loc[1]:]
			seated["folder"] = true
		}
	}
	var unseated []string
	for _, slot := range decl.Slots {
		if seated[slot] {
			continue
		}
		if bound[slot] == "(not heard)" {
			unseated = append(unseated, slot+": (not heard)")
		} else {
			unseated = append(unseated, slot+"="+bound[slot]+mark(slot))
		}
	}
	if len(unseated) > 0 {
		text += " — with " + strings.Join(unseated, ", ")
	}
	return text
}

// wouldRun renders the command line for a declaration
// and the values heard for it. A command that names its
// slots inline ({folder}) has each placeholder replaced
// by the value bound to that slot; any other command
// takes its bound values appended, in slot order. Either
// way the line carries DECLARED SLOTS ONLY: a value the
// extractor heard beyond the slots is reported (in
// present's "also heard" line) but never passed to a
// command that never asked for it.
func wouldRun(decl cmdmatch.Declaration, values []string) string {
	cmd := decl.Command
	if strings.Contains(cmd, "{") {
		for i, slot := range decl.Slots {
			if i < len(values) {
				cmd = strings.ReplaceAll(cmd, "{"+slot+"}", values[i])
			}
		}
		return cmd
	}
	n := len(values)
	if n > len(decl.Slots) {
		n = len(decl.Slots)
	}
	if n == 0 {
		return cmd
	}
	return cmd + " " + strings.Join(values[:n], " ")
}

// addFlow declares a new script interactively and appends
// it to the registry file. The new script is matchable
// from the next line on — the matcher is rebuilt from the
// registry the same way it is after any edit, and nothing
// anywhere is retrained.
func (d *demo) addFlow() {
	ask := func(prompt string) (string, bool) {
		s, ok := d.nextLine(prompt)
		return strings.TrimSpace(s), ok
	}
	name, ok1 := ask("script name: ")
	command, ok2 := ask("command: ")
	description, ok3 := ask("description (what it does, in plain words): ")
	slotsLine, ok4 := ask("slots (comma-separated, blank for none): ")
	stage, ok5 := ask("stage (inspect, prepare, execute, verify, cleanup — blank for none): ")
	if !ok1 || !ok2 || !ok3 || !ok4 || !ok5 {
		fmt.Fprintln(d.out, "add cancelled.")
		return
	}
	stage = strings.ToLower(strings.TrimSpace(stage))
	var slots []string
	for _, s := range strings.Split(slotsLine, ",") {
		if s = strings.TrimSpace(s); s != "" {
			slots = append(slots, s)
		}
	}
	var examples []string
	for {
		ex, ok := ask("example sentence (blank to finish): ")
		if !ok || ex == "" {
			break
		}
		examples = append(examples, ex)
	}
	decl := cmdmatch.Declaration{
		Name:        name,
		Command:     command,
		Description: description,
		Slots:       slots,
		Examples:    examples,
		Stage:       stage,
	}
	if !d.persist(decl) {
		return
	}
	fmt.Fprintf(d.out, "added %s — it matches from the next line on. nothing was retrained.\n", name)
}

// persist is the ONE registry write path: `add`, the
// teach-on-ask flow, and a parsed declaration all land
// here. The declaration is validated, appended, and the
// registry file rewritten — so what is learned in a
// session survives restarts — and the matcher is rebuilt
// over the new registry, so the entry matches from the
// next line on. Nothing anywhere is retrained.
func (d *demo) persist(decl cmdmatch.Declaration) bool {
	// Teach-time relation inference: a declaration that
	// states no relations gets the ones its own shape
	// implies — the from/to slots named as moves_from /
	// moves_to, the Needs list as requires edges — written
	// into the registry, so what was implicit in the
	// sentence becomes inspectable, validatable data.
	if len(decl.Relations) == 0 {
		decl.Relations = cmdmatch.DeriveRelations(decl)
	}
	// And the concept property the sentence implies: the
	// type of the thing the command acts on, read off its
	// description's own object words.
	if decl.ObjectType == "" {
		decl.ObjectType = interpret.DeclarationProperties(decl).ObjectType
	}
	if err := d.reg.Add(decl); err != nil {
		fmt.Fprintf(d.out, "not saved: %v\n", err)
		return false
	}
	if err := d.reg.Save(d.registryPath); err != nil {
		fmt.Fprintf(d.out, "not saved: could not write the registry: %v\n", err)
		d.reg.Scripts = d.reg.Scripts[:len(d.reg.Scripts)-1]
		return false
	}
	d.matcher = cmdmatch.NewMatcher(d.reg)
	d.regStamp = stampOf(d.registryPath)
	return true
}

// cleanValues reports whether an extractor answer is a
// plausible argument list: at least one field, and every
// field a plain token (no path separators, no "..") — the
// answer is only ever printed, but a garbage answer should
// still fall back to the greedy decode, as in filecmd.
func cleanValues(raw string) bool {
	fields := strings.Fields(raw)
	if len(fields) == 0 {
		return false
	}
	for _, f := range fields {
		if strings.ContainsAny(f, `/\`) || strings.Contains(f, "..") {
			return false
		}
	}
	return true
}

func main() {
	modelFlag := flag.String("model", "custombrain.gob", "path to the trained extractor brain (.gob trained on examples/genericcommands.jsonl)")
	registryFlag := flag.String("registry", "examples/commands.json", "path to the registry of your described scripts")
	chatFlag := flag.String("chat", "brain.gob", "path to the trained chat brain (.gob trained on examples/pairs.jsonl)")
	fileFlag := flag.String("filebrain", "filebrain.gob", "path to the trained file brain (.gob trained on examples/filecommands.jsonl)")
	pairsFlag := flag.String("pairs", "examples/pairs.jsonl", "path to the labeled pairs the intent classifier trains on, in memory")
	inspectFlag := flag.Bool("inspect", false, "print a provenance trace ([inspect] lines) after each line's output: which words, cousin links, kind-tree steps and rules produced the result")
	flag.Parse()

	b, err := brain.LoadBrain(*modelFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "customcmd:", err)
		os.Exit(1)
	}
	reg, err := cmdmatch.LoadRegistry(*registryFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "customcmd:", err)
		os.Exit(1)
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	// Execution (run.go) requires a person at a terminal:
	// stdin a character device is what makes this session
	// interactive. A piped session still presents and
	// teaches — it just can never run anything.
	interactive := false
	if fi, err := os.Stdin.Stat(); err == nil && fi.Mode()&os.ModeCharDevice != 0 {
		interactive = true
	}

	d := &demo{
		registryPath: *registryFlag,
		reg:          reg,
		matcher:      cmdmatch.NewMatcher(reg),
		regStamp:     stampOf(*registryFlag),
		out:          os.Stdout,
		color:        colorEnabled(),
		interactive:  interactive,
		inspect:      *inspectFlag,
	}

	// The intent classifier trains IN MEMORY from the
	// pairs file — no model of its own, so it can never
	// go stale against the data. It is a small softmax,
	// but 1,098 labeled rows still take it a few seconds
	// (measured ~9 s on a modest VM at the time of
	// writing); that pause is the startup cost of the
	// read that routes everything else. A pairs file
	// that will not load or train removes the read (and
	// every route it steers) rather than the program.
	if pairs, err := data.LoadPairs(*pairsFlag); err == nil {
		if clf, err := intent.Train(pairs, intent.DefaultThreshold); err == nil {
			d.clf = clf
		}
		// The taught gists ride the same pairs: the modal
		// output per label, for the chat route's gist
		// answers (brains.go: modalGists).
		d.chatGists = modalGists(pairs)
	}

	// The chat brain: conversation and questions. Loaded
	// when its data file is there; absent, its route
	// simply never fires. (Its own intent sidecar travels
	// with it, but the read above is the in-memory one —
	// one classifier, one source of truth.)
	if cb, err := brain.LoadBrain(*chatFlag); err == nil {
		cb.FixTypos = true // cmd/ask's standard policy for this brain
		d.chatAnswer = func(line string) (string, error) {
			checked, err := cb.AnswerChecked(line, rng)
			if err == nil && strings.TrimSpace(checked) != "" {
				return checked, nil
			}
			return cb.Answer(line)
		}
	}

	// The file brain: it reads file wants and reports
	// what it understood — it never acts here. Its
	// classifier rides in its sidecar, as in filecmd.
	if fb, err := brain.LoadBrain(*fileFlag); err == nil && fb.Intents != nil {
		d.filePredict = fb.Intents.Predict
		d.fileExtract = func(line string) (string, error) {
			checked, err := fb.AnswerChecked(line, rng)
			if err == nil {
				if _, ok := cleanFileName(checked); ok {
					return checked, nil
				}
			}
			return fb.Answer(line)
		}
	}
	// The extractor: the seq2seq brain's answer to the same
	// line, trained (by the generic rows) to be the bare
	// argument values. Checked sampling first — the repo's
	// standard decode policy, which prefers a reply that
	// carries a word from your line, as the right value
	// does. Greedy is the fallback when the checked answer
	// is not a clean value list. (Wired as in filecmd.)
	d.extract = func(line string) (string, error) {
		checked, err := b.AnswerChecked(line, rng)
		if err == nil && cleanValues(checked) {
			return checked, nil
		}
		return b.Answer(line)
	}
	d.nextLine = func(prompt string) (string, bool) {
		fmt.Fprint(os.Stdout, prompt)
		if !sc.Scan() {
			return "", false
		}
		return sc.Text(), true
	}

	// No banner (John's rule, 2026-11-26): the program
	// starts at the prompt, and the prompt is the whole
	// greeting.
	for {
		fmt.Print("> ")
		if !sc.Scan() {
			fmt.Println()
			return
		}
		if !d.handle(sc.Text()) {
			return
		}
	}
}
