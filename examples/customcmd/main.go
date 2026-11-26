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
// The extractor is the only trained piece, and it was
// trained ONCE, on generic data (examples/genericcommands.jsonl)
// about invented scripts that do not exist — it knows
// sentence shapes ("from X to Y", "in the X folder",
// "called X"), never your scripts. Your scripts live only
// in the registry, so adding one needs no retraining: edit
// the file (the REPL re-reads it as you type) or use the
// REPL's `add` command, and the new script matches from
// the next line on.
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
// This example RUNS NOTHING. It prints the script it
// understood, the arguments it heard, a plain-words
// preview of what the command would do (the description
// with the heard values worked into it — the gate any
// future execution step would confirm against), and the
// exact command line it WOULD run. Executing that line is a
// separate, gated step — confirmation, allowlists, and
// sandboxing are the host program's job, as in
// examples/filecmd — and it is deliberately not in this
// example.
//
// The extractor brain ships trained (custombrain.gob at
// the repo root), so the example runs as-is:
//
//	go run ./examples/customcmd -model custombrain.gob -registry examples/commands.json
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
	line = strings.TrimSpace(line)
	if line == "" {
		return true
	}
	if line == "quit" || line == "exit" {
		return false
	}
	d.reloadIfChanged()
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

	decl, score, ok := d.matcher.Match(line)
	if !ok {
		ranked := d.matcher.Rank(line)
		if len(ranked) > 0 && ranked[0].Mass > 0 {
			fmt.Fprintf(d.out, "i don't know which script you mean (closest: %s at %.2f, under the %.2f bar).\n", ranked[0].Script.Name, ranked[0].Score, cmdmatch.Threshold)
		} else {
			fmt.Fprintln(d.out, "i don't know which script you mean.")
		}
		// An unmatched WANT is a teaching moment: offer
		// to learn it on the spot. Small talk (no closest
		// script, no action word) gets no offer.
		if d.looksLikeWant(line, ranked) {
			d.teachFlow(line)
		}
		return true
	}

	d.present(decl, score, line)
	return true
}

// present prints the understood block for a matched
// declaration: the script and its score, the argument
// values the extractor heard bound to the declared
// slots, a plain-words PREVIEW of what the command
// would do, and the exact command line that WOULD run.
func (d *demo) present(decl cmdmatch.Declaration, score float64, line string) {
	fmt.Fprintf(d.out, "understood: %s (%.2f) — %s\n", decl.Name, score, decl.Description)
	if len(decl.Slots) == 0 {
		d.printPreview(decl, nil)
		fmt.Fprintf(d.out, "would run: %s\n", decl.Command)
		return
	}
	raw, err := d.extract(line)
	if err != nil {
		fmt.Fprintf(d.out, "but the extractor brain failed: %v — no arguments read.\n", err)
		d.printPreview(decl, nil)
		fmt.Fprintf(d.out, "would run: %s (arguments missing)\n", decl.Command)
		return
	}
	values := strings.Fields(raw)
	var bound []string
	for i, slot := range decl.Slots {
		if i < len(values) {
			bound = append(bound, fmt.Sprintf("%s=%s", slot, values[i]))
		} else {
			bound = append(bound, fmt.Sprintf("%s=(not heard)", slot))
		}
	}
	fmt.Fprintf(d.out, "arguments: %s\n", strings.Join(bound, ", "))
	if len(values) > len(decl.Slots) {
		fmt.Fprintf(d.out, "(also heard, with no slot to bind: %s)\n", strings.Join(values[len(decl.Slots):], " "))
	}
	d.printPreview(decl, values)
	fmt.Fprintf(d.out, "would run: %s\n", wouldRun(decl, values))
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
// The preview is the gate any future execution step
// would confirm against: this is what the user would be
// saying yes to.
func (d *demo) printPreview(decl cmdmatch.Declaration, values []string) {
	fmt.Fprintf(d.out, "preview: %s\n", previewText(decl, values))
	fmt.Fprintf(d.out, "  runs: %s\n", wouldRun(decl, values))
	fmt.Fprintln(d.out, "  nothing has run yet.")
	if cmdmatch.HasRemoveWord(decl.Name) || cmdmatch.HasRemoveWord(decl.Description) {
		fmt.Fprintln(d.out, "  careful: this command removes or changes things.")
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
	text := decl.Description
	bound := map[string]string{}
	for i, slot := range decl.Slots {
		if i < len(values) {
			bound[slot] = values[i]
		} else {
			bound[slot] = "(not heard)"
		}
	}
	seated := map[string]bool{}
	if from, hasFrom := bound["from"]; hasFrom {
		if to, hasTo := bound["to"]; hasTo {
			if loc := reFromToSeat.FindStringIndex(text); loc != nil {
				text = text[:loc[0]] + "from " + from + " to " + to + text[loc[1]:]
				seated["from"], seated["to"] = true, true
			}
		}
	}
	if v, ok := bound["file"]; ok {
		if loc := reFileSeat.FindStringIndex(text); loc != nil {
			text = text[:loc[0]] + "the file " + v + text[loc[1]:]
			seated["file"] = true
		}
	}
	if v, ok := bound["folder"]; ok {
		if loc := reFolderSeat.FindStringIndex(text); loc != nil {
			text = text[:loc[0]] + "the folder " + v + text[loc[1]:]
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
			unseated = append(unseated, slot+"="+bound[slot])
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
	if !ok1 || !ok2 || !ok3 || !ok4 {
		fmt.Fprintln(d.out, "add cancelled.")
		return
	}
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

	d := &demo{
		registryPath: *registryFlag,
		reg:          reg,
		matcher:      cmdmatch.NewMatcher(reg),
		regStamp:     stampOf(*registryFlag),
		out:          os.Stdout,
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

	fmt.Println("customcmd: your scripts, described not trained. Say what you want in your own words;")
	fmt.Println("i will name the script i think you mean and the exact command i WOULD run.")
	fmt.Println("i never run anything — understanding only; executing is a separate, gated step.")
	fmt.Printf("registry: %s (%d scripts). type 'list' to see them, 'add' to declare a new one, 'quit' to exit.\n", *registryFlag, len(reg.Scripts))
	fmt.Println("if i don't know a want, i'll ask to be taught it — and you can declare a command any time:")
	fmt.Println("'i have a command named copy and it will copy files from one folder to another and it is in file copy.sh'.")
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
