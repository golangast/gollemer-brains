// Command filecmd is a runnable example of the intent
// mechanism doing real work: you talk to a folder of files
// in plain sentences, and files appear, disappear, and get
// listed — driven by a trained intent classifier (which
// meaning?) plus a trained seq2seq brain used as an argument
// extractor (which filename?). There is no sentence parsing
// anywhere in this program: no keywords, no patterns, no
// grammar. The only hand-written logic is the binding — one
// small function per meaning — and the safety guards, which
// are the host program's job, not the model's.
//
// Train the demo brain first (see the README section
// "Runnable example: talk to your filesystem"):
//
//	go run ./cmd/train -data examples/filecommands.jsonl -out filebrain.gob -epochs 300 -layers char,copy -augment 0.5 -tokenizer bpe
//	go run ./examples/filecmd -model filebrain.gob
//
// Everything the demo touches lives in a demo-files/ folder
// next to where you run it. Nothing outside that folder is
// ever read, written, or removed.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/golangast/gollemer-brains/brain"
)

// deleteMinConfidence is the host's own bar for the one
// destructive meaning: the classifier's general abstention
// threshold (0.5) is not enough to remove a file. Below
// this, the demo does nothing and says so.
const deleteMinConfidence = 0.8

// demo is the REPL's state. The two function fields are the
// trained pieces; main wires them to the loaded brain, and
// tests wire them to a real classifier plus a stub
// extractor.
type demo struct {
	dir       string // the sandbox folder (absolute)
	predict   func(line string) (label string, conf float64)
	extract   func(line string) (raw string, err error)
	confirmFn func(prompt string) bool
	out       io.Writer
}

// sanitizeName turns the extractor brain's raw answer into
// a filename the host is willing to touch — or refuses it.
// The model proposes; the host disposes. A usable answer is
// exactly one token, a plain filename: letters, digits,
// dots, dashes, underscores, with a real extension. Anything
// containing a path separator or ".." is rejected outright:
// the extractor is a tiny neural net, and its answer is
// never trusted as a path.
//
// This guard is the host program's job, not the model's —
// the same is true of the sandbox confinement in
// sandboxPath and the delete confirmation in handle.
func sanitizeName(raw string) (string, error) {
	fields := strings.Fields(strings.TrimSpace(raw))
	if len(fields) != 1 {
		return "", fmt.Errorf("answer %q is not a single filename", strings.TrimSpace(raw))
	}
	name := fields[0]
	if strings.Contains(name, "/") || strings.Contains(name, "\\") {
		return "", fmt.Errorf("filename %q contains a path separator", name)
	}
	if strings.Contains(name, "..") {
		return "", fmt.Errorf("filename %q contains \"..\"", name)
	}
	if !nameRe.MatchString(name) {
		return "", fmt.Errorf("filename %q is not a plain name with an extension", name)
	}
	return name, nil
}

var nameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]*\.[A-Za-z0-9]+$`)

// sandboxPath joins a sanitized name onto the sandbox
// folder and proves the result is directly inside it —
// confinement is checked, not assumed from sanitizeName.
func sandboxPath(dir, name string) (string, error) {
	p := filepath.Join(dir, name)
	if filepath.Dir(p) != dir {
		return "", fmt.Errorf("path for %q escapes the sandbox folder", name)
	}
	return p, nil
}

// handle runs one input line: predict the meaning, extract
// the argument when the meaning needs one, act inside the
// sandbox, and report what happened. It returns false only
// for quit/exit.
func (d *demo) handle(line string) bool {
	line = strings.TrimSpace(line)
	if line == "" {
		return true
	}
	if line == "quit" || line == "exit" {
		return false
	}
	label, conf := d.predict(line)
	if label == "" {
		fmt.Fprintf(d.out, "I don't have a confident meaning for that — nothing done.\n")
		return true
	}
	switch label {
	case "list_files":
		fmt.Fprintf(d.out, "intent: list_files (%.2f)\n", conf)
		d.list()
	case "create_file", "delete_file":
		if label == "delete_file" && conf < deleteMinConfidence {
			fmt.Fprintf(d.out, "intent: delete_file (%.2f) — not confident enough to delete (this demo wants %.2f); nothing done.\n", conf, deleteMinConfidence)
			return true
		}
		raw, err := d.extract(line)
		if err != nil {
			fmt.Fprintf(d.out, "intent: %s (%.2f), but the extractor brain failed: %v — nothing done.\n", label, conf, err)
			return true
		}
		name, err := sanitizeName(raw)
		if err != nil {
			fmt.Fprintf(d.out, "intent: %s (%.2f), but no clean filename came back (%v) — nothing done.\n", label, conf, err)
			return true
		}
		fmt.Fprintf(d.out, "intent: %s (%.2f), file: %s\n", label, conf, name)
		if label == "create_file" {
			d.create(name)
		} else {
			d.remove(name)
		}
	case "chat":
		// The classifier's fourth class: it read the line
		// as conversation, not as any file meaning. That
		// is a positive "not a command" — the demo says
		// so and does nothing, exactly as it does for a
		// line it cannot read at all.
		fmt.Fprintf(d.out, "intent: chat (%.2f) — that's conversation, not a file command; nothing done.\n", conf)
	default:
		fmt.Fprintf(d.out, "intent: %s (%.2f) — this demo has no binding for that meaning; nothing done.\n", label, conf)
	}
	return true
}

func (d *demo) create(name string) {
	p, err := sandboxPath(d.dir, name)
	if err != nil {
		fmt.Fprintf(d.out, "refused: %v\n", err)
		return
	}
	f, err := os.OpenFile(p, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if os.IsExist(err) {
		fmt.Fprintf(d.out, "%s already exists in demo-files/ — left it exactly as it was.\n", name)
		return
	}
	if err != nil {
		fmt.Fprintf(d.out, "could not create %s: %v\n", name, err)
		return
	}
	f.Close()
	fmt.Fprintf(d.out, "created demo-files/%s\n", name)
}

func (d *demo) remove(name string) {
	p, err := sandboxPath(d.dir, name)
	if err != nil {
		fmt.Fprintf(d.out, "refused: %v\n", err)
		return
	}
	if _, err := os.Stat(p); os.IsNotExist(err) {
		fmt.Fprintf(d.out, "%s isn't in demo-files/ — nothing to delete.\n", name)
		return
	}
	if !d.confirmFn(fmt.Sprintf("delete %s? [y/n] ", name)) {
		fmt.Fprintf(d.out, "ok — left %s alone.\n", name)
		return
	}
	if err := os.Remove(p); err != nil {
		fmt.Fprintf(d.out, "could not delete %s: %v\n", name, err)
		return
	}
	fmt.Fprintf(d.out, "deleted demo-files/%s\n", name)
}

func (d *demo) list() {
	entries, err := os.ReadDir(d.dir)
	if err != nil {
		fmt.Fprintf(d.out, "could not read demo-files/: %v\n", err)
		return
	}
	var names []string
	for _, e := range entries {
		if e.Type().IsRegular() {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	if len(names) == 0 {
		fmt.Fprintln(d.out, "demo-files/ is empty.")
		return
	}
	fmt.Fprintf(d.out, "demo-files/: %s\n", strings.Join(names, ", "))
}

func main() {
	modelFlag := flag.String("model", "filebrain.gob", "path to the trained brain (.gob with its .intents.json sidecar)")
	flag.Parse()

	b, err := brain.LoadBrain(*modelFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "filecmd:", err)
		os.Exit(1)
	}
	if b.Intents == nil {
		fmt.Fprintln(os.Stderr, "filecmd: this brain has no intent classifier — train it on examples/filecommands.jsonl (rows with \"intent\" fields) first")
		os.Exit(1)
	}
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, "filecmd:", err)
		os.Exit(1)
	}
	dir := filepath.Join(cwd, "demo-files")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "filecmd:", err)
		os.Exit(1)
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	d := &demo{
		dir:     dir,
		predict: b.Intents.Predict,
		out:     os.Stdout,
	}
	// The extractor: the seq2seq brain's answer to the same
	// line, trained (by the output-carrying rows) to be the
	// bare filename. Checked sampling first — it is the
	// repo's standard decode policy, and it prefers a reply
	// that carries a word from your line, which the right
	// filename does. Greedy is the fallback when the checked
	// answer is not a clean name.
	d.extract = func(line string) (string, error) {
		checked, err := b.AnswerChecked(line, rng)
		if err == nil {
			if _, serr := sanitizeName(checked); serr == nil {
				return checked, nil
			}
		}
		return b.Answer(line)
	}
	d.confirmFn = func(prompt string) bool {
		fmt.Fprint(os.Stdout, prompt)
		if !sc.Scan() {
			return false
		}
		ans := strings.ToLower(strings.TrimSpace(sc.Text()))
		return ans == "y" || ans == "yes"
	}

	fmt.Println("filecmd: talk to the demo-files/ folder. Try \"create a file called notes.txt\", \"what files do i have\", \"get rid of notes.txt\". Type 'quit' to exit.")
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
