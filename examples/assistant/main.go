// Command assistant is the two runnable examples in one
// program: a chat brain that talks, and a file brain whose
// intent classifier can act on a folder. Every line is
// routed by MEANING, not by parsing: the file brain's
// classifier reads the line first, and when it returns a
// file meaning — create_file, delete_file, list_files —
// the line is handled exactly as examples/filecmd handles
// it, gates and all. When it returns nothing, the line
// falls through to the chat brain and is simply answered.
//
// That routing is the whole design, and its honesty
// condition is the filecmd chapter's caveat: the
// classifier can false-fire on a chat line, so a line
// meant as small talk can occasionally be taken as a
// command. The safety rules are therefore not decoration
// here — the delete confidence bar, the confirmation
// prompt that names the file, and the sandbox are what
// stand between a false-fire and a real file. A delete
// the classifier is not sure enough about is refused as
// a command; it does NOT fall through and turn into
// small talk.
//
// Train both brains first (see the README section
// "Runnable example: one assistant, both brains"):
//
//	go run ./cmd/train -data examples/pairs.jsonl -out brain.gob -epochs 60 -layers copy
//	go run ./cmd/train -data examples/filecommands.jsonl -out filebrain.gob -epochs 300 -layers char,copy -augment 0.5 -tokenizer bpe
//	go run ./examples/assistant -chat brain.gob -files filebrain.gob
//
// The two halves are independent: retrain either brain on
// better data and the assistant improves on that side with
// no code change here. Everything the file half touches
// lives in a demo-files/ folder next to where you run it;
// nothing outside that folder is ever read, written, or
// removed.
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
// this, the assistant refuses the command and says so —
// it never quietly re-routes a refused delete to chat.
// (Same value, same reasoning as examples/filecmd.)
const deleteMinConfidence = 0.8

// assistant is the REPL's state. The function fields are
// the trained pieces; main wires them to the two loaded
// brains, and tests wire them to a real classifier plus
// stubs for the two seq2seq brains.
type assistant struct {
	dir       string // the sandbox folder (absolute)
	predict   func(line string) (label string, conf float64)
	extract   func(line string) (raw string, err error)
	chat      func(line string) (reply string, err error)
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
// (Copied from examples/filecmd; the duplication is
// deliberate — each example stands alone.)
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

// handle runs one input line: route it by the file brain's
// predicted meaning, act or answer accordingly, and report
// what happened. It returns false only for quit/exit.
func (a *assistant) handle(line string) bool {
	line = strings.TrimSpace(line)
	if line == "" {
		return true
	}
	if line == "quit" || line == "exit" {
		return false
	}
	label, conf := a.predict(line)
	switch label {
	case "list_files":
		fmt.Fprintf(a.out, "intent: list_files (%.2f)\n", conf)
		a.list()
	case "create_file", "delete_file":
		if label == "delete_file" && conf < deleteMinConfidence {
			fmt.Fprintf(a.out, "intent: delete_file (%.2f) — not confident enough to delete (this demo wants %.2f); nothing done.\n", conf, deleteMinConfidence)
			return true
		}
		raw, err := a.extract(line)
		if err != nil {
			fmt.Fprintf(a.out, "intent: %s (%.2f), but the extractor brain failed: %v — nothing done.\n", label, conf, err)
			return true
		}
		name, err := sanitizeName(raw)
		if err != nil {
			fmt.Fprintf(a.out, "intent: %s (%.2f), but no clean filename came back (%v) — nothing done.\n", label, conf, err)
			return true
		}
		fmt.Fprintf(a.out, "intent: %s (%.2f), file: %s\n", label, conf, name)
		if label == "create_file" {
			a.create(name)
		} else {
			a.remove(name)
		}
	default:
		// No file meaning the host has a binding for —
		// including the classifier abstaining altogether,
		// and including a confident "chat" reading: the
		// file data teaches the classifier what small
		// talk looks like, so a greeting has somewhere to
		// land besides a file intent. Either way the line
		// is conversation: the chat brain answers it, and
		// nothing else is printed.
		reply, err := a.chat(line)
		if err != nil {
			fmt.Fprintf(a.out, "the chat brain could not answer that: %v\n", err)
			return true
		}
		fmt.Fprintln(a.out, reply)
	}
	return true
}

func (a *assistant) create(name string) {
	p, err := sandboxPath(a.dir, name)
	if err != nil {
		fmt.Fprintf(a.out, "refused: %v\n", err)
		return
	}
	f, err := os.OpenFile(p, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if os.IsExist(err) {
		fmt.Fprintf(a.out, "%s already exists in demo-files/ — left it exactly as it was.\n", name)
		return
	}
	if err != nil {
		fmt.Fprintf(a.out, "could not create %s: %v\n", name, err)
		return
	}
	f.Close()
	fmt.Fprintf(a.out, "created demo-files/%s\n", name)
}

func (a *assistant) remove(name string) {
	p, err := sandboxPath(a.dir, name)
	if err != nil {
		fmt.Fprintf(a.out, "refused: %v\n", err)
		return
	}
	if _, err := os.Stat(p); os.IsNotExist(err) {
		fmt.Fprintf(a.out, "%s isn't in demo-files/ — nothing to delete.\n", name)
		return
	}
	if !a.confirmFn(fmt.Sprintf("delete %s? [y/n] ", name)) {
		fmt.Fprintf(a.out, "ok — left %s alone.\n", name)
		return
	}
	if err := os.Remove(p); err != nil {
		fmt.Fprintf(a.out, "could not delete %s: %v\n", name, err)
		return
	}
	fmt.Fprintf(a.out, "deleted demo-files/%s\n", name)
}

func (a *assistant) list() {
	entries, err := os.ReadDir(a.dir)
	if err != nil {
		fmt.Fprintf(a.out, "could not read demo-files/: %v\n", err)
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
		fmt.Fprintln(a.out, "demo-files/ is empty.")
		return
	}
	fmt.Fprintf(a.out, "demo-files/: %s\n", strings.Join(names, ", "))
}

func main() {
	chatFlag := flag.String("chat", "brain.gob", "path to the chat brain (.gob trained on examples/pairs.jsonl)")
	filesFlag := flag.String("files", "filebrain.gob", "path to the file brain (.gob trained on examples/filecommands.jsonl, with its .intents.json sidecar)")
	flag.Parse()

	chatBrain, err := brain.LoadBrain(*chatFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "assistant:", err)
		os.Exit(1)
	}
	// The chat half behaves exactly as cmd/ask does,
	// typo repair included.
	chatBrain.FixTypos = true

	fileBrain, err := brain.LoadBrain(*filesFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "assistant:", err)
		os.Exit(1)
	}
	if fileBrain.Intents == nil {
		fmt.Fprintln(os.Stderr, "assistant: the file brain has no intent classifier — train it on examples/filecommands.jsonl (rows with \"intent\" fields) first")
		os.Exit(1)
	}
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, "assistant:", err)
		os.Exit(1)
	}
	dir := filepath.Join(cwd, "demo-files")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "assistant:", err)
		os.Exit(1)
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	a := &assistant{
		dir:     dir,
		predict: fileBrain.Intents.Predict,
		chat: func(line string) (string, error) {
			return chatBrain.AnswerChecked(line, rng)
		},
		out: os.Stdout,
	}
	// The extractor: the file brain's seq2seq answer to
	// the same line, trained (by the output-carrying
	// rows) to be the bare filename. Checked sampling
	// first — the repo's standard decode policy, which
	// prefers a reply that carries a word from your line,
	// as the right filename does. Greedy is the fallback
	// when the checked answer is not a clean name.
	// (Wired exactly as in examples/filecmd.)
	a.extract = func(line string) (string, error) {
		checked, err := fileBrain.AnswerChecked(line, rng)
		if err == nil {
			if _, serr := sanitizeName(checked); serr == nil {
				return checked, nil
			}
		}
		return fileBrain.Answer(line)
	}
	a.confirmFn = func(prompt string) bool {
		fmt.Fprint(os.Stdout, prompt)
		if !sc.Scan() {
			return false
		}
		ans := strings.ToLower(strings.TrimSpace(sc.Text()))
		return ans == "y" || ans == "yes"
	}

	fmt.Println("assistant: talk to me — or ask me to create, delete, or list files in demo-files/. Type 'quit' to exit.")
	for {
		fmt.Print("> ")
		if !sc.Scan() {
			fmt.Println()
			return
		}
		if !a.handle(sc.Text()) {
			return
		}
	}
}
