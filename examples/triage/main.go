// Command triage is the packaging round's foreign-domain
// example: a support-ticket interpreter for a small
// software help desk — a domain that is not the repo
// owner's world, included to prove the brains package
// interprets OTHER people's content.
//
// Everything it knows is declared in this folder, in the
// two files that are the whole contract (see FORMATS.md
// at the repo root):
//
//   - commands.json declares the desk's ACTIONS once
//     each — a name, the desk tool's command, a plain
//     description, its argument slots. The matcher reads
//     lines against the descriptions at runtime; nothing
//     is trained per action.
//   - pairs.jsonl teaches the MEANINGS: example customer
//     lines, each carrying the meaning (intent) it
//     belongs to and the kind that meaning lives in —
//     this domain's own kind tree (route, answer,
//     escalate, resolve, track), not the main corpus's
//     nine kinds. brain.gob is the trained brain for
//     those pairs, trained with the one command below
//     and committed, so the example runs from a pull:
//
//	go run ./cmd/train -dir examples/triage
//	go run ./examples/triage
//
// The program INTERPRETS: type a customer line and it
// says what the line means here — the declared action it
// matches (with its argument values lifted verbatim from
// the line), or the learned meaning and its kind, or an
// honest abstention. It runs nothing: binding an action
// name to real code is the desk's own business, exactly
// as the interpret package's contract says. -inspect
// prints the provenance of each reading — which words,
// cousin links, and rules produced it — the debugging
// path when a line misreads.
//
// The honest limit, on the face of the example: this is
// a bounded domain. It interprets customer lines well
// exactly to the extent the desk has declared the
// meanings in them — the descriptions and example lines
// in this folder. It does not read arbitrary text and
// understand it; a line outside the declared domain
// abstains, by design.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/golangast/gollemer-brains/brain"
	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/interpret"
)

func main() {
	commandsFlag := flag.String("commands", "examples/triage/commands.json", "path to the declarations file")
	modelFlag := flag.String("model", "examples/triage/brain.gob", "path to the trained brain (its intent classifier reads the meanings)")
	inspectFlag := flag.Bool("inspect", false, "print each reading's provenance trace")
	flag.Parse()

	reg, err := cmdmatch.LoadRegistry(*commandsFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "triage:", err)
		os.Exit(1)
	}
	b, err := brain.LoadBrain(*modelFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "triage:", err)
		os.Exit(1)
	}
	if b.Intents == nil {
		fmt.Fprintln(os.Stderr, "triage: the brain carries no intent classifier — train it on pairs with \"intent\" fields (go run ./cmd/train -dir examples/triage)")
		os.Exit(1)
	}
	in := interpret.New(reg, b.Intents)

	fmt.Println("triage: type a customer line and I will say what it means here.")
	fmt.Println("I only know the meanings declared in this folder's commands.json and pairs.jsonl —")
	fmt.Println("a line outside that domain gets an honest abstention, not a guess.")
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for {
		fmt.Print("> ")
		if !sc.Scan() {
			fmt.Println()
			return
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if line == "quit" || line == "exit" {
			return
		}
		printReading(in, line, *inspectFlag)
	}
}

// printReading renders one line's interpretation: the
// action it matches, or the meaning it reads, or the
// abstention — and, under -inspect, the provenance.
func printReading(in *interpret.Interpreter, line string, inspect bool) {
	r := in.Interpret(line)
	switch {
	case r.Command != "":
		decl, _ := findDecl(in, r.Command)
		fmt.Printf("action: %s (%.2f) — %s\n", r.Command, r.CommandScore, decl.Description)
		if len(decl.Slots) > 0 {
			var parts []string
			for _, slot := range decl.Slots {
				if v, ok := r.Slots[slot]; ok {
					parts = append(parts, slot+"="+v)
				} else {
					parts = append(parts, slot+"=(not heard)")
				}
			}
			fmt.Printf("  slots: %s\n", strings.Join(parts, ", "))
		}
		fmt.Printf("  bound command: %s\n", boundCommand(decl, r.Slots))
	case r.Label != "":
		note := ""
		if r.Fallback {
			note = " (composed by the fallback: kind + the line's object words)"
		}
		fmt.Printf("meaning: %s (kind %s, %.2f)%s\n", r.Label, r.Kind, r.LabelConf, note)
	case r.Withheld != "":
		fmt.Printf("no read: the classifier leaned %s but was torn against %s — withheld, not blended\n", r.Withheld, r.WithheldRival)
	case r.KindGuess != "":
		fmt.Printf("no read: the neighbourhood looks like kind %s (%.2f), but no declared meaning fits\n", r.KindGuess, r.KindConf)
	default:
		fmt.Println("no read: outside the declared domain")
	}
	if inspect {
		for _, t := range r.Trace {
			fmt.Printf("  [inspect] %s\n", t)
		}
	}
}

func findDecl(in *interpret.Interpreter, name string) (cmdmatch.Declaration, bool) {
	for _, d := range in.Commands() {
		if d.Name == name {
			return d, true
		}
	}
	return cmdmatch.Declaration{}, false
}

// boundCommand renders the declaration's command line
// with the seated slot values appended in declared
// order — the exact line the desk's own code would run
// if it binds this action. The example only prints it.
func boundCommand(decl cmdmatch.Declaration, slots map[string]string) string {
	var values []string
	for _, slot := range decl.Slots {
		if v, ok := slots[slot]; ok {
			values = append(values, v)
		}
	}
	if len(values) == 0 {
		return decl.Command
	}
	return decl.Command + " " + strings.Join(values, " ")
}
