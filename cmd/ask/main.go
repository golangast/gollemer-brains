// Command ask is a minimal demo REPL for a trained brain:
// type a line, get the brain's answer. Type "quit" (or send
// EOF) to exit.
//
//	go run ./cmd/ask -model brain.gob
//
// Answers use checked sampling: the greedy reply stands when
// it carries one of your line's content words; otherwise a
// few sampled replies are tried and the first on-topic one
// wins (see brain.AnswerChecked). This is a demo of the
// brain, NOT the Gollemer chat — no memory, no routing, no
// tools.
//
// With -intents, each reply is followed by the meaning the
// brain's intent classifier predicts for your line (when the
// brain was trained on data with "intent" fields) — the
// label a program would switch on to run a command.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/golangast/gollemer-brains/brain"
)

func main() {
	modelFlag := flag.String("model", "brain.gob", "path to the trained brain (.gob)")
	intentsFlag := flag.Bool("intents", false, "after each reply, print the predicted intent (needs a brain trained on data with \"intent\" fields)")
	flag.Parse()

	b, err := brain.LoadBrain(*modelFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ask:", err)
		os.Exit(1)
	}
	if *intentsFlag && b.Intents == nil {
		fmt.Fprintln(os.Stderr, "ask: -intents given, but this brain has no intent classifier — train on data with \"intent\" fields to add one")
	}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	fmt.Println("brain loaded. type a line, or 'quit' to exit.")
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
		ans, err := b.AnswerChecked(line, rng)
		if err != nil {
			fmt.Fprintln(os.Stderr, "ask:", err)
			continue
		}
		fmt.Println(ans)
		if *intentsFlag && b.Intents != nil {
			fmt.Printf("intent: %s\n", b.Intents.Describe(line))
		}
	}
}
