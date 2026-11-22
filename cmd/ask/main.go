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
	flag.Parse()

	b, err := brain.LoadBrain(*modelFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ask:", err)
		os.Exit(1)
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
	}
}
