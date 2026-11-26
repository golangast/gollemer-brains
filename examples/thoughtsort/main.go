// Command thoughtsort sorts a brain dump by MEANING.
//
// When your head is full of half-formed next actions, you
// dump a few lines of what is on your mind, and this
// program classifies each line's real intent:
//
//	quick_win   (kind: act)     small — minutes, do it now
//	deep_focus  (kind: act)     big — needs a real block of time
//	venting     (kind: express) saying it is the point; no action
//	deferring   (kind: park)    a decision to leave it for later
//
// The two levels are the repo's usual kind → intent shape
// (see the intent package). The KIND is the concept: a
// quick win and a deep-focus job are both "act", but they
// ask for completely different next steps; venting wants
// no action at all; deferring is itself a decision. The
// classifier was declared once, as data — examples/thoughts.jsonl
// carries ~50 first-person dump lines per intent — and new
// phrasings are guessed from the words they carry.
//
// There is nothing to train and no model file: the intent
// classifier is a small softmax that fits in milliseconds,
// so this program trains it IN MEMORY at startup from the
// data file and throws it away at exit. Lines the
// classifier is not confident about are reported honestly
// as "not sure" (it abstains below its threshold) — with,
// when it can, the KIND it thinks the line lives in, which
// is often still useful for sorting a dump.
//
// Type lines, get verdicts; type "sort" to see the whole
// dump grouped by intent — quick wins first, parked last —
// which is the actual payoff: your head, sorted. Or batch
// a file: -lines dump.txt prints one verdict per line and
// the grouped summary.
//
//	go run ./examples/thoughtsort
//	go run ./examples/thoughtsort -lines mydump.txt
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/intent"
)

// headings names each intent's group in the sorted dump,
// in the order the groups print. The order is the point:
// act on the quick wins, schedule the deep work, let the
// venting be heard, and stop carrying the parked ones.
var groupOrder = []struct {
	label   string
	heading string
}{
	{"quick_win", "Quick wins — act now, minutes each"},
	{"deep_focus", "Deep focus — needs a real block, schedule it"},
	{"venting", "Venting — no action, it just needed saying"},
	{"deferring", "Parked — decided: later"},
	{"", "Not sure — the classifier abstained on these"},
}

// verdict is one line's classification: the intent label
// ("" when the classifier abstained), its kind, and the
// confidence. When the label abstains, kindGuess carries
// the kind classifier's best neighbourhood, if any.
type verdict struct {
	line      string
	label     string
	kind      string
	conf      float64
	kindGuess string
	kindConf  float64
}

// sorter holds the in-memory classifier and the session's
// classified lines, in the order they were dumped.
type sorter struct {
	clf     *intent.Classifier
	history []verdict
}

func newSorter(pairs []data.Pair, threshold float64) (*sorter, time.Duration, error) {
	start := time.Now()
	clf, err := intent.Train(pairs, threshold)
	if err != nil {
		return nil, 0, err
	}
	return &sorter{clf: clf}, time.Since(start), nil
}

// classify reads one dumped line. A confident label brings
// its kind with it (the concept the label lives in); an
// abstention is reported as such, with the kind guess when
// the kind classifier has one.
func (s *sorter) classify(line string) verdict {
	label, conf := s.clf.Predict(line)
	v := verdict{line: line, label: label, conf: conf}
	if label != "" {
		v.kind = s.clf.KindOf(label)
		return v
	}
	v.kindGuess, v.kindConf = s.clf.PredictKind(line)
	return v
}

// dump classifies a line and remembers it for the sort.
func (s *sorter) dump(line string) verdict {
	v := s.classify(line)
	s.history = append(s.history, v)
	return v
}

// printVerdict renders one verdict the way the REPL shows
// it: the meaning, its kind, and the confidence — or the
// honest abstention.
func printVerdict(w io.Writer, v verdict) {
	if v.label != "" {
		fmt.Fprintf(w, "  -> %s · kind: %s · %.2f\n", v.label, v.kind, v.conf)
		return
	}
	if v.kindGuess != "" {
		fmt.Fprintf(w, "  -> not sure (best label %.2f, under the bar) — kind looks like %s (%.2f)\n", v.conf, v.kindGuess, v.kindConf)
		return
	}
	fmt.Fprintf(w, "  -> not sure (best label %.2f, under the bar)\n", v.conf)
}

// printSort renders the remembered dump grouped by intent.
func (s *sorter) printSort(w io.Writer) {
	if len(s.history) == 0 {
		fmt.Fprintln(w, "nothing dumped yet — type a few lines of what's on your mind first.")
		return
	}
	groups := map[string][]verdict{}
	for _, v := range s.history {
		groups[v.label] = append(groups[v.label], v)
	}
	for _, g := range groupOrder {
		vs := groups[g.label]
		if len(vs) == 0 {
			continue
		}
		fmt.Fprintf(w, "\n%s (%d):\n", g.heading, len(vs))
		for _, v := range vs {
			fmt.Fprintf(w, "  - %s\n", v.line)
		}
	}
}

func run() error {
	dataFlag := flag.String("data", "examples/thoughts.jsonl", "path to the labeled dump lines the classifier learns from")
	linesFlag := flag.String("lines", "", "batch mode: classify every line of this file, print the grouped summary, and exit")
	flag.Parse()

	pairs, err := data.LoadPairs(*dataFlag)
	if err != nil {
		return err
	}
	s, took, err := newSorter(pairs, intent.DefaultThreshold)
	if err != nil {
		return err
	}

	if *linesFlag != "" {
		return runBatch(s, *linesFlag, os.Stdout)
	}

	counts := intent.CountByLabel(pairs)
	labels := make([]string, 0, len(counts))
	for l := range counts {
		labels = append(labels, l)
	}
	sort.Strings(labels)
	var parts []string
	for _, l := range labels {
		parts = append(parts, fmt.Sprintf("%s ×%d (kind %s)", l, counts[l], s.clf.KindOf(l)))
	}
	fmt.Println("thoughtsort: dump what's on your mind, one thought per line;")
	fmt.Println("i will sort each line by its real intent — quick win, deep focus, venting, or deferring.")
	fmt.Printf("classifier trained in memory from %s in %s (%s). nothing was saved.\n", *dataFlag, took.Round(time.Millisecond), strings.Join(parts, ", "))
	fmt.Println("type 'sort' to see the whole dump grouped, 'quit' to exit.")

	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for {
		fmt.Print("> ")
		if !sc.Scan() {
			fmt.Println()
			return nil
		}
		line := strings.TrimSpace(sc.Text())
		switch line {
		case "":
			continue
		case "quit", "exit":
			return nil
		case "sort":
			s.printSort(os.Stdout)
			continue
		}
		printVerdict(os.Stdout, s.dump(line))
	}
}

// runBatch classifies every non-blank line of a file and
// finishes with the grouped summary — the REPL's sort,
// for a dump you already have written down.
func runBatch(s *sorter, path string, w io.Writer) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		fmt.Fprintln(w, line)
		printVerdict(w, s.dump(line))
	}
	if err := sc.Err(); err != nil {
		return err
	}
	s.printSort(w)
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "thoughtsort:", err)
		os.Exit(1)
	}
}
