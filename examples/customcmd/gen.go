//go:build ignore

// Generator for examples/genericcommands.jsonl, the
// training file for customcmd's extractor brain. Run from
// the repo root:
//
//	go run examples/customcmd/gen.go
//
// The file it writes is deliberately GENERIC: every request
// in it is for an invented script that does not exist
// (transfer-ledger, sort-recipes, pin-notice, ...), in
// domains the shipped registry never touches. What the
// extractor brain learns from it is sentence SHAPES —
// "from X to Y", "in the X folder", "called X", "on X",
// "into X" — and the output for each row is exactly the
// shape's values, in order. The registry's real scripts
// (examples/commands.json) are never read by this program
// and never appear in its output: that separation is the
// design's guarantee, and a test in examples/customcmd
// pins it.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
)

// values is the pool of slot values the shapes are taught
// over: person names and folder-ish words. A real user's
// values (their folder names) mostly will not be in this
// pool — transcribing unseen values is the extractor's
// known weak edge, and the held-out measurement of the
// example says so with numbers. Deliberately absent: the
// registry seed-test-data script's env values (staging,
// test, prod, dev) — no shipped training should foresee a
// particular user's slot vocabulary.
var values = []string{
	"jim", "jake", "alex", "sam", "maria", "anna", "peter",
	"lucy", "mark", "emma", "paul", "nina", "owen", "ruby",
	"adam", "ella", "hugo", "iris", "karl", "lena", "milo",
	"nora", "omar", "sara", "tom", "vera", "wade", "zoe",
	"holidays", "birthdays", "trips", "family", "school",
	"taxes", "wedding", "summer", "winter", "attic",
	"cellar", "pantry", "garage", "studio", "office",
	"kitchen", "garden", "library", "vault",
}

// task is one invented script's worth of frames: a verb and
// noun to talk about, and the sentence frames of its shape.
// In each frame, {a} (and {b}) mark where a value goes.
type task struct {
	frames []string
}

var fromToTasks = []task{
	{frames: []string{
		"transfer the ledger from {a} to {b}",
		"transfer the ledger from the {a} folder to the {b} folder",
		"please transfer the ledger from {a} to {b}",
		"can you transfer the ledger from {a} to {b}",
		"transfer my ledger from {a} over to {b}",
		"move the ledger in {a} over to {b}",
	}},
	{frames: []string{
		"copy the archive from {a} to {b}",
		"copy the archive from the {a} folder to the {b} folder",
		"please copy the archive from {a} to {b}",
		"can you copy the archive from {a} to {b}",
		"copy my archive from {a} over to {b}",
		"send the archive in {a} over to {b}",
	}},
	{frames: []string{
		"move the workshop from {a} to {b}",
		"move the workshop from the {a} folder to the {b} folder",
		"please move the workshop from {a} to {b}",
		"can you move the workshop from {a} to {b}",
		"move my workshop from {a} over to {b}",
		"shift the workshop in {a} over to {b}",
	}},
	{frames: []string{
		"send the parcel from {a} to {b}",
		"send the parcel from the {a} folder to the {b} folder",
		"please send the parcel from {a} to {b}",
		"can you send the parcel from {a} to {b}",
		"send my parcel from {a} over to {b}",
		"carry the parcel in {a} over to {b}",
	}},
	{frames: []string{
		"shift the toolbox from {a} to {b}",
		"shift the toolbox from the {a} folder to the {b} folder",
		"please shift the toolbox from {a} to {b}",
		"can you shift the toolbox from {a} to {b}",
		"shift my toolbox from {a} over to {b}",
		"move the toolbox in {a} over to {b}",
	}},
}

var inTasks = []task{
	{frames: []string{
		"sort the recipes in {a}",
		"sort the recipes in the {a} folder",
		"sort the recipes inside {a}",
		"please sort the recipes in {a}",
		"can you sort the recipes in {a}",
		"sort my recipes in the {a} box",
	}},
	{frames: []string{
		"tag the songs in {a}",
		"tag the songs in the {a} folder",
		"tag the songs inside {a}",
		"please tag the songs in {a}",
		"can you tag the songs in {a}",
		"tag my songs in the {a} box",
	}},
	{frames: []string{
		"shelve the books in {a}",
		"shelve the books in the {a} folder",
		"shelve the books inside {a}",
		"please shelve the books in {a}",
		"can you shelve the books in {a}",
		"shelve my books in the {a} box",
	}},
	{frames: []string{
		"stash the gear in {a}",
		"stash the gear in the {a} folder",
		"stash the gear inside {a}",
		"please stash the gear in {a}",
		"can you stash the gear in {a}",
		"stash my gear in the {a} box",
	}},
	{frames: []string{
		"file the invoices in {a}",
		"file the invoices in the {a} folder",
		"file the invoices inside {a}",
		"please file the invoices in {a}",
		"can you file the invoices in {a}",
		"file my invoices in the {a} box",
	}},
}

var calledTasks = []task{
	{frames: []string{
		"open the project called {a}",
		"open the project named {a}",
		"please open the project called {a}",
		"can you open the project named {a}",
		"open my project called {a}",
	}},
	{frames: []string{
		"find the document called {a}",
		"find the document named {a}",
		"please find the document called {a}",
		"can you find the document named {a}",
		"find my document called {a}",
	}},
	{frames: []string{
		"play the playlist called {a}",
		"play the playlist named {a}",
		"please play the playlist called {a}",
		"can you play the playlist named {a}",
		"play my playlist called {a}",
	}},
	{frames: []string{
		"print the report called {a}",
		"print the report named {a}",
		"please print the report called {a}",
		"can you print the report named {a}",
		"print my report called {a}",
	}},
	{frames: []string{
		"load the album called {a}",
		"load the album named {a}",
		"please load the album called {a}",
		"can you load the album named {a}",
		"load my album called {a}",
	}},
}

var onTasks = []task{
	{frames: []string{
		"pin the notice on {a}",
		"pin the notice on the {a} board",
		"please pin the notice on {a}",
		"can you pin the notice on {a}",
	}},
	{frames: []string{
		"post the notice on {a}",
		"post the notice on the {a} board",
		"please post the notice on {a}",
		"can you post the notice on {a}",
	}},
	{frames: []string{
		"mount the image on {a}",
		"mount the image on the {a} board",
		"please mount the image on {a}",
		"can you mount the image on {a}",
	}},
	{frames: []string{
		"stick the label on {a}",
		"stick the label on the {a} board",
		"please stick the label on {a}",
		"can you stick the label on {a}",
	}},
}

var intoTasks = []task{
	{frames: []string{
		"pour the ingredients into {a}",
		"pour the ingredients into the {a}",
		"please pour the ingredients into {a}",
		"can you pour the ingredients into {a}",
	}},
	{frames: []string{
		"pack the supplies into {a}",
		"pack the supplies into the {a}",
		"please pack the supplies into {a}",
		"can you pack the supplies into {a}",
	}},
	{frames: []string{
		"toss the scraps into {a}",
		"toss the scraps into the {a}",
		"please toss the scraps into {a}",
		"can you toss the scraps into {a}",
	}},
}

type row struct {
	Input  string `json:"input"`
	Output string `json:"output"`
}

func fill(frame string, vals ...string) string {
	out := frame
	slots := []string{"{a}", "{b}"}
	for i, v := range vals {
		out = strings.ReplaceAll(out, slots[i], v)
	}
	return out
}

func main() {
	outFlag := flag.String("out", "examples/genericcommands.jsonl", "where to write the generated rows")
	seedFlag := flag.Int64("seed", 20261125, "sampling seed (fixed: the shipped file is reproducible)")
	flag.Parse()

	rng := rand.New(rand.NewSource(*seedFlag))
	seen := map[string]bool{}
	var rows []row
	add := func(input, output string) {
		if seen[input] {
			return
		}
		seen[input] = true
		rows = append(rows, row{Input: input, Output: output})
	}
	pick := func() string { return values[rng.Intn(len(values))] }
	pickPair := func() (string, string) {
		a := pick()
		b := pick()
		for b == a {
			b = pick()
		}
		return a, b
	}

	// Two-value shapes: 10 sampled pairs per frame.
	for _, t := range fromToTasks {
		for _, f := range t.frames {
			for i := 0; i < 10; i++ {
				a, b := pickPair()
				add(fill(f, a, b), a+" "+b)
			}
		}
	}
	// One-value shapes: 12 sampled values per frame (10 for
	// the smaller on/into sets).
	for _, group := range []struct {
		tasks []task
		per   int
	}{
		{inTasks, 12},
		{calledTasks, 12},
		{onTasks, 10},
		{intoTasks, 10},
	} {
		for _, t := range group.tasks {
			for _, f := range t.frames {
				for i := 0; i < group.per; i++ {
					a := pick()
					add(fill(f, a), a)
				}
			}
		}
	}

	// Deterministic file order: sort by input, so the
	// shipped file is stable across regenerations.
	sort.Slice(rows, func(i, j int) bool { return rows[i].Input < rows[j].Input })

	var sb strings.Builder
	enc := json.NewEncoder(&sb)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			fmt.Fprintln(os.Stderr, "gen:", err)
			os.Exit(1)
		}
	}
	if err := os.WriteFile(*outFlag, []byte(sb.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "gen:", err)
		os.Exit(1)
	}
	fmt.Printf("gen: wrote %d rows to %s\n", len(rows), *outFlag)
}
