package main

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Provenance, as an inspect mode (round 4). Every route
// a line takes is decided by named mechanisms — a word
// matched a profile, a cousin link carried a word, the
// classifier read a label, the fallback walked to a
// parent kind, a disjointness check withheld a torn
// read — and with inspect OFF none of that is visible:
// answers stand alone, per the standing chrome rules.
// With inspect ON (-inspect), each decision point on
// the line's path appends one plain trace line, printed
// after the line's ordinary output under an [inspect]
// prefix: which cousin link, which kind-tree step,
// which rule produced the result, so a match is
// something its owner can inspect instead of a
// black-box guess. The traces are the ONLY difference:
// inspect off collects nothing and prints nothing, and
// the ordinary output is byte-identical either way.

// tracef records one provenance line for the input line
// being handled. It is a no-op unless inspect is on, so
// the decision points can call it unconditionally.
func (d *demo) tracef(format string, args ...any) {
	if !d.inspect {
		return
	}
	d.trace = append(d.trace, fmt.Sprintf(format, args...))
}

// flushInspect prints the collected trace lines after a
// handled line's ordinary output and clears them. With
// inspect off it prints nothing.
func (d *demo) flushInspect() {
	if !d.inspect {
		return
	}
	for _, t := range d.trace {
		fmt.Fprintf(d.out, "  [inspect] %s\n", t)
	}
	d.trace = nil
}

// traceMatch records the matcher provenance of a
// presented match: the words that reached the winning
// profile, and how each one reached it (a defining
// word, an example word, a cousin link naming its
// source, a typo repair naming what was scored).
func (d *demo) traceMatch(line string, decl cmdmatch.Declaration, score float64) {
	if !d.inspect {
		return
	}
	var parts []string
	for _, hit := range d.matcher.Explain(line, decl.Name) {
		if hit.Via == "" {
			continue
		}
		word := hit.Word
		if hit.ScoredAs != hit.Word {
			word = fmt.Sprintf("%s (scored as %s, typo repair)", hit.Word, hit.ScoredAs)
		}
		switch hit.Via {
		case "cousin":
			parts = append(parts, fmt.Sprintf("%s: cousin of %s", word, hit.Source))
		default:
			parts = append(parts, fmt.Sprintf("%s: %s", word, hit.Via))
		}
	}
	d.tracef("matcher: %s (%.2f) — %s", decl.Name, score, strings.Join(parts, "; "))
}

// traceMatcherAbstain records the matcher provenance of
// an abstention: the best script and its score, so an
// inspector can see how close the line came.
func (d *demo) traceMatcherAbstain(ranked []cmdmatch.Scored) {
	if !d.inspect || len(ranked) == 0 {
		return
	}
	d.tracef("matcher: no script (best %s %.2f, below the bars)", ranked[0].Script.Name, ranked[0].Score)
}
