package main

import (
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/intent"
)

const corpusPath = "../../examples/thoughts.jsonl"

func trainCorpus(t *testing.T) *sorter {
	t.Helper()
	pairs, err := data.LoadPairs(corpusPath)
	if err != nil {
		t.Fatal(err)
	}
	s, _, err := newSorter(pairs, intent.DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// The shipped corpus must carry a real mass per intent —
// a meaning with a handful of rows is a wish, not a
// classifier.
func TestCorpusShape(t *testing.T) {
	pairs, err := data.LoadPairs(corpusPath)
	if err != nil {
		t.Fatal(err)
	}
	counts := intent.CountByLabel(pairs)
	for _, label := range []string{"quick_win", "deep_focus", "venting", "deferring"} {
		if counts[label] < 40 {
			t.Errorf("label %s has %d rows, want at least 40", label, counts[label])
		}
	}
	km, err := intent.KindMap(pairs)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"quick_win": "act", "deep_focus": "act", "venting": "express", "deferring": "park"}
	for label, kind := range want {
		if km[label] != kind {
			t.Errorf("label %s kind = %q, want %q", label, km[label], kind)
		}
	}
}

// Canonical dump lines land on their meanings, and the
// sort groups a mixed dump the way the headings promise.
func TestClassifyAndSort(t *testing.T) {
	s := trainCorpus(t)
	cases := map[string]string{
		"text jules back, it takes ten seconds":                              "quick_win",
		"the dissertation chapter needs whole weeks of writing":              "deep_focus",
		"the upstairs neighbours stomp around all night, i am so sick of it": "venting",
		"the loft clearout is a someday job, it can wait":                    "deferring",
	}
	for line, want := range cases {
		if v := s.dump(line); v.label != want {
			t.Errorf("classify %q = %q (%.2f), want %s", line, v.label, v.conf, want)
		}
	}
	var b strings.Builder
	s.printSort(&b)
	out := b.String()
	for _, heading := range []string{"Quick wins", "Deep focus", "Venting", "Parked"} {
		if !strings.Contains(out, heading) {
			t.Errorf("sort output missing heading %q:\n%s", heading, out)
		}
	}
}

// score prints one set's right/wrong/abstain table, per
// intent and overall, for labels and for kinds. It is the
// measurement harness behind the README numbers: the test
// asserts only that every line was scored, never a score —
// the numbers are reported, not gated.
func score(t *testing.T, s *sorter, path string) {
	t.Helper()
	pairs, err := data.LoadPairs(path)
	if err != nil {
		t.Fatal(err)
	}
	type tally struct{ right, wrong, abstain, kindRight, kindWrong, kindAbstain, kindGuessRight int }
	tallies := map[string]*tally{}
	var labels []string
	for _, p := range pairs {
		if _, ok := tallies[p.Intent]; !ok {
			tallies[p.Intent] = &tally{}
			labels = append(labels, p.Intent)
		}
		ty := tallies[p.Intent]
		v := s.classify(p.Input)
		switch {
		case v.label == "":
			ty.abstain++
			ty.kindAbstain++
		case v.label == p.Intent:
			ty.right++
			ty.kindRight++
		default:
			ty.wrong++
			if v.kind == p.Kind {
				ty.kindRight++
			} else {
				ty.kindWrong++
			}
		}
		if kg, _ := s.clf.PredictKind(p.Input); kg == p.Kind && kg != "" {
			ty.kindGuessRight++
		}
	}
	sort.Strings(labels)
	var tot tally
	fmt.Printf("  %-10s %5s %5s %7s | %9s %9s %11s | %s\n", "intent", "right", "wrong", "abstain", "kindRight", "kindWrong", "kindAbstain", "PredictKind right")
	for _, l := range labels {
		ty := tallies[l]
		fmt.Printf("  %-10s %5d %5d %7d | %9d %9d %11d | %d\n", l, ty.right, ty.wrong, ty.abstain, ty.kindRight, ty.kindWrong, ty.kindAbstain, ty.kindGuessRight)
		tot.right += ty.right
		tot.wrong += ty.wrong
		tot.abstain += ty.abstain
		tot.kindRight += ty.kindRight
		tot.kindWrong += ty.kindWrong
		tot.kindAbstain += ty.kindAbstain
		tot.kindGuessRight += ty.kindGuessRight
	}
	fmt.Printf("  %-10s %5d %5d %7d | %9d %9d %11d | %d  (of %d)\n", "TOTAL", tot.right, tot.wrong, tot.abstain, tot.kindRight, tot.kindWrong, tot.kindAbstain, tot.kindGuessRight, len(pairs))
}

func TestEvalHeldOut(t *testing.T) {
	pairs, err := data.LoadPairs(corpusPath)
	if err != nil {
		t.Fatal(err)
	}
	// Startup timing: median of 5 in-memory trainings.
	var times []time.Duration
	for i := 0; i < 5; i++ {
		_, took, err := newSorter(pairs, intent.DefaultThreshold)
		if err != nil {
			t.Fatal(err)
		}
		times = append(times, took)
	}
	sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
	fmt.Printf("startup (load excluded, intent.Train on %d rows): median %s of %v\n", len(pairs), times[2].Round(time.Millisecond), times)

	s := trainCorpus(t)
	fmt.Println("HELD-OUT (32 fresh lines, never trained on):")
	score(t, s, "testdata/heldout.jsonl")
}

func TestEvalBorder(t *testing.T) {
	s := trainCorpus(t)
	fmt.Println("BORDER (8 lines on the declared border rules):")
	score(t, s, "testdata/border.jsonl")
	// Print each border line's verdict: on the borders the
	// individual misses are the information.
	pairs, err := data.LoadPairs("testdata/border.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range pairs {
		v := s.classify(p.Input)
		fmt.Printf("  want %-10s got %-10s (%.2f)  %s\n", p.Intent, v.label, v.conf, p.Input)
	}
}
