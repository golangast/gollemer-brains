package main

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/brain"
	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/interpret"
)

// The held-out probe set (probes.jsonl): unseen phrasings
// of the declared meanings, action lines, and control
// lines from outside the domain entirely. No probe line
// appears in pairs.jsonl. The instrument reports
// right / wrong / abstain per class; the hard fence is
// the controls — a control read as anything is a WRONG
// answer to a line the desk never declared, and the
// failure mode of this system is abstention, never a
// confident guess. Non-control misses are reported in
// the test log (and in the round's build record), not
// hidden: they are the domain data's honest frontier.

type probe struct {
	Input  string `json:"input"`
	Expect string `json:"expect"` // a label, "cmd:<name>", or "" for abstain
}

func loadProbes(t *testing.T) []probe {
	t.Helper()
	f, err := os.Open("probes.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []probe
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var p probe
		if err := json.Unmarshal([]byte(line), &p); err != nil {
			t.Fatalf("bad probe line %q: %v", line, err)
		}
		out = append(out, p)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

func triageInterpreter(t *testing.T) *interpret.Interpreter {
	t.Helper()
	reg, err := cmdmatch.LoadRegistry("commands.json")
	if err != nil {
		t.Fatal(err)
	}
	b, err := brain.LoadBrain("brain.gob")
	if err != nil {
		t.Fatalf("the shipped trained brain must load from a pull: %v", err)
	}
	if b.Intents == nil {
		t.Fatal("the shipped brain carries no intent classifier")
	}
	return interpret.New(reg, b.Intents)
}

func TestHeldOutProbes(t *testing.T) {
	in := triageInterpreter(t)
	probes := loadProbes(t)
	var right, wrong, abstain int
	var controlWrong []string
	for _, p := range probes {
		r := in.Interpret(p.Input)
		got := r.Label
		if r.Command != "" {
			got = "cmd:" + r.Command
		}
		switch {
		case got == p.Expect:
			right++
		case got == "":
			abstain++
		default:
			wrong++
			if p.Expect == "" {
				controlWrong = append(controlWrong, p.Input+" -> "+got)
			}
			t.Logf("WRONG: %q -> %s, want %q", p.Input, got, p.Expect)
		}
		if got == "" && p.Expect != "" {
			t.Logf("ABSTAIN: %q, want %q", p.Input, p.Expect)
		}
	}
	t.Logf("probes: %d right / %d wrong / %d abstain of %d", right, wrong, abstain, len(probes))
	if len(controlWrong) > 0 {
		t.Errorf("control lines read as something — the failure mode must be abstention: %v", controlWrong)
	}
}

// TestForwardSlotsVerbatim: the span-copy seat through
// the library surface — the vendor line's values are
// lifted from the line itself, word for word.
func TestForwardSlotsVerbatim(t *testing.T) {
	in := triageInterpreter(t)
	r := in.Interpret("forward this from support to the vendor")
	if r.Command != "forward-to-vendor" {
		t.Fatalf("Command = %q, want forward-to-vendor", r.Command)
	}
	if r.Slots["from"] != "support" || r.Slots["to"] != "vendor" {
		t.Errorf("Slots = %v, want from=support to=vendor", r.Slots)
	}
}

// TestShippedBrainAnswers: the trained seq2seq half of
// the shipped brain loads and answers — the example's
// first run is never a training run.
func TestShippedBrainAnswers(t *testing.T) {
	b, err := brain.LoadBrain("brain.gob")
	if err != nil {
		t.Fatal(err)
	}
	if b.Intents == nil {
		t.Fatal("no intent classifier in the shipped brain")
	}
	label, conf := b.Intents.Predict("my invoice is wrong")
	if label != "route_billing" {
		t.Errorf("trained read of a corpus line = %q (%.2f), want route_billing", label, conf)
	}
}
