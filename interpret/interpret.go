// Package interpret is the library surface of
// gollemer-brains: the interpreter a program imports to
// read its OWN user's lines against meanings that user
// declared.
//
// Hand it two files — a declarations file (the
// commands.json shape: each meaning named once, with a
// plain description of what it does; see FORMATS.md) and
// a training-pairs file (pairs.jsonl: example lines
// carrying the meaning and kind they belong to) — and
// Open gives back an Interpreter. Interpret reads one
// line and returns what the line means in those terms:
// the declared command it matches (with its argument
// values lifted verbatim from the line), or the learned
// meaning (label) and its kind, or an abstention. The
// surface is three calls:
//
//	in, err := interpret.Open("commands.json", "pairs.jsonl")
//	reading := in.Interpret("copy the photos from jim to jake")
//	// reading.Command == "copy", reading.Slots["from"] == "jim"
//
// Explain returns the per-word provenance of a line
// against one declaration — which words reached it, and
// how (a defining word, an example word, a cousin link,
// a typo repair) — and every Reading carries a Trace of
// the decisions that produced it, in plain lines. That
// is the debugging path: when a line misreads, the trace
// names the link or rule responsible, and the fix is a
// declaration or a training row, not code.
//
// The honest limit, stated plainly: this is a
// BOUNDED-domain interpreter. It interprets a user's
// lines well exactly to the extent that user has
// declared the meanings in them — the descriptions they
// wrote, the example lines they supplied. It does not
// read arbitrary text and understand it. A line outside
// the declared domain abstains, by design: a wrong
// answer is worse than none.
//
// The precedence Interpret applies is the one the
// repo's own programs use: the matcher (declared
// commands) reads first, because a declaration is the
// owner's own words for their own thing; the intent
// classifier's read is computed alongside and reported
// either way, so a caller can see both. Inside the
// intent read, a confident label stands unless the
// disjointness check withholds it (a winner torn
// against an opposite-kind rival within a hair); an
// abstention gets one composed second chance from the
// factorized fallback (factor.go), and no more.
//
// Everything here is deterministic: the same files and
// the same line produce the same Reading in every
// process. Standard library only, like the rest of the
// repo.
package interpret

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/intent"
)

// Interpreter is a loaded pair of meaning sources — a
// declarations registry and an intent classifier — read
// as one. Either source may be absent (a declarations-
// only interpreter matches commands and abstains on
// meanings; a corpus-only one reads meanings and matches
// no commands), but not both.
type Interpreter struct {
	reg     *cmdmatch.Registry
	matcher *cmdmatch.Matcher
	clf     *intent.Classifier
	factors *FactorTable
}

// Open loads an interpreter from files: a declarations
// file (commands.json shape) and/or a training-pairs
// file (pairs.jsonl shape). Either path may be "" to
// omit that source; both empty is an error. The intent
// classifier is trained in memory from the pairs, at the
// classifier's standing threshold — the same wiring
// examples/customcmd starts with. A pairs file whose
// rows carry no intents trains no classifier and is not
// an error: the interpreter simply has no meaning read.
func Open(declarationsPath, pairsPath string) (*Interpreter, error) {
	if declarationsPath == "" && pairsPath == "" {
		return nil, fmt.Errorf("interpret: no sources — give a declarations file, a pairs file, or both")
	}
	var reg *cmdmatch.Registry
	if declarationsPath != "" {
		r, err := cmdmatch.LoadRegistry(declarationsPath)
		if err != nil {
			return nil, err
		}
		reg = r
	}
	var clf *intent.Classifier
	if pairsPath != "" {
		pairs, err := data.LoadPairs(pairsPath)
		if err != nil {
			return nil, err
		}
		if len(intent.Rows(pairs)) > 0 {
			c, err := intent.Train(pairs, intent.DefaultThreshold)
			if err != nil {
				return nil, err
			}
			clf = c
		}
	}
	return New(reg, clf), nil
}

// New builds an interpreter from already-loaded sources
// — a registry and/or a trained classifier (one loaded
// from a brain's sidecar, say, via intent.Load, or
// brain.LoadBrain's Intents). Either may be nil, but
// not both; New panics on two nils, a programming error
// Open cannot produce.
func New(reg *cmdmatch.Registry, clf *intent.Classifier) *Interpreter {
	if reg == nil && clf == nil {
		panic("interpret: New needs a registry, a classifier, or both")
	}
	in := &Interpreter{reg: reg, clf: clf}
	if reg != nil {
		in.matcher = cmdmatch.NewMatcher(reg)
	}
	if clf != nil {
		in.factors = DeriveFactors(clf)
	}
	return in
}

// Reading is one line's interpretation: everything the
// interpreter could honestly say about it, each part
// empty when its source abstained.
//
// Command is the declared command the line matches (a
// declaration's Name), with CommandScore its match
// score and Slots the argument values lifted verbatim
// from the line for that command's declared slots
// (span-copy: a value is a substring of the line, word
// for word, never a generated guess; a slot the line
// did not seat is simply absent). A caller binds the
// command name to its own code — the interpreter
// supplies the meaning, never the action.
//
// Label is the intent classifier's read: the meaning
// the line belongs to, LabelConf its confidence, Kind
// that meaning's kind. Fallback marks a label composed
// by the factorized fallback after the primary read
// abstained. Withheld names a confident label the
// disjointness check set aside (WithheldRival is the
// opposite-kind rival it was torn against); a withheld
// read continues as an abstention, so Label may still
// be filled — by the fallback — or stay empty.
//
// KindGuess is the kind classifier's own read of the
// line, independent of the label: the neighbourhood the
// line lives in, available even when no label is.
//
// Trace is the provenance: one plain line per decision
// that produced this Reading, in the order the
// decisions were taken (matcher, intent read,
// disjointness, fallback). It is the inspect surface —
// the same lines examples/customcmd prints under its
// -inspect flag.
type Reading struct {
	Line string

	Command      string
	CommandScore float64
	Slots        map[string]string

	Label     string
	LabelConf float64
	Kind      string
	Fallback  bool

	Withheld      string
	WithheldRival string

	KindGuess string
	KindConf  float64

	Trace []string
}

// Interpret reads one line and returns its Reading. It
// never fails: a line nothing can read returns a
// Reading with every field empty but Line and Trace —
// the abstention is the answer, and the Trace says why.
func (in *Interpreter) Interpret(line string) Reading {
	r := Reading{Line: line}
	if in.matcher != nil {
		if decl, score, ok := in.matcher.Match(line); ok {
			r.Command = decl.Name
			r.CommandScore = score
			r.Slots = SpanBind(decl, line)
			r.Trace = append(r.Trace, in.traceMatch(line, decl, score))
		} else {
			ranked := in.matcher.Rank(line)
			if len(ranked) > 0 {
				r.Trace = append(r.Trace, fmt.Sprintf("matcher: no script (best %s %.2f, below the bars)", ranked[0].Script.Name, ranked[0].Score))
			}
		}
	}
	if in.clf != nil {
		in.readIntent(line, &r)
	}
	return r
}

// readIntent fills r's intent fields from the
// classifier, following the read's standing precedence:
// the primary read, the disjointness withhold, the kind
// guess, and — on an abstention — the factorized
// fallback. The trace lines are the ones
// examples/customcmd's inspect mode prints for the same
// decisions; the two surfaces read a line the same way
// because they are the same code.
func (in *Interpreter) readIntent(line string, r *Reading) {
	clf := in.clf
	r.Label, r.LabelConf = clf.Predict(line)
	if r.Label != "" {
		r.Kind = clf.KindOf(r.Label)
		r.Trace = append(r.Trace, fmt.Sprintf("intent: %s (%.2f), kind %s", r.Label, r.LabelConf, r.Kind))
		// The disjointness validation: a confident read
		// torn against a disjoint-kind rival within the
		// near-tie gap is withheld — the line continues
		// as an abstention below, fallback included.
		ranked := clf.RankedLabels(line)
		if len(ranked) > 0 && ranked[0].Label == r.Label {
			if rival, ok := DisjointWithhold(clf.KindOf, ranked); ok {
				r.Withheld = r.Label
				r.WithheldRival = rival
				r.Trace = append(r.Trace, fmt.Sprintf("disjointness: withheld %s (%s, %.2f) — rival %s (%s, %.2f) is within %.2f and the kinds are opposites; the read is set aside, not blended",
					r.Label, r.Kind, r.LabelConf, rival, clf.KindOf(rival), ranked[1].Prob, DisjointGap))
				r.Label = ""
				r.Kind = ""
			}
		}
	} else {
		if ranked := clf.RankedLabels(line); len(ranked) > 0 {
			r.Trace = append(r.Trace, fmt.Sprintf("intent: abstained (best %s %.2f, below %.2f)", ranked[0].Label, ranked[0].Prob, clf.Threshold))
		} else {
			r.Trace = append(r.Trace, "intent: abstained (no feature evidence)")
		}
	}
	r.KindGuess, r.KindConf = clf.PredictKind(line)
	if r.KindGuess != "" {
		r.Trace = append(r.Trace, fmt.Sprintf("intent: kind guess %s (%.2f)", r.KindGuess, r.KindConf))
	}
	if r.Label == "" {
		if label, ok, trace := in.factors.FallbackRead(clf, line, r.KindGuess); ok {
			r.Label = label
			r.Kind = clf.KindOf(label)
			r.Fallback = true
			r.Trace = append(r.Trace, trace)
		} else {
			if trace != "" {
				r.Trace = append(r.Trace, trace)
			}
			r.Trace = append(r.Trace, "fallback: no unique composition — the abstention stands")
		}
	}
}

// traceMatch renders the matcher provenance of a matched
// command: the words that reached the winning profile,
// and how each one reached it — the same lines
// examples/customcmd's inspect mode prints.
func (in *Interpreter) traceMatch(line string, decl cmdmatch.Declaration, score float64) string {
	var parts []string
	for _, hit := range in.matcher.Explain(line, decl.Name) {
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
	return fmt.Sprintf("matcher: %s (%.2f) — %s", decl.Name, score, strings.Join(parts, "; "))
}

// Explain returns the per-word provenance of a line
// against one named declaration: which of the line's
// words that declaration's profile knows, and how it
// knows them (see cmdmatch.WordHit). An unknown name —
// or an interpreter with no declarations — explains
// nothing (nil). This is the adopter's debugging path
// for a command that matched when it should not have,
// or did not when it should.
func (in *Interpreter) Explain(line, commandName string) []cmdmatch.WordHit {
	if in.matcher == nil {
		return nil
	}
	return in.matcher.Explain(line, commandName)
}

// Commands returns the declarations the interpreter can
// match, in registry order (nil when it has none).
func (in *Interpreter) Commands() []cmdmatch.Declaration {
	if in.reg == nil {
		return nil
	}
	return in.reg.Scripts
}

// Labels returns the meaning labels the interpreter's
// classifier can read, sorted (nil when it has none).
func (in *Interpreter) Labels() []string {
	if in.clf == nil {
		return nil
	}
	return in.clf.Labels
}

// KindOf returns the kind a label belongs to in the
// interpreter's tree, or "" when the label carries none
// (or is unknown, or there is no classifier).
func (in *Interpreter) KindOf(label string) string {
	if in.clf == nil {
		return ""
	}
	return in.clf.KindOf(label)
}
