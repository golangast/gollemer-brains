package main

import (
	"fmt"
	"sort"
	"strings"
	"unicode"

	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/intent"
	"github.com/golangast/gollemer-brains/tokenizer"
)

// The factorized fallback read. The intent classifier's
// label read abstains below its threshold, and since the
// gist build an abstention also means no gist — the line
// falls to free generation and its garble. But the label
// NAMES factorize: an action prefix and an object part
// (create_file, explain_goroutine, tell_story), and the
// classifier already predicts the label's KIND separately.
// When the label read abstains, this composes the two
// factors it can still see — the predicted kind, and the
// object words the line itself carries — and adopts a
// label only when exactly one known meaning matches both.
// Zero or several candidates: the abstention stands and
// behaviour is unchanged. A confident primary read is
// never overridden; the fallback is consumed at exactly
// the point the primary read is (see readIntent).
//
// The kind tree (round 4) makes the label→kind parent
// relationship the composition walks EXPLICIT, and adds
// the parent walk: when composition inside the predicted
// kind finds NO candidate — the kind read may itself
// have abstained, or pointed at a neighbourhood the
// line's objects do not live in — the walk starts from
// the classifier's narrow read instead: its top label,
// abstained or not, up to that label's PARENT kind, and
// composition runs once more among the parent's children.
// A weak narrow read thus inherits its parent kind's
// candidates systematically instead of the composition
// depending on the kind guess alone. Two levels only —
// label and kind — by design: no deeper hierarchy, no
// transitive closure. Ambiguity is final at either level:
// a step that finds two or more candidates abstains
// outright, and the walk never rescues an ambiguous step
// by trying another kind.

// labelFactor is one label's factorization: its kind and
// the object tokens of its name (the tokens beyond the
// action prefix).
type labelFactor struct {
	kind    string
	objects []string
}

// factorTable is the factorization of the classifier's
// whole label inventory, derived from the classifier
// itself (Labels + LabelKinds — the corpus's labels and
// label→kind map, learned by intent.Train from the same
// pairs the read serves). One source of truth: the table
// can never name a label or kind the classifier lacks.
type factorTable struct {
	byLabel map[string]labelFactor
	tree    *kindTree
}

// kindTree is the explicit two-level tree over the
// classifier's inventory: every label that carries a
// kind has that kind as its PARENT, and every kind's
// CHILDREN are its labels. It is derived from the same
// LabelKinds map the factors come from — one source of
// truth, so the tree can never disagree with the
// classifier about which neighbourhood a meaning lives
// in. Children are sorted: the walk's candidate order
// is fixed, never map order.
type kindTree struct {
	parent   map[string]string   // label -> kind
	children map[string][]string // kind -> sorted labels
}

// deriveKindTree builds the tree from a label→kind map.
func deriveKindTree(kinds map[string]string) *kindTree {
	t := &kindTree{
		parent:   make(map[string]string, len(kinds)),
		children: make(map[string][]string),
	}
	for label, kind := range kinds {
		t.parent[label] = kind
		t.children[kind] = append(t.children[kind], label)
	}
	for kind := range t.children {
		sort.Strings(t.children[kind])
	}
	return t
}

// parentOf returns a label's kind — the one step up the
// tree — or "" when the label carries no kind.
func (t *kindTree) parentOf(label string) string {
	return t.parent[label]
}

// deriveFactors builds the table. Labels with no object
// part (single-token names — greeting, feelings) and
// labels whose rows carried no kind are excluded: neither
// can ever be the unique composition of a kind and an
// object.
func deriveFactors(clf *intent.Classifier) *factorTable {
	kinds := clf.LabelKinds()
	t := &factorTable{
		byLabel: make(map[string]labelFactor, len(clf.Labels)),
		tree:    deriveKindTree(kinds),
	}
	for _, label := range clf.Labels {
		kind := kinds[label]
		parts := strings.Split(label, "_")
		if kind == "" || len(parts) < 2 {
			continue
		}
		t.byLabel[label] = labelFactor{kind: kind, objects: parts[1:]}
	}
	return t
}

// factors returns the demo's factor table, derived from
// its classifier on first use (tests wire the classifier
// after constructing the demo, so derivation is lazy).
func (d *demo) factors() *factorTable {
	if d.factorTbl == nil && d.clf != nil {
		d.factorTbl = deriveFactors(d.clf)
	}
	return d.factorTbl
}

// lineObjectWords splits a line into the words the object
// factor can see: the intent classifier's own tokenization
// (lowercased, tokens containing a letter), as a set.
func lineObjectWords(line string) map[string]bool {
	set := map[string]bool{}
	for _, tok := range tokenizer.Tokenize(strings.ToLower(line)) {
		for _, r := range tok {
			if unicode.IsLetter(r) {
				set[tok] = true
				break
			}
		}
	}
	return set
}

// objectPresent reports whether an object token is present
// in the line's words: verbatim, or as its ordinary plural
// surface form (the token + "s" — lines say "pointers" and
// "channels" where labels say pointer and channel). The
// fold is one-directional: line words are never
// singularized, so a line carrying "errors" matches both
// explain_errors (verbatim) and explain_error (fold) — two
// candidates, and the abstention stands. The fold can only
// ever push toward abstaining, never toward a lone guess.
func objectPresent(words map[string]bool, object string) bool {
	return words[object] || words[object+"s"]
}

// objectCousinPresent reports whether a COUSIN of the
// object token is present in the line's words, in the
// same verbatim-or-plural forms objectPresent accepts.
// The cousins are cmdmatch's synonym families — the one
// vocabulary of word kinship in the repo — so an unseen
// word for a known object ("directory" for folder,
// "document" for file, "alarm" for reminder) counts as
// the object it names. This is the kind affinity the
// fallback needed: the cousin alone adopts nothing —
// the candidate must still match the predicted KIND,
// and the exactly-one-candidate rule is untouched, so a
// cousin that opens a second candidate pushes toward
// abstaining exactly as the plural fold does. The
// families' deliberate separations carry the safety:
// file and folder are NOT cousins, so "delete the folder"
// still cannot compose delete_file.
func objectCousinPresent(words map[string]bool, object string) bool {
	for _, c := range cmdmatch.Cousins(object) {
		if words[c] || words[c+"s"] {
			return true
		}
	}
	return false
}

// composeIn runs the factor composition inside ONE kind:
// a label is a candidate when it is a child of the kind
// (its factor kind matches) and its WHOLE object set is
// present in the line (subset, not intersection —
// otherwise explain_buffered_channel would collide with
// explain_channel on every channel line). It returns the
// candidate and the candidate COUNT: the count is what
// separates "nothing here" (0 — the walk may look
// elsewhere) from "too much here" (2+ — ambiguity, which
// is final). Candidates are visited in sorted label
// order; the count makes the outcome order-independent
// anyway, but a fixed visit order keeps the code as
// deterministic as its result.
func (t *factorTable) composeIn(words map[string]bool, kind string) (string, int) {
	labels := make([]string, 0, len(t.byLabel))
	for label := range t.byLabel {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	found := ""
	n := 0
	for _, label := range labels {
		f := t.byLabel[label]
		if f.kind != kind {
			continue
		}
		ok := true
		for _, obj := range f.objects {
			if !objectPresent(words, obj) && !objectCousinPresent(words, obj) {
				ok = false
				break
			}
		}
		if !ok {
			continue
		}
		n++
		found = label
	}
	return found, n
}

// fallbackRead composes a read for a line the label read
// left behind. Step one is the original composition: the
// predicted kind plus the line's object words, inside the
// predicted kind's children. Step two is the parent walk
// (see the header): when step one found NO candidate —
// never when it found two or more — the classifier's top
// label names a parent kind, and composition runs once
// more inside that kind. Exactly one candidate at either
// step is adopted; any other outcome abstains.
func (d *demo) fallbackRead(line, kindGuess string) (string, bool) {
	t := d.factors()
	if t == nil {
		return "", false
	}
	words := lineObjectWords(line)
	if kindGuess != "" {
		label, n := t.composeIn(words, kindGuess)
		if n == 1 {
			d.tracef("fallback: kind guess %s + the line's objects compose %s (%s)", kindGuess, label, t.explainComposition(words, label))
			return label, true
		}
		if n > 1 {
			d.tracef("fallback: %d candidates inside kind %s — ambiguity is final, no walk", n, kindGuess)
			return "", false // ambiguity is final: no walk
		}
	}
	// The parent walk: the narrow read's top label, up
	// one level to its parent kind, composition among
	// the parent's children. A classifier whose ranking
	// is empty (no feature evidence) names no parent,
	// and a top label with no kind has none to walk to.
	if d.clf == nil {
		return "", false
	}
	ranked := d.clf.RankedLabels(line)
	if len(ranked) == 0 {
		return "", false
	}
	parent := t.tree.parentOf(ranked[0].Label)
	if parent == "" || parent == kindGuess {
		return "", false
	}
	label, n := t.composeIn(words, parent)
	if n == 1 {
		d.tracef("fallback: parent walk — top label %s (%.2f) up to parent kind %s, whose children compose %s (%s)", ranked[0].Label, ranked[0].Prob, parent, label, t.explainComposition(words, label))
		return label, true
	}
	return "", false
}

// explainComposition renders, for the inspect trace,
// how each of an adopted label's object words was found
// in the line: verbatim, as its plural surface form, or
// through a cousin link (naming the cousin the line
// actually carried). It re-derives what composeIn
// checked, for the one adopted label only.
func (t *factorTable) explainComposition(words map[string]bool, label string) string {
	f := t.byLabel[label]
	parts := make([]string, 0, len(f.objects))
	for _, obj := range f.objects {
		switch {
		case words[obj]:
			parts = append(parts, obj+" verbatim")
		case words[obj+"s"]:
			parts = append(parts, obj+" as plural "+obj+"s")
		default:
			via := ""
			for _, c := range cmdmatch.Cousins(obj) {
				if words[c] {
					via = c
					break
				}
				if words[c+"s"] {
					via = c + "s"
					break
				}
			}
			parts = append(parts, fmt.Sprintf("%s via cousin %s", obj, via))
		}
	}
	return strings.Join(parts, "; ")
}
