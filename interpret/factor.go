package interpret

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
// label read abstains below its threshold, and an
// abstention otherwise means no read at all. But the
// label NAMES factorize: an action prefix and an object
// part (create_file, explain_goroutine, tell_story), and
// the classifier already predicts the label's KIND
// separately. When the label read abstains, this composes
// the two factors it can still see — the predicted kind,
// and the object words the line itself carries — and
// adopts a label only when exactly one known meaning
// matches both. Zero or several candidates: the
// abstention stands. A confident primary read is never
// overridden.
//
// The kind tree makes the label→kind parent relationship
// the composition walks EXPLICIT, and adds the parent
// walk: when the kind read ABSTAINED — it named no
// neighbourhood at all — the walk starts from the
// classifier's narrow read instead: its top label,
// abstained or not, up to that label's PARENT kind, and
// composition runs among the parent's children. A weak
// narrow read thus inherits its parent kind's candidates
// systematically instead of the composition depending on
// the kind guess alone. The tree runs at most three
// levels — kind → sub-kind → label — and the walk climbs
// the label's whole chain, composing at each level; no
// deeper hierarchy, no transitive closure. Ambiguity is
// final: a step that finds two or more candidates
// abstains outright, and the walk never rescues an
// ambiguous step by trying another kind. (Sub-kind
// narrowing is the one refinement: ambiguity at KIND
// level may be resolved inside the narrow read's own
// sub-kind, a narrower question — see FallbackRead.)
//
// The walk fills a SILENCE, it does not override a
// statement: when the kind read spoke and its
// neighbourhood composed nothing, the walk does not
// cross into a different kind. Measure, not taste: the
// unscoped walk's one changed line across the probe
// corpora was "what does a timer do" — a find question
// (kind guess find 0.89) whose top label is set_timer;
// walking into schedule composed set_timer from the bare
// object "timer", a wrong adoption. With the scope, the
// walk fires only where the kind read is missing — the
// case it exists for.
//
// This machinery lived in examples/customcmd until the
// packaging round moved it here, under the programs that
// use it; customcmd calls it through thin adapters and
// its behaviour is unchanged.

// LabelFactor is one label's factorization: its kind and
// the object tokens of its name (the tokens beyond the
// action prefix).
type LabelFactor struct {
	Kind    string
	SubKind string // the middle level, when the label's rows carried one
	Objects []string
}

// FactorTable is the factorization of a classifier's
// whole label inventory, derived from the classifier
// itself (Labels + LabelKinds — the corpus's labels and
// label→kind map, learned by intent.Train from the same
// pairs the read serves). One source of truth: the table
// can never name a label or kind the classifier lacks.
type FactorTable struct {
	ByLabel map[string]LabelFactor
	Tree    *KindTree
}

// KindTree is the explicit two-level tree over a
// classifier's inventory: every label that carries a
// kind has that kind as its PARENT, and every kind's
// CHILDREN are its labels. It is derived from the same
// LabelKinds map the factors come from — one source of
// truth, so the tree can never disagree with the
// classifier about which neighbourhood a meaning lives
// in. Children are sorted: the walk's candidate order
// is fixed, never map order.
type KindTree struct {
	Parent   map[string]string   // label -> kind
	Children map[string][]string // kind -> sorted labels

	// The third level, present only when the corpus
	// declared sub-kinds: label -> sub-kind, sub-kind ->
	// its parent kind, sub-kind -> its sorted labels.
	// Three levels at most — kind → sub-kind → concept —
	// by design, the same bound the corpus format states.
	SubParent     map[string]string   // label -> sub-kind
	SubKindParent map[string]string   // sub-kind -> kind
	SubChildren   map[string][]string // sub-kind -> sorted labels
}

// DeriveKindTree builds the tree from a label→kind map.
func DeriveKindTree(kinds map[string]string) *KindTree {
	return DeriveKindTreeDeep(kinds, nil, nil)
}

// DeriveKindTreeDeep builds the tree with its sub-kind
// level: subs maps label -> sub-kind, subParents maps
// sub-kind -> kind. A label with a sub-kind keeps its
// kind as Parent (the top level never moves); the
// sub-kind level is additional structure the walk can
// climb through, not a replacement for the kind.
func DeriveKindTreeDeep(kinds, subs, subParents map[string]string) *KindTree {
	t := &KindTree{
		Parent:        make(map[string]string, len(kinds)),
		Children:      make(map[string][]string),
		SubParent:     make(map[string]string, len(subs)),
		SubKindParent: make(map[string]string, len(subParents)),
		SubChildren:   make(map[string][]string),
	}
	for label, kind := range kinds {
		t.Parent[label] = kind
		t.Children[kind] = append(t.Children[kind], label)
	}
	for kind := range t.Children {
		sort.Strings(t.Children[kind])
	}
	for sub, kind := range subParents {
		t.SubKindParent[sub] = kind
	}
	for label, sub := range subs {
		t.SubParent[label] = sub
		t.SubChildren[sub] = append(t.SubChildren[sub], label)
	}
	for sub := range t.SubChildren {
		sort.Strings(t.SubChildren[sub])
	}
	return t
}

// Climb returns a label's ancestry in the tree, nearest
// first: its sub-kind (when it has one), then its kind.
// The parent walk climbs this chain — more than one
// step, where round 4's walk climbed exactly one.
func (t *KindTree) Climb(label string) []string {
	var out []string
	if sub := t.SubParent[label]; sub != "" {
		out = append(out, sub)
	}
	if kind := t.Parent[label]; kind != "" {
		out = append(out, kind)
	}
	return out
}

// TopKind resolves any tree node — a label, a sub-kind,
// or a kind — to the KIND at the top of its chain. It
// is the inheritance lookup: a sub-kind's labels answer
// disjointness and kind questions with their top kind,
// because the pairs and neighbourhoods are stated at
// kind level unless a sub-kind declares its own.
func (t *KindTree) TopKind(node string) string {
	if kind, ok := t.SubKindParent[node]; ok {
		return kind
	}
	if kind := t.Parent[node]; kind != "" {
		return kind
	}
	if _, isKind := t.Children[node]; isKind {
		return node
	}
	return ""
}

// ParentOf returns a label's kind — the one step up the
// tree — or "" when the label carries no kind.
func (t *KindTree) ParentOf(label string) string {
	return t.Parent[label]
}

// DeriveFactors builds the table. Labels with no object
// part (single-token names — greeting, feelings) and
// labels whose rows carried no kind are excluded: neither
// can ever be the unique composition of a kind and an
// object.
func DeriveFactors(clf *intent.Classifier) *FactorTable {
	kinds := clf.LabelKinds()
	subs := clf.LabelSubKinds()
	subParents := map[string]string{}
	for _, label := range clf.Labels {
		if sub := subs[label]; sub != "" {
			subParents[sub] = clf.SubKindParent(sub)
		}
	}
	t := &FactorTable{
		ByLabel: make(map[string]LabelFactor, len(clf.Labels)),
		Tree:    DeriveKindTreeDeep(kinds, subs, subParents),
	}
	for _, label := range clf.Labels {
		kind := kinds[label]
		parts := strings.Split(label, "_")
		if kind == "" || len(parts) < 2 {
			continue
		}
		t.ByLabel[label] = LabelFactor{Kind: kind, SubKind: subs[label], Objects: parts[1:]}
	}
	return t
}

// LineObjectWords splits a line into the words the object
// factor can see: the intent classifier's own
// tokenization (lowercased, tokens containing a letter),
// as a set.
func LineObjectWords(line string) map[string]bool {
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

// ComposeIn runs the factor composition inside ONE kind:
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
func (t *FactorTable) ComposeIn(words map[string]bool, kind string) (string, int) {
	labels := make([]string, 0, len(t.ByLabel))
	for label := range t.ByLabel {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	found := ""
	n := 0
	for _, label := range labels {
		f := t.ByLabel[label]
		// A node matches a label when it is the label's
		// kind OR its sub-kind: composition runs inside
		// whichever level of the tree it is given.
		if f.Kind != kind && f.SubKind != kind {
			continue
		}
		ok := true
		for _, obj := range f.Objects {
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

// FallbackRead composes a read for a line the label read
// left behind. Step one is the original composition: the
// predicted kind plus the line's object words, inside the
// predicted kind's children. Step two is the parent walk
// (see the header): it runs only when the kind read
// abstained — a spoken kind guess whose neighbourhood
// composed nothing is an answer, not an invitation to
// try another kind. Exactly one candidate at either
// step is adopted; any other outcome abstains.
//
// The third result is the provenance line for the
// outcome — the adoption's composition, or the ambiguity
// that closed the line — in the wording the inspect
// traces have always used; "" when nothing worth
// tracing happened. Callers print it or drop it; the
// read is the same either way.
func (t *FactorTable) FallbackRead(clf *intent.Classifier, line, kindGuess string) (label string, ok bool, trace string) {
	words := LineObjectWords(line)
	if kindGuess != "" {
		label, n := t.ComposeIn(words, kindGuess)
		if n == 1 {
			return label, true, fmt.Sprintf("fallback: kind guess %s + the line's objects compose %s (%s)", kindGuess, label, t.ExplainComposition(words, label))
		}
		if n > 1 {
			// Sub-kind narrowing: ambiguity at KIND level
			// is not the end when the narrow read's top
			// label lives in a sub-kind of the guessed
			// kind — composition inside the sub-kind is a
			// different, narrower question, and exactly
			// one candidate there adopts. A corpus with no
			// sub-kinds never reaches this: ambiguity
			// stays final, as round 4 left it.
			if clf != nil {
				if ranked := clf.RankedLabels(line); len(ranked) > 0 {
					if sub := t.Tree.SubParent[ranked[0].Label]; sub != "" && t.Tree.TopKind(sub) == kindGuess {
						if subLabel, m := t.ComposeIn(words, sub); m == 1 {
							return subLabel, true, fmt.Sprintf("fallback: %d candidates inside kind %s — narrowed by top label %s to sub-kind %s, whose children compose %s (%s)", n, kindGuess, ranked[0].Label, sub, subLabel, t.ExplainComposition(words, subLabel))
						}
					}
				}
			}
			return "", false, fmt.Sprintf("fallback: %d candidates inside kind %s — ambiguity is final, no walk", n, kindGuess)
		}
		return "", false, ""
	}
	// The parent walk: the narrow read's top label, up
	// one level to its parent kind, composition among
	// the parent's children. A classifier whose ranking
	// is empty (no feature evidence) names no parent,
	// and a top label with no kind has none to walk to.
	if clf == nil {
		return "", false, ""
	}
	ranked := clf.RankedLabels(line)
	if len(ranked) == 0 {
		return "", false, ""
	}
	// The climb: the top label's ancestry, nearest level
	// first — its sub-kind, then its kind. Composition
	// runs at each level; exactly one candidate adopts,
	// ambiguity at any level is final, and an empty level
	// climbs on. With no sub-kinds the chain is the one
	// parent kind round 4's walk climbed, worded as it
	// always was.
	chain := t.Tree.Climb(ranked[0].Label)
	for _, node := range chain {
		if node == "" || node == kindGuess {
			continue
		}
		label, n := t.ComposeIn(words, node)
		if n == 1 {
			if _, isSub := t.Tree.SubKindParent[node]; isSub {
				return label, true, fmt.Sprintf("fallback: parent walk — top label %s (%.2f) up to sub-kind %s, whose children compose %s (%s)", ranked[0].Label, ranked[0].Prob, node, label, t.ExplainComposition(words, label))
			}
			return label, true, fmt.Sprintf("fallback: parent walk — top label %s (%.2f) up to parent kind %s, whose children compose %s (%s)", ranked[0].Label, ranked[0].Prob, node, label, t.ExplainComposition(words, label))
		}
		if n > 1 {
			return "", false, fmt.Sprintf("fallback: parent walk — %d candidates inside %s — ambiguity is final, no climb", n, node)
		}
	}
	return "", false, ""
}

// ExplainComposition renders, for a provenance trace,
// how each of an adopted label's object words was found
// in the line: verbatim, as its plural surface form, or
// through a cousin link (naming the cousin the line
// actually carried). It re-derives what ComposeIn
// checked, for the one adopted label only.
func (t *FactorTable) ExplainComposition(words map[string]bool, label string) string {
	f := t.ByLabel[label]
	parts := make([]string, 0, len(f.Objects))
	for _, obj := range f.Objects {
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
