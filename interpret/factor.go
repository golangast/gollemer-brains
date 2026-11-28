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
// the kind guess alone. Two levels only — label and
// kind — by design: no deeper hierarchy, no transitive
// closure. Ambiguity is final: a step that finds two or
// more candidates abstains outright, and the walk never
// rescues an ambiguous step by trying another kind.
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
}

// DeriveKindTree builds the tree from a label→kind map.
func DeriveKindTree(kinds map[string]string) *KindTree {
	t := &KindTree{
		Parent:   make(map[string]string, len(kinds)),
		Children: make(map[string][]string),
	}
	for label, kind := range kinds {
		t.Parent[label] = kind
		t.Children[kind] = append(t.Children[kind], label)
	}
	for kind := range t.Children {
		sort.Strings(t.Children[kind])
	}
	return t
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
	t := &FactorTable{
		ByLabel: make(map[string]LabelFactor, len(clf.Labels)),
		Tree:    DeriveKindTree(kinds),
	}
	for _, label := range clf.Labels {
		kind := kinds[label]
		parts := strings.Split(label, "_")
		if kind == "" || len(parts) < 2 {
			continue
		}
		t.ByLabel[label] = LabelFactor{Kind: kind, Objects: parts[1:]}
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
		if f.Kind != kind {
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
	parent := t.Tree.ParentOf(ranked[0].Label)
	if parent == "" || parent == kindGuess {
		return "", false, ""
	}
	label, n := t.ComposeIn(words, parent)
	if n == 1 {
		return label, true, fmt.Sprintf("fallback: parent walk — top label %s (%.2f) up to parent kind %s, whose children compose %s (%s)", ranked[0].Label, ranked[0].Prob, parent, label, t.ExplainComposition(words, label))
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
