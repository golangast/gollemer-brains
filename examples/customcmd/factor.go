package main

import (
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
}

// deriveFactors builds the table. Labels with no object
// part (single-token names — greeting, feelings) and
// labels whose rows carried no kind are excluded: neither
// can ever be the unique composition of a kind and an
// object.
func deriveFactors(clf *intent.Classifier) *factorTable {
	kinds := clf.LabelKinds()
	t := &factorTable{byLabel: make(map[string]labelFactor, len(clf.Labels))}
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

// fallbackRead composes a read for a line the label read
// left behind: the predicted kind plus the line's object
// words. A label is a candidate when its kind is the
// predicted kind and its WHOLE object set is present in
// the line (subset, not intersection — otherwise
// explain_buffered_channel would collide with
// explain_channel on every channel line). Exactly one
// candidate is adopted; any other count abstains.
func (d *demo) fallbackRead(line, kindGuess string) (string, bool) {
	if kindGuess == "" {
		return "", false
	}
	t := d.factors()
	if t == nil {
		return "", false
	}
	words := lineObjectWords(line)
	found := ""
	for label, f := range t.byLabel {
		if f.kind != kindGuess {
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
		if found != "" {
			return "", false // a second candidate: abstain
		}
		found = label
	}
	if found == "" {
		return "", false
	}
	return found, true
}
