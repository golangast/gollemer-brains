// Package bpe is a small byte-pair-encoding tokenizer,
// standard library only.
//
// Word tokenization splits text into whole words, so a word
// that never appeared in training has no token at all: the
// model sees <unk> and can never say the word back. BPE
// instead learns, from the training text itself, a ranked
// list of merges over characters. Frequent words end up as
// one or two pieces; a word the corpus never saw still
// decomposes into pieces of words it did see ("knitting"
// becomes "knitt" + "ing" when the corpus knows "painting"),
// so the model has tokens to work with on both sides.
//
// Pieces come in two forms. The first piece of a word is
// stored bare ("paint"); every continuation piece carries a
// "##" prefix ("##ing"). Join reverses the split exactly, so
// decode never has to guess where words end.
//
// BPE composes pieces of known words. It does not teach the
// model new topics: a brain that never saw a line about
// sailing still has nothing to say about sailing, it just
// has a way to spell the word.
package bpe

import (
	"sort"
	"strings"
)

// Merge is one learned merge: the adjacent pieces A and B
// combine into the single piece A+B. A tokenizer's merges
// apply in slice order — earlier merges outrank later ones.
type Merge struct {
	A, B string
}

// Tokenizer is a trained BPE tokenizer: a ranked merge list
// plus the set of pieces encoding can produce.
type Tokenizer struct {
	merges []Merge
	ranks  map[[2]string]int
	pieces map[string]bool
}

// Train learns up to maxMerges merges from words, a stream
// of pre-split tokens (duplicates carry frequency — pass
// every occurrence, not a set). Words are segmented
// independently; a single-character token never merges.
// Merge choice is fully deterministic: highest weighted
// count wins, ties break on the lexicographically smaller
// pair. Training stops early when no adjacent pair occurs
// at least twice.
func Train(words []string, maxMerges int) *Tokenizer {
	freq := make(map[string]int)
	for _, w := range words {
		if w != "" {
			freq[w]++
		}
	}
	reps := make(map[string][]string, len(freq))
	pieces := make(map[string]bool)
	for w := range freq {
		chars := make([]string, 0, len(w))
		for _, r := range w {
			s := string(r)
			chars = append(chars, s)
			pieces[s] = true
		}
		reps[w] = chars
	}
	var merges []Merge
	for len(merges) < maxMerges {
		counts := make(map[[2]string]int)
		for w, rep := range reps {
			for i := 0; i+1 < len(rep); i++ {
				counts[[2]string{rep[i], rep[i+1]}] += freq[w]
			}
		}
		var best [2]string
		bestCount := 0
		for pair, n := range counts {
			if n > bestCount || (n == bestCount && n > 0 &&
				(pair[0] < best[0] || (pair[0] == best[0] && pair[1] < best[1]))) {
				best, bestCount = pair, n
			}
		}
		if bestCount < 2 {
			break
		}
		merges = append(merges, Merge{A: best[0], B: best[1]})
		joined := best[0] + best[1]
		pieces[joined] = true
		for w, rep := range reps {
			reps[w] = applyMerge(rep, best[0], best[1])
		}
	}
	return fromMerges(merges, pieces)
}

// FromMerges rebuilds a tokenizer from a persisted merge
// list. Encoding depends only on the merges' ranks, so a
// tokenizer built this way encodes exactly like the one
// Train produced. Its Pieces set is recomputed from the
// merges alone and can miss a character that appeared in
// training but never took part in a merge — harmless,
// because Pieces only feeds vocabulary construction, which
// happens once, at training time, on the live tokenizer.
func FromMerges(merges []Merge) *Tokenizer {
	pieces := make(map[string]bool)
	for _, m := range merges {
		for _, r := range m.A {
			pieces[string(r)] = true
		}
		for _, r := range m.B {
			pieces[string(r)] = true
		}
		pieces[m.A] = true
		pieces[m.B] = true
		pieces[m.A+m.B] = true
	}
	return fromMerges(merges, pieces)
}

// fromMerges is the shared constructor.
func fromMerges(merges []Merge, pieces map[string]bool) *Tokenizer {
	t := &Tokenizer{
		merges: append([]Merge{}, merges...),
		ranks:  make(map[[2]string]int, len(merges)),
		pieces: pieces,
	}
	for i, m := range merges {
		t.ranks[[2]string{m.A, m.B}] = i
	}
	return t
}

// Merges returns a copy of the learned merges, in rank
// order — the tokenizer's persistable state.
func (t *Tokenizer) Merges() []Merge {
	return append([]Merge{}, t.merges...)
}

// Pieces returns every piece encoding can produce, sorted:
// the characters seen in training plus every merge result.
// A vocabulary built from these covers every word whose
// characters were all seen in training.
func (t *Tokenizer) Pieces() []string {
	out := make([]string, 0, len(t.pieces))
	for p := range t.pieces {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// EncodeWord splits one word into its pieces (bare forms).
// A character never seen in training stays a piece of its
// own — encoding never fails and never invents an <unk>
// piece; mapping pieces to ids is the caller's vocabulary
// problem.
func (t *Tokenizer) EncodeWord(word string) []string {
	if word == "" {
		return nil
	}
	rep := make([]string, 0, len(word))
	for _, r := range word {
		rep = append(rep, string(r))
	}
	for len(rep) > 1 {
		// Merge the lowest-ranked adjacent pair present,
		// everywhere it occurs, then rescan: the standard
		// greedy-by-rank BPE application.
		bestRank := -1
		var best [2]string
		for i := 0; i+1 < len(rep); i++ {
			if rank, ok := t.ranks[[2]string{rep[i], rep[i+1]}]; ok {
				if bestRank < 0 || rank < bestRank {
					bestRank, best = rank, [2]string{rep[i], rep[i+1]}
				}
			}
		}
		if bestRank < 0 {
			break
		}
		rep = applyMerge(rep, best[0], best[1])
	}
	return rep
}

// EncodeForms splits one word into vocabulary forms: the
// first piece bare, every continuation piece prefixed with
// "##". These are the strings a vocabulary should hold.
func (t *Tokenizer) EncodeForms(word string) []string {
	pieces := t.EncodeWord(word)
	forms := make([]string, len(pieces))
	for i, p := range pieces {
		if i == 0 {
			forms[i] = p
		} else {
			forms[i] = "##" + p
		}
	}
	return forms
}

// Join renders vocabulary forms back into text: a "##"
// piece glues onto what precedes it, any other piece starts
// a new word. Join(EncodeForms(w)) == w for a single word,
// and joining the forms of a whole line rebuilds the line
// with single spaces between words.
func Join(forms []string) string {
	var b strings.Builder
	for _, f := range forms {
		if strings.HasPrefix(f, "##") {
			b.WriteString(f[2:])
		} else {
			if b.Len() > 0 {
				b.WriteString(" ")
			}
			b.WriteString(f)
		}
	}
	return b.String()
}

// applyMerge replaces every non-overlapping occurrence of
// the adjacent pair (a, b) in rep with the single piece a+b.
func applyMerge(rep []string, a, b string) []string {
	out := make([]string, 0, len(rep))
	for i := 0; i < len(rep); i++ {
		if i+1 < len(rep) && rep[i] == a && rep[i+1] == b {
			out = append(out, a+b)
			i++
		} else {
			out = append(out, rep[i])
		}
	}
	return out
}
