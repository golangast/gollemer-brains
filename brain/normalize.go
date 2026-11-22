package brain

// The typo normalizer: a deterministic seat in front of the
// model that repairs a misspelled word before the line is
// encoded — ported from Gollemer's chat (the seat John
// approved 2026-11-10 after the frozen robustness set
// measured the production model at 9/40 on typo'd inputs).
// A word the model has never seen encodes as UNK, and one
// UNK in the wrong place can derail the whole reply; the
// seat hands the encoder the word the user meant, when —
// and only when — the evidence is unambiguous.
//
// The rules are deliberately narrow — a correction the seat
// cannot defend does not happen:
//
//   - a token is a candidate only when it is purely letters,
//     at least 4 letters long, and out of the model's
//     vocabulary;
//   - a candidate is replaced only when exactly one
//     vocabulary word is its unique nearest neighbour under
//     Damerau–Levenshtein distance — at most 1 edit for a
//     4–6 letter token, at most 2 for 7 letters or more.
//     Any tie for nearest, or no neighbour inside the
//     limit, leaves the token alone;
//   - a line needing more than two replacements is not a
//     typo'd sentence, it is a different sentence: it passes
//     through completely untouched.
//
// The seat runs on the normalized token stream (the same
// lowercasing and punctuation stripping the encoder itself
// applies — see seq2seq.NormalizeQuery), so it sees exactly
// the tokens the model would. It is inference-only:
// training never passes through here. In this repo the seat
// is opt-in per Brain (Brain.FixTypos); cmd/ask turns it on
// unless told otherwise.

import (
	"github.com/golangast/gollemer-brains/vocab"
)

// TypoFix records one replacement the seat made.
type TypoFix struct {
	From string
	To   string
}

// NormalizeTokens applies the seat to a tokenized line and
// returns the corrected tokens and the fixes it made. A
// line with more than two defensible fixes comes back
// untouched with no fixes — it was never a typo'd sentence.
func NormalizeTokens(toks []string, v *vocab.Vocabulary, targets []string) ([]string, []TypoFix) {
	var fixes []TypoFix
	out := append([]string{}, toks...)
	for i, tok := range toks {
		if len(tok) < 4 || !isAlphaToken(tok) {
			continue
		}
		if _, ok := v.WordToToken[tok]; ok {
			continue
		}
		to, ok := typoUniqueNearest(tok, targets)
		if !ok {
			continue
		}
		out[i] = to
		fixes = append(fixes, TypoFix{From: tok, To: to})
		if len(fixes) > 2 {
			return toks, nil
		}
	}
	if len(fixes) == 0 {
		return toks, nil
	}
	return out, fixes
}

// TypoTargets lists the vocabulary words a typo may be
// corrected TO: purely alphabetic words. Punctuation tokens,
// the special tokens, and anything with a digit or a symbol
// inside are never a correction target.
func TypoTargets(v *vocab.Vocabulary) []string {
	var out []string
	for w := range v.WordToToken {
		if isAlphaToken(w) {
			out = append(out, w)
		}
	}
	return out
}

// isAlphaToken reports whether s is one or more ASCII
// letters and nothing else.
func isAlphaToken(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < 'a' || s[i] > 'z' {
			return false
		}
	}
	return true
}

// typoMaxDist is the distance table: how many edits a
// correction may cost, by the typo'd token's length. One
// edit covers the observed typo shapes (a dropped letter, a
// swapped pair, one wrong letter) for short words; longer
// words earn a second edit because they have more letters
// to go wrong in.
func typoMaxDist(n int) int {
	if n >= 7 {
		return 2
	}
	return 1
}

// typoUniqueNearest finds tok's nearest neighbour among the
// target words under Damerau–Levenshtein distance. It
// answers only when one word is strictly nearest and inside
// the length's distance limit — a tie, however close, is
// two defensible readings, and the seat takes neither.
func typoUniqueNearest(tok string, targets []string) (string, bool) {
	maxD := typoMaxDist(len(tok))
	best := maxD + 1
	bestWord := ""
	tied := false
	for _, w := range targets {
		if d := len(w) - len(tok); d > maxD || -d > maxD {
			continue
		}
		dist := damerauLevenshtein(tok, w)
		if dist > maxD {
			continue
		}
		switch {
		case dist < best:
			best, bestWord, tied = dist, w, false
		case dist == best:
			tied = true
		}
	}
	if bestWord == "" || tied {
		return "", false
	}
	return bestWord, true
}

// damerauLevenshtein is the true Damerau–Levenshtein
// distance (Lowrance–Wagner): insertions, deletions,
// substitutions, and transpositions of adjacent letters —
// the swap being the classic typed typo ("gonig" for
// "going"), which plain Levenshtein prices at two edits and
// this metric at one. Inputs here are short lowercase ASCII
// words; the full matrix is the honest price.
func damerauLevenshtein(a, b string) int {
	la, lb := len(a), len(b)
	inf := la + lb
	// d carries a sentinel row and column (index 0) around
	// the usual (la+1)x(lb+1) grid.
	d := make([][]int, la+2)
	for i := range d {
		d[i] = make([]int, lb+2)
	}
	d[0][0] = inf
	for i := 0; i <= la; i++ {
		d[i+1][0] = inf
		d[i+1][1] = i
	}
	for j := 0; j <= lb; j++ {
		d[0][j+1] = inf
		d[1][j+1] = j
	}
	lastRow := map[byte]int{}
	for i := 1; i <= la; i++ {
		lastMatchCol := 0
		for j := 1; j <= lb; j++ {
			i1 := lastRow[b[j-1]]
			j1 := lastMatchCol
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
				lastMatchCol = j
			}
			d[i+1][j+1] = minInt4(
				d[i][j]+cost, // substitution
				d[i+1][j]+1,  // insertion
				d[i][j+1]+1,  // deletion
				d[i1][j1]+(i-i1-1)+1+(j-j1-1), // transposition
			)
		}
		lastRow[a[i-1]] = i
	}
	return d[la+1][lb+1]
}

func minInt4(a, b, c, d int) int {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	if d < m {
		m = d
	}
	return m
}

// fixTokens applies the typo seat to a line's tokens when
// this brain has it switched on, using the model's own
// vocabulary. It is the only place the seat touches the
// answer path.
func (b *Brain) fixTokens(toks []string) []string {
	if !b.FixTypos {
		return toks
	}
	if b.typoTargets == nil {
		b.typoTargets = TypoTargets(b.Model.Base.OutputVocab)
	}
	out, _ := NormalizeTokens(toks, b.Model.Base.OutputVocab, b.typoTargets)
	return out
}
