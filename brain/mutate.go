package brain

// Mutation augmentation: an opt-in way to teach the brain
// that a line's meaning survives the way people actually
// type it. During training, each pair's INPUT is re-rolled
// every epoch: with probability TrainOptions.Augment it is
// replaced, for that epoch only, by a lightly mangled variant
// — a dropped filler word, two swapped neighbours, one small
// typo, a dropped trailing punctuation mark. The TARGET is
// never touched: the reply stays the correct one, so the
// model learns the mangling does not change the meaning.
//
// This is the spirit of Gollemer's mutation augmentation
// (internal/ai/training/chat/mutation_augment.go in the
// source, built for its robustness round), simplified to fit
// this repo: the source adds one permanent mutated sibling
// per pair under a frozen-eval blocklist; here there is no
// eval to protect and no sibling file to manage, so the
// mutation is re-rolled per epoch instead — over a run, a
// pair is seen both clean and mangled, in many variants.
//
// The stream is seeded (augmentSeed), so a run's mutations
// are reproducible: the same corpus and the same probability
// always mangle the same way.

import (
	"math/rand"
	"strings"
	"unicode"

	"github.com/golangast/gollemer-brains/seq2seq"
	"github.com/golangast/gollemer-brains/tokenizer"
	"github.com/golangast/gollemer-brains/vocab"
)

// augmentSeed fixes the mutation stream: the same corpus
// always produces the same manglings, so runs are comparable.
const augmentSeed = 20261121

// trailingFillerWords are the words the trailing-drop
// mutation may remove from the end of a line: politeness and
// emphasis words whose absence changes nothing.
var trailingFillerWords = map[string]bool{
	"please": true, "now": true, "just": true,
	"really": true, "very": true,
}

// MutateInput applies ONE light mutation to a line and
// returns the result, or the line itself when no mutation
// applies. The mutations are beginner-plausible typing
// accidents, chosen at random:
//
//   - drop one non-content (function) word, never the first;
//   - swap two adjacent words, never involving the first;
//   - one small typo in a word of 4+ letters: delete a
//     letter, swap two letters, or duplicate a letter —
//     never the word's first letter;
//   - drop trailing punctuation or a trailing filler word.
//
// A line is never mutated into emptiness, and a line of one
// or two words may only lose trailing punctuation. All
// randomness comes from rng, so a fixed seed reproduces a
// fixed stream of mutations.
func MutateInput(line string, rng *rand.Rand) string {
	out, changed := mutateTokens(TokenizeLower(line), rng)
	if !changed {
		return line
	}
	return TidyDecode(strings.Join(out, " "))
}

// mutateTokens is MutateInput over pre-split tokens; it
// reports whether anything visibly changed.
func mutateTokens(toks []string, rng *rand.Rand) ([]string, bool) {
	if countWords(toks) <= 2 {
		// Tiny lines carry their whole meaning in a word or
		// two; the only safe accident is losing a trailing
		// punctuation mark ("good morning!" -> "good morning").
		return dropTrailingPunct(toks)
	}
	kinds := []func([]string, *rand.Rand) ([]string, bool){
		mutateDropNonContent,
		mutateSwapAdjacent,
		mutateTypoWord,
		mutateDropTrailing,
	}
	// Try the kinds in a random order; the first that visibly
	// changes the line wins. A kind that cannot apply to this
	// line simply declines.
	for _, k := range rng.Perm(len(kinds)) {
		if out, ok := kinds[k](toks, rng); ok {
			return out, true
		}
	}
	return toks, false
}

// mutateDropNonContent drops one non-content word (an
// article, pronoun, auxiliary, preposition — anything
// seq2seq.IsContentWord rejects), never the first token:
// "do you like gardening" -> "do like gardening". Content
// words are the line's meaning and are never dropped here.
func mutateDropNonContent(toks []string, rng *rand.Rand) ([]string, bool) {
	var cand []int
	for i, w := range toks {
		if i > 0 && hasLetter(w) && !seq2seq.IsContentWord(w) {
			cand = append(cand, i)
		}
	}
	if len(cand) == 0 {
		return toks, false
	}
	i := cand[rng.Intn(len(cand))]
	out := append([]string{}, toks[:i]...)
	out = append(out, toks[i+1:]...)
	return out, true
}

// mutateSwapAdjacent swaps two adjacent word tokens, never
// involving the first token: "what do you think of music" ->
// "what do think you of music". Swapping identical neighbours
// changes nothing and declines.
func mutateSwapAdjacent(toks []string, rng *rand.Rand) ([]string, bool) {
	var cand []int
	for i := 1; i+1 < len(toks); i++ {
		if hasLetter(toks[i]) && hasLetter(toks[i+1]) && toks[i] != toks[i+1] {
			cand = append(cand, i)
		}
	}
	if len(cand) == 0 {
		return toks, false
	}
	i := cand[rng.Intn(len(cand))]
	out := append([]string{}, toks...)
	out[i], out[i+1] = out[i+1], out[i]
	return out, true
}

// mutateTypoWord puts one small typo in a word of at least 4
// letters, all letters: delete a letter, swap two adjacent
// letters, or duplicate a letter. The first letter is never
// touched — a word's opening is how readers (and this model)
// recognize it. Retries when an operation is a no-op
// (swapping identical letters), with a final fallback of
// dropping the word's last letter.
func mutateTypoWord(toks []string, rng *rand.Rand) ([]string, bool) {
	var cand []int
	for i, w := range toks {
		if len([]rune(w)) >= 4 && allLetters(w) {
			cand = append(cand, i)
		}
	}
	if len(cand) == 0 {
		return toks, false
	}
	orig := strings.Join(toks, " ")
	for try := 0; try < 8; try++ {
		out := append([]string{}, toks...)
		i := cand[rng.Intn(len(cand))]
		w := []rune(out[i])
		pos := 1 + rng.Intn(len(w)-1) // 1..len-1: never the first letter
		switch rng.Intn(3) {
		case 0: // delete the letter at pos
			out[i] = string(w[:pos]) + string(w[pos+1:])
		case 1: // swap the letters at pos and pos+1
			if pos == len(w)-1 {
				continue
			}
			w[pos], w[pos+1] = w[pos+1], w[pos]
			out[i] = string(w)
		case 2: // duplicate the letter at pos
			out[i] = string(w[:pos+1]) + string(w[pos:])
		}
		if strings.Join(out, " ") != orig {
			return out, true
		}
	}
	out := append([]string{}, toks...)
	w := []rune(out[cand[0]])
	out[cand[0]] = string(w[:len(w)-1])
	return out, true
}

// mutateDropTrailing drops a trailing punctuation mark or a
// trailing filler word: "thank you very much" keeps its
// meaning without the "very"; "good morning!" reads the same
// without the "!".
func mutateDropTrailing(toks []string, rng *rand.Rand) ([]string, bool) {
	if out, ok := dropTrailingPunct(toks); ok {
		return out, true
	}
	if n := len(toks); n > 0 && trailingFillerWords[toks[n-1]] {
		return append([]string{}, toks[:n-1]...), true
	}
	return toks, false
}

// dropTrailingPunct removes a trailing punctuation mark:
// either a punctuation token of its own, or — the shape
// TokenizeLower actually produces for raw text — one mark
// still attached to the final word ("morning!").
func dropTrailingPunct(toks []string) ([]string, bool) {
	n := len(toks)
	if n == 0 {
		return toks, false
	}
	last := toks[n-1]
	if isPunctToken(last) {
		return append([]string{}, toks[:n-1]...), true
	}
	if r := []rune(last); len(r) > 1 && isPunctRune(r[len(r)-1]) {
		out := append([]string{}, toks...)
		out[n-1] = string(r[:len(r)-1])
		return out, true
	}
	return toks, false
}

// isPunctToken reports whether a token is one of the
// single-character punctuation tokens TokenizeLower splits
// out.
func isPunctToken(t string) bool {
	return len(t) == 1 && isPunctRune(rune(t[0]))
}

// isPunctRune reports whether r is one of the punctuation
// characters TokenizeLower splits on.
func isPunctRune(r rune) bool {
	return strings.ContainsRune(`{}[],:"?!.`, r)
}

// countWords counts the tokens that carry letters (words, as
// opposed to punctuation tokens).
func countWords(toks []string) int {
	n := 0
	for _, t := range toks {
		if hasLetter(t) {
			n++
		}
	}
	return n
}

func hasLetter(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) {
			return true
		}
	}
	return false
}

// allLetters reports whether every rune of w is a letter.
func allLetters(w string) bool {
	for _, r := range w {
		if !unicode.IsLetter(r) {
			return false
		}
	}
	return len(w) > 0
}

// mutateEncoded returns a copy of enc re-rolled for one
// epoch: each pair's input is replaced by a mutated variant
// with probability p (and re-encoded), and left clean
// otherwise. Targets are carried over untouched — that is
// the whole point of the augmentation. A mutation that fails
// to encode (it should not happen: unknown words encode as
// the unknown token) leaves the pair clean rather than
// dropping it.
func mutateEncoded(enc []encodedPair, rng *rand.Rand, p float64, v *vocab.Vocabulary, tok *tokenizer.Tokenizer) []encodedPair {
	out := make([]encodedPair, len(enc))
	copy(out, enc)
	for i := range out {
		if rng.Float64() >= p {
			continue
		}
		mut := MutateInput(out[i].srcInput, rng)
		if mut == out[i].srcInput {
			continue
		}
		ids, surfs := encodeInputSide(mut, v, tok)
		if len(ids) == 0 {
			continue
		}
		out[i].input = ids
		out[i].inputSurfs = surfs
	}
	return out
}
