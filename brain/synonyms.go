package brain

// Synonym augmentation: the second trainer-side augmentation
// (the first is mutation — see mutate.go). Where mutation
// teaches the brain that a line's meaning survives sloppy
// TYPING, synonyms teach it that meaning survives sloppy
// WORD CHOICE: with probability TrainOptions.Synonyms, one
// word of a pair's INPUT is swapped for a known synonym in
// a given epoch — "i love music" is seen as "i enjoy music"
// — while the TARGET stays exactly as written.
//
// The table below is deliberately small and honest: common
// verbs, adjectives, and nouns of everyday small talk, most
// of them words the bundled example corpus itself uses.
// It is not a thesaurus and does not pretend to be one;
// pairs are near-synonyms in everyday usage, applied in
// both directions.
//
// One guard keeps the augmentation truthful: a swap only
// lands when the synonym is IN the training vocabulary.
// Swapping in a word the model has never seen would train
// it on an unknown token in that slot — noise, not a
// paraphrase.

import (
	"math/rand"
	"strings"

	"github.com/golangast/gollemer-brains/tokenizer"
	"github.com/golangast/gollemer-brains/vocab"
)

// synonymPairs are the built-in near-synonym pairs, applied
// in both directions.
var synonymPairs = [][2]string{
	{"like", "love"}, {"like", "enjoy"}, {"love", "enjoy"},
	{"good", "great"}, {"good", "nice"}, {"great", "wonderful"},
	{"happy", "glad"}, {"happy", "pleased"}, {"glad", "pleased"},
	{"start", "begin"},
	{"big", "large"}, {"big", "huge"},
	{"small", "little"}, {"small", "tiny"},
	{"easy", "simple"},
	{"hard", "difficult"},
	{"fast", "quick"},
	{"calm", "quiet"}, {"calm", "peaceful"}, {"quiet", "silent"},
	{"warm", "cozy"},
	{"cold", "chilly"},
	{"sunny", "bright"},
	{"kind", "sweet"}, {"kind", "friendly"},
	{"funny", "silly"},
	{"pretty", "beautiful"},
	{"smart", "clever"},
	{"strong", "sturdy"},
	{"tired", "sleepy"},
	{"scared", "afraid"},
	{"angry", "mad"},
	{"sad", "unhappy"},
	{"sure", "certain"},
	{"new", "fresh"},
	{"say", "tell"},
	{"see", "watch"},
	{"make", "build"},
	{"pick", "choose"},
	{"walk", "stroll"},
}

// synonyms maps each word to its known synonyms (both
// directions of every pair).
var synonyms = buildSynonyms()

func buildSynonyms() map[string][]string {
	m := map[string][]string{}
	add := func(a, b string) {
		for _, s := range m[a] {
			if s == b {
				return
			}
		}
		m[a] = append(m[a], b)
	}
	for _, p := range synonymPairs {
		add(p[0], p[1])
		add(p[1], p[0])
	}
	return m
}

// SynonymsFor returns the built-in synonyms of a word (nil
// when the table has none).
func SynonymsFor(word string) []string {
	return synonyms[word]
}

// SwapSynonym swaps ONE word of the line for a synonym and
// returns the result, or the line itself when no swap
// applies. Only swaps whose replacement is in v land (a nil
// v disables that guard — tests use it to see the raw
// table). All randomness comes from rng.
func SwapSynonym(line string, rng *rand.Rand, v *vocab.Vocabulary) string {
	toks := TokenizeLower(line)
	var cand []int
	for i, w := range toks {
		if len(SynonymsFor(w)) > 0 {
			cand = append(cand, i)
		}
	}
	if len(cand) == 0 {
		return line
	}
	// Try candidates in random order; the first with an
	// in-vocabulary synonym wins.
	for _, k := range rng.Perm(len(cand)) {
		i := cand[k]
		opts := SynonymsFor(toks[i])
		var usable []string
		for _, s := range opts {
			if v == nil {
				usable = append(usable, s)
			} else if _, ok := v.WordToToken[s]; ok {
				usable = append(usable, s)
			}
		}
		if len(usable) == 0 {
			continue
		}
		out := append([]string{}, toks...)
		out[i] = usable[rng.Intn(len(usable))]
		return TidyDecode(strings.Join(out, " "))
	}
	return line
}

// rerollEncoded returns a copy of enc re-rolled for one
// epoch under BOTH augmentations: each pair's input may be
// synonym-swapped with probability synP and/or mutated with
// probability augP (the two rolls are independent, synonyms
// first), then re-encoded. Targets are carried over
// untouched. With synP == 0 the rng is consumed exactly as
// mutateEncoded consumes it, so an augment-only run is
// bit-identical to the mutation-only trainer.
func rerollEncoded(enc []encodedPair, rng *rand.Rand, augP, synP float64, v *vocab.Vocabulary, tok *tokenizer.Tokenizer) []encodedPair {
	out := make([]encodedPair, len(enc))
	copy(out, enc)
	for i := range out {
		text := out[i].srcInput
		if synP > 0 && rng.Float64() < synP {
			text = SwapSynonym(text, rng, v)
		}
		if augP > 0 && rng.Float64() < augP {
			text = MutateInput(text, rng)
		}
		if text == out[i].srcInput {
			continue
		}
		ids, surfs := encodeInputSide(text, v, tok)
		if len(ids) == 0 {
			continue
		}
		out[i].input = ids
		out[i].inputSurfs = surfs
	}
	return out
}
