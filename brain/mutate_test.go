package brain

import (
	"math/rand"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/seq2seq"
)

// mutateSamples are lines of the shapes the example corpus
// uses: long enough to mangle, none ending in a filler word
// (a trailing filler drop is allowed to remove a word, and
// "please" counts as a content word to the checker).
var mutateSamples = []string{
	"what do you think of gardening",
	"do you like cooking a lot",
	"how do i start cycling in the morning",
	"i like reading books in the evening",
	"tell me about trains and music",
	"how can i learn photography",
	"good morning!",
	"hi!",
	"hello",
	"bye",
}

func TestMutateInputDeterministic(t *testing.T) {
	run := func() []string {
		rng := rand.New(rand.NewSource(42))
		var out []string
		for i := 0; i < 100; i++ {
			out = append(out, MutateInput(mutateSamples[i%len(mutateSamples)], rng))
		}
		return out
	}
	a, b := run(), run()
	if !reflect.DeepEqual(a, b) {
		t.Fatal("same seed, different mutation streams")
	}
	changed := 0
	for i, s := range a {
		if s != mutateSamples[i%len(mutateSamples)] {
			changed++
		}
	}
	if changed == 0 {
		t.Fatal("100 rolls and nothing ever changed: the mutator is a no-op")
	}
}

func TestMutateInputTinyLines(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	// One- and two-word lines may only lose trailing
	// punctuation; their words are untouchable.
	cases := map[string]string{
		"hello":         "hello",
		"bye":           "bye",
		"hi!":           "hi",
		"good morning!": "good morning",
		"thank you":     "thank you",
	}
	for in, want := range cases {
		if got := MutateInput(in, rng); got != want {
			t.Errorf("MutateInput(%q) = %q, want %q", in, got, want)
		}
	}
	// Nothing ever mutates into emptiness.
	for _, line := range mutateSamples {
		for seed := int64(0); seed < 20; seed++ {
			out := MutateInput(line, rand.New(rand.NewSource(seed)))
			if strings.TrimSpace(out) == "" || countWords(TokenizeLower(out)) == 0 {
				t.Fatalf("MutateInput(%q, seed %d) = %q: mutated away", line, seed, out)
			}
		}
	}
}

// typoVariantOf reports whether b is a with exactly one small
// typo applied: one letter deleted, one letter duplicated, or
// two adjacent letters swapped.
func typoVariantOf(a, b string) bool {
	ra, rb := []rune(a), []rune(b)
	switch {
	case len(rb) == len(ra)-1: // one letter deleted
		i := 0
		for i < len(rb) && ra[i] == rb[i] {
			i++
		}
		return string(ra[:i])+string(ra[i+1:]) == b
	case len(rb) == len(ra)+1: // one letter duplicated
		i := 0
		for i < len(ra) && ra[i] == rb[i] {
			i++
		}
		// rb has one extra rune at or before i; removing it
		// must give back a.
		return string(rb[:i])+string(rb[i+1:]) == a || (i > 0 && string(rb[:i-1])+string(rb[i:]) == a)
	case len(rb) == len(ra): // two adjacent letters swapped
		var diff []int
		for i := range ra {
			if ra[i] != rb[i] {
				diff = append(diff, i)
			}
		}
		return len(diff) == 2 && diff[1] == diff[0]+1 &&
			ra[diff[0]] == rb[diff[1]] && ra[diff[1]] == rb[diff[0]]
	}
	return false
}

func TestMutatePreservesContentWords(t *testing.T) {
	// The loose contract: a mutation never loses a content
	// word — except the typo kind, which may bend exactly one
	// content word into a one-typo variant of itself.
	rng := rand.New(rand.NewSource(99))
	for _, line := range mutateSamples {
		want := seq2seq.ContentWordsOf(TokenizeLower(line))
		for roll := 0; roll < 200; roll++ {
			out := MutateInput(line, rng)
			if out == line {
				continue
			}
			got := seq2seq.ContentWordsOf(TokenizeLower(out))
			gotSet := map[string]bool{}
			for _, w := range got {
				gotSet[w] = true
			}
			var missing []string
			for _, w := range want {
				if !gotSet[w] {
					missing = append(missing, w)
				}
			}
			if len(missing) > 1 {
				t.Fatalf("MutateInput(%q) = %q: lost content words %v", line, out, missing)
			}
			if len(missing) == 1 {
				ok := false
				for _, w := range got {
					if typoVariantOf(missing[0], w) {
						ok = true
					}
				}
				if !ok {
					t.Fatalf("MutateInput(%q) = %q: content word %q lost, not typo'd", line, out, missing[0])
				}
			}
		}
	}
}

func TestMutateEncodedTargetsUntouched(t *testing.T) {
	pairs := []data.Pair{
		{Input: "what do you think of gardening", Output: "Gardening is peaceful."},
		{Input: "do you like cooking a lot", Output: "Yes, I like cooking a lot."},
		{Input: "how do i start cycling in the morning", Output: "Start with short rides."},
		{Input: "tell me about trains and music", Output: "Trains are calm."},
		{Input: "how can i learn photography", Output: "Start with your phone."},
	}
	v, tok, err := BuildVocab(pairs)
	if err != nil {
		t.Fatalf("BuildVocab: %v", err)
	}
	var enc []encodedPair
	for _, p := range pairs {
		ep, ok := encodePair(p, v, tok)
		if !ok {
			t.Fatalf("pair %q did not encode", p.Input)
		}
		enc = append(enc, ep)
	}

	// p = 0: the re-roll is the identity, byte for byte.
	zero := mutateEncoded(enc, rand.New(rand.NewSource(1)), 0, v, tok)
	for i := range enc {
		if !reflect.DeepEqual(zero[i].input, enc[i].input) || !reflect.DeepEqual(zero[i].target, enc[i].target) {
			t.Fatalf("p=0 changed pair %d", i)
		}
	}

	// p = 1: every input is re-rolled (and these lines all
	// have a mutation that applies); targets never move.
	one := mutateEncoded(enc, rand.New(rand.NewSource(1)), 1, v, tok)
	changed := 0
	for i := range enc {
		if !reflect.DeepEqual(one[i].target, enc[i].target) || !reflect.DeepEqual(one[i].targetSurfs, enc[i].targetSurfs) {
			t.Fatalf("p=1 pair %d: target moved", i)
		}
		if !reflect.DeepEqual(one[i].input, enc[i].input) {
			changed++
		}
	}
	if changed == 0 {
		t.Fatal("p=1 and no input changed")
	}
}

func TestTrainAugmentValidationAndRun(t *testing.T) {
	var pairs []data.Pair
	for i := 0; i < 12; i++ {
		pairs = append(pairs, data.Pair{Input: "what do you think of gardening", Output: "Gardening is peaceful."})
	}
	out := filepath.Join(t.TempDir(), "b.gob")
	if err := Train(pairs, out, TrainOptions{Epochs: 1, Augment: 1.5}); err == nil ||
		!strings.Contains(err.Error(), "augment") {
		t.Fatalf("Augment=1.5: err = %v, want an augment range error", err)
	}
	if err := Train(pairs, out, TrainOptions{Epochs: 2, Augment: 0.5}); err != nil {
		t.Fatalf("Train with Augment=0.5: %v", err)
	}
}
