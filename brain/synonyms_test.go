package brain

import (
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/vocab"
)

func TestSynonymTableShape(t *testing.T) {
	if len(synonymPairs) < 30 || len(synonymPairs) > 60 {
		t.Fatalf("table has %d pairs, want an honest 30-60", len(synonymPairs))
	}
	for _, p := range synonymPairs {
		if p[0] == p[1] {
			t.Fatalf("pair %v maps a word to itself", p)
		}
		// Both directions must exist.
		found := false
		for _, s := range SynonymsFor(p[0]) {
			if s == p[1] {
				found = true
			}
		}
		if !found {
			t.Fatalf("SynonymsFor(%q) lacks %q", p[0], p[1])
		}
		found = false
		for _, s := range SynonymsFor(p[1]) {
			if s == p[0] {
				found = true
			}
		}
		if !found {
			t.Fatalf("SynonymsFor(%q) lacks %q", p[1], p[0])
		}
	}
}

func TestSwapSynonymSwapsOneWord(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	out := SwapSynonym("i love music", rng, nil)
	words := strings.Fields(out)
	if len(words) != 3 || words[0] != "i" || words[2] != "music" {
		t.Fatalf("out = %q, want only the middle word changed", out)
	}
	ok := false
	for _, s := range SynonymsFor("love") {
		if words[1] == s {
			ok = true
		}
	}
	if !ok {
		t.Fatalf("out = %q, middle word is not a synonym of love", out)
	}
	// Deterministic under the seed.
	a := SwapSynonym("i love music", rand.New(rand.NewSource(3)), nil)
	b := SwapSynonym("i love music", rand.New(rand.NewSource(3)), nil)
	if a != b {
		t.Fatalf("same seed, different swaps: %q vs %q", a, b)
	}
	// No table word in the line: no swap.
	if got := SwapSynonym("the cat sleeps soundly", rand.New(rand.NewSource(3)), nil); got != "the cat sleeps soundly" {
		t.Fatalf("no-candidate line changed: %q", got)
	}
}

func TestSwapSynonymVocabGuard(t *testing.T) {
	// The vocab knows "enjoy" but not "like": love may swap
	// to enjoy only.
	v := vocab.NewVocabulary()
	for _, w := range []string{"i", "love", "music", "enjoy"} {
		v.AddToken(w)
	}
	out := SwapSynonym("i love music", rand.New(rand.NewSource(3)), v)
	if out != "i enjoy music" {
		t.Fatalf("out = %q, want the in-vocabulary synonym", out)
	}
	// A vocab with neither synonym: the swap must not land.
	v2 := vocab.NewVocabulary()
	for _, w := range []string{"i", "love", "music"} {
		v2.AddToken(w)
	}
	if got := SwapSynonym("i love music", rand.New(rand.NewSource(3)), v2); got != "i love music" {
		t.Fatalf("out-of-vocabulary swap landed: %q", got)
	}
}

func TestRerollSynonymsOnlyAndTargetsUntouched(t *testing.T) {
	pairs := []data.Pair{
		{Input: "i love music", Output: "Music is wonderful."},
		{Input: "do you like cooking", Output: "Yes, I like cooking."},
		{Input: "i am happy today", Output: "That is great."},
		{Input: "tell me about trains", Output: "Trains are calm."},
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

	// synP = 1, augP = 0: inputs may change (only by a
	// synonym swap), targets never do.
	out := rerollEncoded(enc, rand.New(rand.NewSource(5)), 0, 1, v, tok)
	changed := 0
	for i := range enc {
		if !reflect.DeepEqual(out[i].target, enc[i].target) {
			t.Fatalf("pair %d: target moved", i)
		}
		if !reflect.DeepEqual(out[i].input, enc[i].input) {
			changed++
			// The changed input must decode to the original
			// line with exactly one word swapped for a
			// synonym.
			orig := strings.Fields(enc[i].srcInput)
			got := make([]string, len(out[i].inputSurfs))
			copy(got, out[i].inputSurfs)
			if len(got) != len(orig) {
				t.Fatalf("pair %d: swapped line %v has different length than %v", i, got, orig)
			}
			diffs := 0
			for j := range orig {
				if got[j] != orig[j] {
					diffs++
					okSyn := false
					for _, s := range SynonymsFor(orig[j]) {
						if s == got[j] {
							okSyn = true
						}
					}
					if !okSyn {
						t.Fatalf("pair %d: %q -> %q is not a synonym swap", i, orig[j], got[j])
					}
				}
			}
			if diffs != 1 {
				t.Fatalf("pair %d: %d words changed, want exactly 1", i, diffs)
			}
		}
	}
	if changed == 0 {
		t.Fatal("synP=1 changed nothing — the swap never landed")
	}
}

func TestRerollAugmentOnlyMatchesMutateEncoded(t *testing.T) {
	// The bit-identity promise: with synP == 0, rerollEncoded
	// consumes the rng exactly as mutateEncoded does, so an
	// augment-only run is unchanged by this step's existence.
	pairs := []data.Pair{
		{Input: "what do you think of gardening", Output: "Gardening is peaceful."},
		{Input: "do you like cooking a lot", Output: "Yes, I like cooking a lot."},
		{Input: "how do i start cycling in the morning", Output: "Start with short rides."},
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
	a := rerollEncoded(enc, rand.New(rand.NewSource(11)), 0.7, 0, v, tok)
	b := mutateEncoded(enc, rand.New(rand.NewSource(11)), 0.7, v, tok)
	for i := range enc {
		if !reflect.DeepEqual(a[i].input, b[i].input) || !reflect.DeepEqual(a[i].inputSurfs, b[i].inputSurfs) {
			t.Fatalf("pair %d: reroll(augment-only) != mutateEncoded", i)
		}
	}
}
