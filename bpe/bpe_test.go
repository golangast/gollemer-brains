package bpe

import (
	"reflect"
	"strings"
	"testing"
)

// A small corpus in the example pairs' style: several -ing
// hobbies, a few shared suffixes, some short words.
func testCorpus() []string {
	text := `i love gardening and painting in the morning
gardening is peaceful and painting is fun
i like reading and fishing and hiking
what do you think of cooking
cooking is a happy skill and reading takes you anywhere
the train is calm and the garden is green
cats are quiet and dogs are loyal
i am learning to paint and to cook`
	var words []string
	for _, w := range strings.Fields(text) {
		words = append(words, w, w) // every word twice: pairs need counts >= 2
	}
	return words
}

func TestRoundTripTrainedWords(t *testing.T) {
	words := testCorpus()
	tok := Train(words, 200)
	seen := map[string]bool{}
	for _, w := range words {
		if seen[w] {
			continue
		}
		seen[w] = true
		if got := Join(tok.EncodeForms(w)); got != w {
			t.Errorf("round trip: Join(EncodeForms(%q)) = %q", w, got)
		}
	}
}

func TestDeterministic(t *testing.T) {
	words := testCorpus()
	a := Train(words, 120).Merges()
	b := Train(words, 120).Merges()
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("training twice gave different merges:\n%v\n%v", a, b)
	}
}

func TestMergeCapRespected(t *testing.T) {
	tok := Train(testCorpus(), 10)
	if n := len(tok.Merges()); n > 10 {
		t.Fatalf("learned %d merges with a cap of 10", n)
	}
}

func TestUnseenWordDecomposes(t *testing.T) {
	tok := Train(testCorpus(), 300)
	pieces := tok.EncodeWord("knitting")
	if len(pieces) < 2 {
		t.Fatalf("unseen word knitting encoded to %v — want a decomposition", pieces)
	}
	for _, p := range pieces {
		if p == "knitting" || p == "<unk>" {
			t.Fatalf("unseen word produced the forbidden piece %q in %v", p, pieces)
		}
	}
	if got := Join(tok.EncodeForms("knitting")); got != "knitting" {
		t.Fatalf("unseen word round trip: got %q", got)
	}
}

func TestFromMergesEncodesTheSame(t *testing.T) {
	words := testCorpus()
	tok := Train(words, 150)
	rebuilt := FromMerges(tok.Merges())
	for _, w := range []string{"gardening", "painting", "cooking", "knitting", "the", "a"} {
		if a, b := tok.EncodeWord(w), rebuilt.EncodeWord(w); !reflect.DeepEqual(a, b) {
			t.Errorf("EncodeWord(%q): trained %v, rebuilt %v", w, a, b)
		}
	}
}

func TestJoinGluesContinuations(t *testing.T) {
	if got := Join([]string{"paint", "##ing", "is", "fun"}); got != "painting is fun" {
		t.Fatalf("Join = %q, want %q", got, "painting is fun")
	}
}
