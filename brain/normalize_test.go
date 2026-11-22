package brain

import (
	"reflect"
	"testing"

	"github.com/golangast/gollemer-brains/seq2seq"
	"github.com/golangast/gollemer-brains/tokenizer"
	"github.com/golangast/gollemer-brains/vocab"
)

func typoTestVocab(words ...string) *vocab.Vocabulary {
	v := vocab.NewVocabulary()
	for _, w := range words {
		v.AddToken(w)
	}
	return v
}

func TestNormalizeReplacesUniqueNeighbour(t *testing.T) {
	v := typoTestVocab("do", "you", "like", "gardening", "cooking")
	out, fixes := NormalizeTokens([]string{"do", "you", "like", "gardning"}, v, TypoTargets(v))
	if !reflect.DeepEqual(out, []string{"do", "you", "like", "gardening"}) {
		t.Fatalf("out = %v", out)
	}
	if len(fixes) != 1 || fixes[0] != (TypoFix{From: "gardning", To: "gardening"}) {
		t.Fatalf("fixes = %v", fixes)
	}
	// A transposition is one edit under Damerau-Levenshtein.
	out, _ = NormalizeTokens([]string{"cooknig"}, v, TypoTargets(v))
	if !reflect.DeepEqual(out, []string{"cooking"}) {
		t.Fatalf("transposition out = %v", out)
	}
}

func TestNormalizeLeavesKnownAndShortWordsAlone(t *testing.T) {
	v := typoTestVocab("i", "like", "cats", "cat")
	// "cats" is known; "cta" is a plausible typo of "cat" but
	// under 4 letters, so the seat never touches it.
	in := []string{"i", "like", "cta", "cats"}
	out, fixes := NormalizeTokens(in, v, TypoTargets(v))
	if !reflect.DeepEqual(out, in) || len(fixes) != 0 {
		t.Fatalf("out = %v fixes = %v, want untouched", out, fixes)
	}
}

func TestNormalizeDeclinesAmbiguousAndFar(t *testing.T) {
	v := typoTestVocab("cats", "hats", "gardening")
	// "bats" is one edit from both "cats" and "hats": a tie
	// is two defensible readings, so no replacement.
	out, fixes := NormalizeTokens([]string{"bats"}, v, TypoTargets(v))
	if !reflect.DeepEqual(out, []string{"bats"}) || len(fixes) != 0 {
		t.Fatalf("ambiguous out = %v fixes = %v", out, fixes)
	}
	// Nothing within the distance limit: untouched.
	out, fixes = NormalizeTokens([]string{"zanzibar"}, v, TypoTargets(v))
	if !reflect.DeepEqual(out, []string{"zanzibar"}) || len(fixes) != 0 {
		t.Fatalf("far out = %v fixes = %v", out, fixes)
	}
}

func TestNormalizeCapOfTwoLeavesLineUntouched(t *testing.T) {
	v := typoTestVocab("gardening", "cooking", "painting", "is", "fun")
	in := []string{"gardning", "cooknig", "paintnig"}
	out, fixes := NormalizeTokens(in, v, TypoTargets(v))
	if !reflect.DeepEqual(out, in) || len(fixes) != 0 {
		t.Fatalf("three typos should pass through untouched, got out = %v fixes = %v", out, fixes)
	}
	// Two is fine.
	out, fixes = NormalizeTokens([]string{"gardning", "is", "fun"}, v, TypoTargets(v))
	if len(fixes) != 1 || out[0] != "gardening" {
		t.Fatalf("out = %v fixes = %v", out, fixes)
	}
}

func TestEncodeLineAppliesFixTypos(t *testing.T) {
	// An untrained model is enough: encoding is a pure
	// function of the vocabulary. With the seat on, a typo'd
	// line must encode exactly like the corrected line.
	v := typoTestVocab("do", "you", "like", "gardening", "a", "lot")
	tok, err := tokenizer.NewTokenizer(v)
	if err != nil {
		t.Fatal(err)
	}
	base, err := seq2seq.NewSeq2Seq(v.Size(), v.Size(), 128, 256, tok, v)
	if err != nil {
		t.Fatal(err)
	}
	model, err := seq2seq.NewLayersModel(base, seq2seq.DefaultLayersConfig(), seq2seq.LayersDerivations{})
	if err != nil {
		t.Fatal(err)
	}
	off := &Brain{Model: model, tok: tok}
	on := &Brain{Model: model, tok: tok, FixTypos: true}

	typoIDs, _ := off.encodeLine("do you like gardning")
	if typoIDs[3] != v.UnkID {
		t.Fatalf("seat off: typo token id = %d, want UNK (%d)", typoIDs[3], v.UnkID)
	}
	fixedIDs, fixedSurfs := on.encodeLine("do you like gardning")
	wantIDs, _ := off.encodeLine("do you like gardening")
	if !reflect.DeepEqual(fixedIDs, wantIDs) {
		t.Fatalf("seat on: ids = %v, want the corrected line's ids %v", fixedIDs, wantIDs)
	}
	if fixedSurfs[3] != "gardening" {
		t.Fatalf("seat on: surface = %q, want gardening", fixedSurfs[3])
	}
}
