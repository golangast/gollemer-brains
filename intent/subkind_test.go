package intent_test

import (
	"testing"

	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/intent"
)

// Sub-kinds: the middle level of the kind tree, learned
// from rows that carry one, validated like kinds are.

func subkindPairs() []data.Pair {
	var pairs []data.Pair
	add := func(label, kind, sub string, lines ...string) {
		for _, line := range lines {
			pairs = append(pairs, data.Pair{Input: line, Intent: label, Kind: kind, SubKind: sub})
		}
	}
	add("explain_channel", "find", "explain",
		"what is a channel", "explain the channel", "meaning of a channel")
	add("find_channel", "find", "locate",
		"where is the channel", "find the channel", "locate the channel")
	add("play_music", "play", "",
		"play some music", "start the music", "music please")
	return pairs
}

func TestTrainLearnsSubKinds(t *testing.T) {
	clf, err := intent.Train(subkindPairs(), intent.DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	if got := clf.SubKindOf("explain_channel"); got != "explain" {
		t.Errorf("SubKindOf(explain_channel) = %q, want explain", got)
	}
	if got := clf.SubKindOf("play_music"); got != "" {
		t.Errorf("SubKindOf(play_music) = %q, want none", got)
	}
	if got := clf.SubKindParent("locate"); got != "find" {
		t.Errorf("SubKindParent(locate) = %q, want find", got)
	}
	// The top level never moves: a sub-kind label's kind
	// is still its kind.
	if got := clf.KindOf("find_channel"); got != "find" {
		t.Errorf("KindOf(find_channel) = %q, want find", got)
	}
}

func TestSubKindValidation(t *testing.T) {
	bad := []struct {
		name  string
		pairs []data.Pair
	}{
		{"two sub-kinds on one label", []data.Pair{
			{Input: "a", Intent: "x_one", Kind: "find", SubKind: "explain"},
			{Input: "b", Intent: "x_one", Kind: "find", SubKind: "locate"},
			{Input: "c", Intent: "y_two", Kind: "play"},
			{Input: "d", Intent: "y_two", Kind: "play"},
		}},
		{"one sub-kind under two kinds", []data.Pair{
			{Input: "a", Intent: "x_one", Kind: "find", SubKind: "explain"},
			{Input: "b", Intent: "y_two", Kind: "play", SubKind: "explain"},
		}},
		{"sub-kind without a kind", []data.Pair{
			{Input: "a", Intent: "x_one", SubKind: "explain"},
			{Input: "b", Intent: "y_two", Kind: "play"},
			{Input: "c", Intent: "y_two", Kind: "play"},
		}},
	}
	for _, b := range bad {
		if _, err := intent.Train(b.pairs, intent.DefaultThreshold); err == nil {
			t.Errorf("%s: Train passed, want a data error", b.name)
		}
	}
}
