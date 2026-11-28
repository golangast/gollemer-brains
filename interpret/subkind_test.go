package interpret_test

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/intent"
	"github.com/golangast/gollemer-brains/interpret"
)

// Piece 3 probe set: sub-kinds. The corpus files two
// channel meanings under ONE kind (find) in two
// sub-kinds (explain, locate), both named for the same
// object. At kind level the fallback's composition is
// ambiguous BY CONSTRUCTION — explain_channel and
// find_channel both match the object "channel" — so
// round 4's rule (ambiguity is final) abstains. Only
// the sub-kind level can resolve these lines: the
// narrow read's top label names the sub-kind, and
// composition inside it is unique. The control corpus
// is the same rows with the sub-kind field dropped:
// every probe line must abstain there.

func subkindPairs(withSubs bool) []data.Pair {
	var pairs []data.Pair
	add := func(label, kind, sub string, lines ...string) {
		for _, line := range lines {
			p := data.Pair{Input: line, Intent: label, Kind: kind}
			if withSubs {
				p.SubKind = sub
			}
			pairs = append(pairs, p)
		}
	}
	add("explain_channel", "find", "explain",
		"what is a channel", "explain the channel", "what does channel mean",
		"tell me about channels", "channel explained please", "meaning of a channel",
		"how does a channel work", "describe a channel for me")
	add("find_channel", "find", "locate",
		"where is the channel", "find the channel", "locate the channel",
		"find me the channel", "where can i find the channel", "look for the channel",
		"search for the channel", "hunt down the channel")
	add("explain_widget", "find", "explain",
		"what is a widget", "explain the widget", "what does widget mean",
		"tell me about widgets", "widget explained please", "meaning of a widget",
		"how does a widget work", "describe a widget for me")
	add("play_music", "play", "",
		"play some music", "put on a song", "start the music",
		"i want to hear music", "music please", "play a tune",
		"turn on some songs", "queue up a song")
	add("cancel_music", "remove", "",
		"stop the music", "cancel the song", "no more music",
		"turn the music off", "stop playing songs", "kill the music",
		"end the song", "halt the music")
	add("move_channel", "make", "move",
		"move the channel to the folder", "shift the channel over",
		"move the channel please", "relocate the channel")
	add("delete_channel", "remove", "erase",
		"delete the channel", "remove the channel",
		"erase the channel", "drop the channel")
	return pairs
}

func TestSubKindProbeSet(t *testing.T) {
	clf, err := intent.Train(subkindPairs(true), intent.DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	ctrl, err := intent.Train(subkindPairs(false), intent.DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	factors := interpret.DeriveFactors(clf)
	ctrlFactors := interpret.DeriveFactors(ctrl)

	// Premise: at kind level the composition is exactly
	// as ambiguous as designed — two candidates.
	words := interpret.LineObjectWords("what is a channel")
	if _, n := factors.ComposeIn(words, "find"); n != 2 {
		t.Fatalf("premise broken: ComposeIn(find) = %d candidates, want 2", n)
	}

	probes := []struct {
		line      string
		kindGuess string
		want      string
	}{
		// Kind guessed, composition ambiguous at kind
		// level; the narrow read's sub-kind resolves it.
		{"what is a channel", "find", "explain_channel"},
		// No kind guess: the walk climbs label →
		// sub-kind and composes there.
		{"where is the channel", "", "find_channel"},
	}
	right, wrong, abstain := 0, 0, 0
	for _, p := range probes {
		label, ok, trace := factors.FallbackRead(clf, p.line, p.kindGuess)
		switch {
		case ok && label == p.want:
			right++
			if !strings.Contains(trace, "sub-kind") {
				t.Errorf("probe %q: adopted %s but the trace names no sub-kind step: %q", p.line, label, trace)
			}
		case ok:
			wrong++
			t.Errorf("probe %q: adopted %s, want %s", p.line, label, p.want)
		default:
			abstain++
			t.Errorf("probe %q: abstained, want %s (trace %q)", p.line, p.want, trace)
		}
		// Control: without sub-kinds the same read
		// abstains — the sub-kind level is the only path.
		if cl, ok, _ := ctrlFactors.FallbackRead(ctrl, p.line, p.kindGuess); ok {
			t.Errorf("control probe %q resolved to %s WITHOUT sub-kinds — the probe proves nothing", p.line, cl)
		}
	}
	t.Logf("piece 3 probe set: %d right, %d wrong, %d abstain (of %d); control abstains on all", right, wrong, abstain, len(probes))
	if wrong > 0 || abstain > 0 {
		t.Fatalf("probe set: %d wrong, %d abstain", wrong, abstain)
	}
}

// Inheritance: a sub-kind's labels answer kind-level
// questions with their TOP kind — the disjointness
// pairs, stated at kind level, reach labels filed two
// levels down. move_channel (sub-kind move, under make)
// torn against delete_channel (sub-kind erase, under
// remove) withholds exactly as a kind-level pair does.
func TestSubKindInheritsDisjointness(t *testing.T) {
	clf, err := intent.Train(subkindPairs(true), intent.DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	tree := interpret.DeriveFactors(clf).Tree
	if got := tree.TopKind("move"); got != "make" {
		t.Fatalf("TopKind(move) = %q, want make", got)
	}
	if got := tree.Climb("move_channel"); len(got) != 2 || got[0] != "move" || got[1] != "make" {
		t.Fatalf("Climb(move_channel) = %v, want [move make]", got)
	}
	ranked := []intent.LabelScore{
		{Label: "move_channel", Prob: 0.54},
		{Label: "delete_channel", Prob: 0.46},
	}
	rival, ok := interpret.DisjointWithhold(clf.KindOf, ranked)
	if !ok || rival != "delete_channel" {
		t.Errorf("torn sub-kind read: rival=%q ok=%v, want delete_channel withheld", rival, ok)
	}
}
