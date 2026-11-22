package brain

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/data"
)

// TestTrainSaveLoadGenerate is the tiny end-to-end: train a
// handful of pairs for a few epochs in a temp dir, save, load
// the brain back from its .gob alone, and generate — the reply
// must be built from words the training data taught it.
func TestTrainSaveLoadGenerate(t *testing.T) {
	var pairs []data.Pair
	qs := []struct{ q, a string }{
		{"hello", "Hello! How are you today?"},
		{"hi", "Hi there! How are you?"},
		{"what is your name", "I am a tiny brain."},
		{"do you like gardening", "Yes, I like gardening a lot."},
		{"do you like cooking", "Yes, I like cooking a lot."},
		{"tell me about gardening", "Gardening is peaceful and calm."},
		{"tell me about cooking", "Cooking is a happy skill."},
		{"goodbye", "Goodbye! Take care."},
		{"thank you", "You are very welcome."},
		{"how are you", "I am doing well, thank you."},
		{"i like cats", "Cats are quiet, funny friends."},
		{"tell me about cats", "Cats are quiet, funny friends."},
	}
	for _, qa := range qs {
		pairs = append(pairs, data.Pair{Input: qa.q, Output: qa.a})
	}

	out := filepath.Join(t.TempDir(), "brain.gob")
	if err := Train(pairs, out, TrainOptions{Epochs: 60}); err != nil {
		t.Fatalf("Train: %v", err)
	}
	if _, err := os.Stat(out); err != nil {
		t.Fatalf("model not written: %v", err)
	}
	// A completed run cleans its checkpoints up.
	for _, suffix := range []string{".ckpt", ".ckpt.json", ".latest", ".latest.json"} {
		if _, err := os.Stat(out + suffix); !os.IsNotExist(err) {
			t.Errorf("checkpoint %s left behind after completion", suffix)
		}
	}

	b, err := LoadBrain(out)
	if err != nil {
		t.Fatalf("LoadBrain: %v", err)
	}
	ans, err := b.Answer("hello")
	if err != nil {
		t.Fatalf("Answer: %v", err)
	}
	if strings.TrimSpace(ans) == "" {
		t.Fatal("empty answer")
	}
	// Every word of the reply must come from the training
	// vocabulary (punctuation aside): the brain can only say
	// what it was taught.
	v := b.Model.Base.OutputVocab
	for _, w := range strings.Fields(strings.ToLower(ans)) {
		w = strings.Trim(w, ".,!?")
		if w == "" {
			continue
		}
		if id := v.GetTokenID(w); id == v.UnkID {
			t.Errorf("answer word %q is not in the training vocabulary (answer: %q)", w, ans)
		}
	}
}

func TestTrainRejectsDroppedLayers(t *testing.T) {
	pairs := make([]data.Pair, 12)
	for i := range pairs {
		pairs[i] = data.Pair{Input: "hello", Output: "Hi!"}
	}
	err := Train(pairs, filepath.Join(t.TempDir(), "b.gob"), TrainOptions{Epochs: 1, Layers: []string{"frame"}})
	if err == nil || !strings.Contains(err.Error(), "frame") {
		t.Fatalf("err = %v, want a clear dropped-layer error", err)
	}
}

func TestReplyCarries(t *testing.T) {
	if !ReplyCarries("tell me about gardening", "Gardening is peaceful.") {
		t.Error("a reply naming the topic should carry")
	}
	if ReplyCarries("tell me about gardening", "I like warm soup.") {
		t.Error("an off-topic reply should not carry")
	}
	if ReplyCarries("hello", "") {
		t.Error("an empty reply never carries")
	}
}
