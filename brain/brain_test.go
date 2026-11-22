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

// chatPairsForIntents is a small chat set a brain can train
// on while intent rows ride along.
func chatPairsForIntents() []data.Pair {
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
	return pairs
}

func TestTrainWritesIntentSidecarAndLoadBrainReadsIt(t *testing.T) {
	pairs := chatPairsForIntents()
	// Intent rows: most label-only, one per label carrying
	// an output — both shapes must train the classifier,
	// and the label-only ones must not break the brain.
	pairs = append(pairs,
		data.Pair{Input: "turn on the lights", Intent: "lights_on"},
		data.Pair{Input: "lights on please", Intent: "lights_on"},
		data.Pair{Input: "switch the lights on", Output: "Lights on.", Intent: "lights_on"},
		data.Pair{Input: "turn off the lights", Intent: "lights_off"},
		data.Pair{Input: "lights off please", Intent: "lights_off"},
		data.Pair{Input: "switch the lights off", Output: "Lights off.", Intent: "lights_off"},
	)
	out := filepath.Join(t.TempDir(), "brain.gob")
	if err := Train(pairs, out, TrainOptions{Epochs: 10}); err != nil {
		t.Fatalf("Train: %v", err)
	}
	if _, err := os.Stat(out + ".intents.json"); err != nil {
		t.Fatalf("intent sidecar not written: %v", err)
	}
	b, err := LoadBrain(out)
	if err != nil {
		t.Fatalf("LoadBrain: %v", err)
	}
	if b.Intents == nil {
		t.Fatal("LoadBrain did not pick up the intent sidecar")
	}
	if label, _ := b.Intents.Predict("turn on the lights"); label != "lights_on" {
		t.Errorf("Predict(turn on the lights) = %q, want lights_on", label)
	}
	if label, _ := b.Intents.Predict("turn off the lights"); label != "lights_off" {
		t.Errorf("Predict(turn off the lights) = %q, want lights_off", label)
	}

	// Retrain at the same path WITHOUT intents: the stale
	// sidecar must not survive to be mistaken for this
	// brain's classifier.
	if err := Train(chatPairsForIntents(), out, TrainOptions{Epochs: 10}); err != nil {
		t.Fatalf("retrain: %v", err)
	}
	if _, err := os.Stat(out + ".intents.json"); !os.IsNotExist(err) {
		t.Fatalf("stale intent sidecar survived a no-intent retrain (stat err = %v)", err)
	}
	b2, err := LoadBrain(out)
	if err != nil {
		t.Fatalf("LoadBrain after retrain: %v", err)
	}
	if b2.Intents != nil {
		t.Error("brain retrained without intents should load with Intents == nil")
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
