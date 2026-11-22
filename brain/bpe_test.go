package brain

import (
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/data"
)

func bpeToyPairs() []data.Pair {
	return []data.Pair{
		{Input: "i love gardening", Output: "Gardening is peaceful."},
		{Input: "do you like painting", Output: "Painting is fun."},
		{Input: "i like reading books", Output: "Reading takes you anywhere."},
		{Input: "tell me about fishing", Output: "Fishing teaches patience."},
		{Input: "how is the weather", Output: "The sun is warm today."},
		{Input: "i love cooking food", Output: "Cooking is a happy skill."},
		{Input: "do you enjoy hiking", Output: "Hiking clears the mind."},
		{Input: "cats are cute", Output: "Cats are quiet friends."},
		{Input: "dogs are loyal", Output: "Dogs are joyful friends."},
		{Input: "i like music", Output: "Music lifts the heart."},
		{Input: "tell me about trains", Output: "Trains are calm."},
		{Input: "gardening is fun", Output: "Yes, gardening is peaceful work."},
		{Input: "painting is relaxing", Output: "Painting keeps a moment."},
		{Input: "reading is great", Output: "A good book is a friend."},
	}
}

func TestBPETrainSaveLoadAnswer(t *testing.T) {
	out := filepath.Join(t.TempDir(), "b.gob")
	if err := Train(bpeToyPairs(), out, TrainOptions{
		Layers: []string{"copy"}, Epochs: 3, Tokenizer: "bpe", BPEMerges: 200,
	}); err != nil {
		t.Fatalf("Train(bpe): %v", err)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if first := strings.SplitN(string(raw), "\n", 2)[0]; first != brainFileMagic {
		t.Fatalf("saved brain's first line = %q, want the wrapped-format magic", first)
	}
	br, err := LoadBrain(out)
	if err != nil {
		t.Fatalf("LoadBrain(bpe): %v", err)
	}
	if br.bpeTok == nil {
		t.Fatal("loaded brain has no BPE tokenizer")
	}
	ans, err := br.Answer("i love gardening")
	if err != nil {
		t.Fatalf("Answer: %v", err)
	}
	if strings.Contains(ans, "##") {
		t.Fatalf("reply leaks BPE continuation markers: %q", ans)
	}
	if _, err := br.AnswerChecked("do you like painting", rand.New(rand.NewSource(7))); err != nil {
		t.Fatalf("AnswerChecked: %v", err)
	}
	// An unseen word must encode to pieces, never to a lone
	// unknown: every id comes from a real piece form.
	ids, surfs := br.encodeLine("i love knitting")
	unk := br.Model.Base.OutputVocab.UnkID
	for i, id := range ids {
		if id == unk {
			t.Fatalf("encodeLine(knitting) produced UNK at %d (%v)", i, surfs)
		}
	}
	joined := ""
	for _, s := range surfs {
		joined += s + " "
	}
	if !strings.Contains(joined, "knitt") && !strings.Contains(joined, "##") {
		t.Fatalf("knitting did not decompose into pieces: %v", surfs)
	}
}

func TestWordBrainStillLoadsUnwrapped(t *testing.T) {
	out := filepath.Join(t.TempDir(), "w.gob")
	if err := Train(bpeToyPairs(), out, TrainOptions{
		Layers: []string{"copy"}, Epochs: 3,
	}); err != nil {
		t.Fatalf("Train(word): %v", err)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if strings.HasPrefix(string(raw), brainFileMagic) {
		t.Fatal("word brain was wrapped; it must stay the bare gob")
	}
	br, err := LoadBrain(out)
	if err != nil {
		t.Fatalf("LoadBrain(word): %v", err)
	}
	if br.bpeTok != nil {
		t.Fatal("word brain loaded with a BPE tokenizer")
	}
}

func TestBrainFileMismatchErrors(t *testing.T) {
	dir := t.TempDir()
	cases := map[string]string{
		"unknown kind":  brainFileMagic + "\n{\"kind\":\"wat\"}\njunk",
		"bpe no merges": brainFileMagic + "\n{\"kind\":\"bpe\"}\njunk",
		"bad header":    brainFileMagic + "\nnot json\njunk",
	}
	for name, content := range cases {
		p := filepath.Join(dir, name+".gob")
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadBrain(p); err == nil {
			t.Errorf("%s: LoadBrain succeeded, want a loud error", name)
		} else {
			t.Logf("%s: %v", name, err)
		}
	}
	if err := Train(bpeToyPairs(), filepath.Join(dir, "x.gob"), TrainOptions{
		Layers: []string{"copy"}, Epochs: 1, Tokenizer: "wat",
	}); err == nil {
		t.Error("Train with tokenizer \"wat\" succeeded, want an error")
	}
}
