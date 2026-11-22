package intent

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/data"
)

// toyPairs is a two-meaning command set: greetings and
// farewells with disjoint vocabularies, so a working
// classifier has no excuse.
func toyPairs() []data.Pair {
	return []data.Pair{
		{Input: "hello there friend", Intent: "greet"},
		{Input: "hello", Intent: "greet"},
		{Input: "hi there", Intent: "greet"},
		{Input: "hey friend hello", Intent: "greet"},
		{Input: "goodbye now", Intent: "farewell"},
		{Input: "bye friend", Intent: "farewell"},
		{Input: "see you later", Intent: "farewell"},
		{Input: "goodbye see you", Intent: "farewell"},
		// One row with an output too: it must train the
		// classifier exactly like the label-only rows.
		{Input: "hey there", Output: "Hello!", Intent: "greet"},
	}
}

func TestTrainLearnsToySet(t *testing.T) {
	c, err := Train(toyPairs(), DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range Rows(toyPairs()) {
		label, conf := c.Predict(p.Input)
		if label != p.Intent {
			t.Errorf("Predict(%q) = %q (%.2f), want %q", p.Input, label, conf, p.Intent)
		}
	}
}

func TestTrainDeterministic(t *testing.T) {
	a, err := Train(toyPairs(), DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Train(toyPairs(), DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Weights, b.Weights) || !reflect.DeepEqual(a.Bias, b.Bias) {
		t.Fatal("two trainings on the same rows produced different classifiers")
	}
}

func TestPredictAbstainsWithoutEvidence(t *testing.T) {
	c, err := Train(toyPairs(), DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	// No word of this line was ever trained on: no evidence,
	// no label, whatever the biases say.
	label, conf := c.Predict("zanzibar quixotic lullaby")
	if label != "" || conf != 0 {
		t.Fatalf("Predict(unknown words) = %q, %.2f; want abstention at 0", label, conf)
	}
	if got := c.Describe("zanzibar quixotic lullaby"); got != "(none confident enough)" {
		t.Fatalf("Describe = %q", got)
	}
}

func TestThresholdAbstains(t *testing.T) {
	c, err := Train(toyPairs(), DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	label, conf := c.Predict("hello there friend")
	if label == "" {
		t.Fatal("expected a confident prediction to raise the threshold against")
	}
	c.Threshold = conf + (1-conf)/2 // halfway between confidence and certainty
	if label, _ := c.Predict("hello there friend"); label != "" {
		t.Fatalf("with threshold above the confidence, Predict = %q, want abstention", label)
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	c, err := Train(toyPairs(), DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "brain.gob.intents.json")
	if err := c.Save(path); err != nil {
		t.Fatal(err)
	}
	back, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range []string{"hello there", "goodbye now", "zanzibar quixotic"} {
		l1, p1 := c.Predict(line)
		l2, p2 := back.Predict(line)
		if l1 != l2 || p1 != p2 {
			t.Errorf("round trip changed Predict(%q): %q %.4f -> %q %.4f", line, l1, p1, l2, p2)
		}
	}
}

func TestLoadRejectsCorruptSidecar(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.intents.json")
	if err := os.WriteFile(path, []byte("{\"labels\":[\"a\"],\"weights\":[],\"bias\":[]}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil || !strings.Contains(err.Error(), "inconsistent") {
		t.Fatalf("err = %v, want an inconsistent-shapes error", err)
	}
}

func TestTrainNeedsRowsAndChoice(t *testing.T) {
	if _, err := Train([]data.Pair{{Input: "hello", Output: "Hi!"}}, DefaultThreshold); err == nil {
		t.Fatal("training with no intent rows should error")
	}
	one := []data.Pair{
		{Input: "hello", Intent: "greet"},
		{Input: "hi", Intent: "greet"},
	}
	if _, err := Train(one, DefaultThreshold); err == nil {
		t.Fatal("training with a single label should error")
	}
}

func TestCountByLabel(t *testing.T) {
	counts := CountByLabel(toyPairs())
	if counts["greet"] != 5 || counts["farewell"] != 4 {
		t.Fatalf("counts = %v", counts)
	}
}
