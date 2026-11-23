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
	if err := os.WriteFile(path, []byte("{\"labels\":[\"a\"],\"scheme\":\"words\",\"weights\":[],\"bias\":[]}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil || !strings.Contains(err.Error(), "inconsistent") {
		t.Fatalf("err = %v, want an inconsistent-shapes error", err)
	}
}

// A sidecar written before feature schemes existed has no
// "scheme" field. Loading it must fail loudly — with the
// remedy named — rather than score its word-trained weights
// with a piece featurizer and quietly change its answers.
func TestLoadRejectsPreSchemeSidecar(t *testing.T) {
	path := filepath.Join(t.TempDir(), "old.intents.json")
	old := "{\"labels\":[\"a\",\"b\"],\"words\":[\"hi\"],\"weights\":[[0.1],[0.2]],\"bias\":[0,0],\"threshold\":0.5,\"rows\":2}"
	if err := os.WriteFile(path, []byte(old), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Load(path)
	if err == nil || !strings.Contains(err.Error(), "retrain") {
		t.Fatalf("err = %v, want a retrain-to-upgrade error", err)
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

// sprocketPairs separates two labels by a single word
// family: everything else in the lines is shared. A line
// whose sprocket-word is a form the classifier never saw
// ("sprockets") offers a whole-words classifier nothing to
// tell the labels apart; piece features still can.
func sprocketPairs() []data.Pair {
	return []data.Pair{
		{Input: "turn the sprocket clockwise", Intent: "alpha"},
		{Input: "turn the sprocket slowly", Intent: "alpha"},
		{Input: "sprocket turning now", Intent: "alpha"},
		{Input: "turn the handle clockwise", Intent: "beta"},
		{Input: "turn the handle slowly", Intent: "beta"},
		{Input: "handle turning now", Intent: "beta"},
	}
}

func TestSubwordFeaturesFireForUnseenWord(t *testing.T) {
	words, err := TrainWith(sprocketPairs(), DefaultThreshold, Options{Scheme: SchemeWords})
	if err != nil {
		t.Fatal(err)
	}
	if label, _ := words.Predict("turn the sprockets clockwise"); label == "alpha" {
		t.Fatal("whole-words classifier unexpectedly resolved the unseen word")
	}
	pieces, err := TrainWith(sprocketPairs(), DefaultThreshold, Options{Scheme: SchemeWordsNgrams})
	if err != nil {
		t.Fatal(err)
	}
	label, conf := pieces.Predict("turn the sprockets clockwise")
	if label != "alpha" {
		t.Fatalf("Predict(unseen plural) = %q (%.2f), want alpha — its pieces are sprocket's", label, conf)
	}
}

// kindPairs gives four labels three kinds, so both the
// label classifier and the kind classifier have a choice.
func kindPairs() []data.Pair {
	return []data.Pair{
		{Input: "hello there friend", Intent: "greet", Kind: "talk"},
		{Input: "hello", Intent: "greet", Kind: "talk"},
		{Input: "hi there", Intent: "greet", Kind: "talk"},
		{Input: "goodbye now", Intent: "farewell", Kind: "talk"},
		{Input: "bye friend", Intent: "farewell", Kind: "talk"},
		{Input: "see you later", Intent: "farewell", Kind: "talk"},
		{Input: "create a file", Intent: "create_file", Kind: "make"},
		{Input: "make a new file", Intent: "create_file", Kind: "make"},
		{Input: "start a document", Intent: "create_file", Kind: "make"},
		{Input: "delete the file", Intent: "delete_file", Kind: "remove"},
		{Input: "remove the document", Intent: "delete_file", Kind: "remove"},
		{Input: "get rid of the file", Intent: "delete_file", Kind: "remove"},
	}
}

func TestKindPrediction(t *testing.T) {
	c, err := Train(kindPairs(), DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	if !c.HasKinds() {
		t.Fatal("classifier trained on kinded rows reports no kinds")
	}
	if got := c.KindOf("create_file"); got != "make" {
		t.Fatalf("KindOf(create_file) = %q, want make", got)
	}
	if got := c.KindOf("greet"); got != "talk" {
		t.Fatalf("KindOf(greet) = %q, want talk", got)
	}
	if kind, conf := c.PredictKind("hello there"); kind != "talk" {
		t.Fatalf("PredictKind(hello there) = %q (%.2f), want talk", kind, conf)
	}
	if kind, _ := c.PredictKind("delete the document please"); kind != "remove" {
		t.Fatalf("PredictKind(delete line) = %q, want remove", kind)
	}
	d := c.Describe("create a file")
	if !strings.Contains(d, "create_file") || !strings.Contains(d, "· kind: make") {
		t.Fatalf("Describe with kinds = %q, want the label and its kind", d)
	}
}

func TestNoKindsDescribeUnchanged(t *testing.T) {
	c, err := Train(toyPairs(), DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	if c.HasKinds() {
		t.Fatal("classifier trained without kinds reports kinds")
	}
	if kind, conf := c.PredictKind("hello there"); kind != "" || conf != 0 {
		t.Fatalf("PredictKind without kinds = %q, %.2f; want abstention at 0", kind, conf)
	}
	if got := c.Describe("hello there"); strings.Contains(got, "kind") {
		t.Fatalf("Describe without kinds = %q, want the old format untouched", got)
	}
}

func TestKindConflictRejected(t *testing.T) {
	pairs := kindPairs()
	pairs[0].Kind = "make" // greet now claims talk (rows 1-2) and make (row 0)
	_, err := Train(pairs, DefaultThreshold)
	if err == nil || !strings.Contains(err.Error(), "greet") {
		t.Fatalf("err = %v, want a kind-conflict error naming the label", err)
	}
}

func TestKindsSaveLoadRoundTrip(t *testing.T) {
	c, err := Train(kindPairs(), DefaultThreshold)
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
	if !back.HasKinds() || back.KindOf("delete_file") != "remove" {
		t.Fatalf("round trip lost kinds: %v", back.LabelKinds())
	}
	for _, line := range []string{"hello there", "create a document", "goodbye now"} {
		k1, p1 := c.PredictKind(line)
		k2, p2 := back.PredictKind(line)
		if k1 != k2 || p1 != p2 {
			t.Errorf("round trip changed PredictKind(%q): %q %.4f -> %q %.4f", line, k1, p1, k2, p2)
		}
		if c.Describe(line) != back.Describe(line) {
			t.Errorf("round trip changed Describe(%q): %q -> %q", line, c.Describe(line), back.Describe(line))
		}
	}
}

func TestCountByLabel(t *testing.T) {
	counts := CountByLabel(toyPairs())
	if counts["greet"] != 5 || counts["farewell"] != 4 {
		t.Fatalf("counts = %v", counts)
	}
}
