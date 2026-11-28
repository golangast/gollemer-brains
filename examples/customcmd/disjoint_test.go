package main

import (
	"math"
	"testing"

	"github.com/golangast/gollemer-brains/intent"
)

// tornClf is a hand-built classifier whose read of the
// line "thing" is torn: buildProb for build_server
// (kind make) against rivalProb for the rival label,
// with the third label negligible. Biases are log
// probabilities, so the softmax returns (very nearly)
// the stated probabilities. Its kind read abstains
// (uniform weights), and neither label's object word
// appears in "thing", so the fallback composes nothing:
// what the disjointness check does to the read is the
// whole outcome.
func tornClf(rival string, buildProb, rivalProb float64) *intent.Classifier {
	labels := []string{"build_server", "delete_file", "greeting"}
	kinds := map[string]string{
		"build_server": "make",
		"delete_file":  "remove",
		"greeting":     "talk",
	}
	bias := map[string]float64{
		"build_server": math.Log(buildProb),
		"delete_file":  -20,
		"greeting":     -20,
	}
	bias[rival] = math.Log(rivalProb)
	c := &intent.Classifier{
		Labels:      labels,
		Words:       []string{"thing"},
		Weights:     [][]float64{{0}, {0}, {0}},
		Threshold:   0.5,
		Kinds:       kinds,
		KindLabels:  []string{"make", "remove", "talk"},
		KindWeights: [][]float64{{0}, {0}, {0}},
		KindBias:    []float64{0, 0, 0},
	}
	for i, l := range labels {
		c.Bias = append(c.Bias, bias[l])
		_ = i
	}
	return c
}

func TestDisjointWithholdTornRead(t *testing.T) {
	d := &demo{clf: tornClf("delete_file", 0.55, 0.45)}
	r, ok := d.readIntent("thing")
	if !ok {
		t.Fatal("readIntent not ok with a wired classifier")
	}
	if r.label != "" {
		t.Errorf("torn read adopted %q, want withheld (empty label)", r.label)
	}
	if r.withheld != "build_server" || r.withheldRival != "delete_file" {
		t.Errorf("withheld = %q vs %q, want build_server vs delete_file", r.withheld, r.withheldRival)
	}
}

func TestDisjointClearMarginStands(t *testing.T) {
	d := &demo{clf: tornClf("delete_file", 0.75, 0.20)}
	r, _ := d.readIntent("thing")
	if r.label != "build_server" || r.withheld != "" {
		t.Errorf("clear-margin read = %+v, want build_server standing", r)
	}
}

func TestDisjointNonDisjointRivalStands(t *testing.T) {
	// The same torn shape, but the rival is a greeting
	// (kind talk): make and talk are not opposites, so
	// the read stands — the check is about polarity,
	// not about closeness alone.
	d := &demo{clf: tornClf("greeting", 0.55, 0.45)}
	r, _ := d.readIntent("thing")
	if r.label != "build_server" || r.withheld != "" {
		t.Errorf("non-disjoint torn read = %+v, want build_server standing", r)
	}
}

func TestKindsDisjoint(t *testing.T) {
	for _, p := range [][2]string{{"make", "remove"}, {"remove", "make"}, {"schedule", "remove"}, {"play", "remove"}} {
		if !kindsDisjoint(p[0], p[1]) {
			t.Errorf("kindsDisjoint(%q, %q) = false, want true", p[0], p[1])
		}
	}
	for _, p := range [][2]string{{"find", "talk"}, {"talk", "schedule"}, {"make", "make"}, {"find", "remove"}, {"", "remove"}, {"make", ""}} {
		if kindsDisjoint(p[0], p[1]) {
			t.Errorf("kindsDisjoint(%q, %q) = true, want false", p[0], p[1])
		}
	}
}
