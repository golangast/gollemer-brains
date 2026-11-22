// Package intent ties a sentence to a MEANING the user
// defines: an intent label like "lights_off", carried on a
// training row's "intent" field (see the data package). A
// program using a brain can switch on the predicted label and
// run whatever command it binds to that meaning — the repo
// supplies the mechanism; the labels and the commands are
// the user's.
//
// The classifier is deliberately small: multinomial logistic
// regression (a bag-of-words softmax) over the words of the
// intent rows it was trained on, tokenized the way the brain
// tokenizes. Standard library only, trained by seeded SGD,
// so the same rows always produce the same classifier.
//
// Abstention is a feature, not a failure: below the
// confidence threshold, Predict returns NO label. A program
// that acts on predictions will sometimes be handed a
// sentence the classifier barely understands, and a wrong
// command is worse than none.
package intent

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"unicode"

	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/tokenizer"
)

// DefaultThreshold is the confidence below which Predict
// abstains: the best label's probability must reach it.
const DefaultThreshold = 0.5

// trainSeed fixes the SGD shuffle stream: the same rows
// always train to the same classifier.
const trainSeed = 20261121

const (
	trainEpochs = 400
	trainLR     = 0.5
	trainL2     = 1e-4
)

// Classifier is a trained intent classifier. The exported
// fields are its persisted form (see Save/Load): everything
// Predict needs travels in one JSON sidecar file.
type Classifier struct {
	Labels    []string    `json:"labels"`    // sorted label names
	Words     []string    `json:"words"`     // feature order: sorted vocabulary words
	Weights   [][]float64 `json:"weights"`   // [label][feature]
	Bias      []float64   `json:"bias"`      // [label]
	Threshold float64     `json:"threshold"` // abstain below this confidence
	Rows      int         `json:"rows"`      // intent rows it was trained on

	wordIdx map[string]int // rebuilt by Train and Load; not persisted
}

// lineWords splits a line into feature words: the brain's own
// tokenizer split, lowercased, keeping only tokens that
// contain a letter (punctuation carries no meaning here).
func lineWords(line string) []string {
	toks := tokenizer.Tokenize(strings.ToLower(line))
	out := make([]string, 0, len(toks))
	for _, t := range toks {
		for _, r := range t {
			if unicode.IsLetter(r) {
				out = append(out, t)
				break
			}
		}
	}
	return out
}

// Rows returns the pairs that carry an intent — the
// classifier's training rows, with or without an output.
func Rows(pairs []data.Pair) []data.Pair {
	var out []data.Pair
	for _, p := range pairs {
		if p.Intent != "" {
			out = append(out, p)
		}
	}
	return out
}

// CountByLabel counts intent rows per label.
func CountByLabel(pairs []data.Pair) map[string]int {
	counts := map[string]int{}
	for _, p := range pairs {
		if p.Intent != "" {
			counts[p.Intent]++
		}
	}
	return counts
}

// Train fits a classifier on every pair that carries an
// intent. It errors when there are no intent rows or only
// one label (a classifier needs a choice to learn).
func Train(pairs []data.Pair, threshold float64) (*Classifier, error) {
	rows := Rows(pairs)
	if len(rows) == 0 {
		return nil, fmt.Errorf("intent: no rows carry an \"intent\" field")
	}
	counts := CountByLabel(rows)
	if len(counts) < 2 {
		return nil, fmt.Errorf("intent: only one label (%v) — a classifier needs at least two meanings to choose between", keys(counts))
	}

	labelSet := map[string]bool{}
	wordSet := map[string]bool{}
	for _, p := range rows {
		labelSet[p.Intent] = true
		for _, w := range lineWords(p.Input) {
			wordSet[w] = true
		}
	}
	c := &Classifier{Threshold: threshold, Rows: len(rows)}
	c.Labels = keys(labelSet)
	c.Words = keys(wordSet)
	c.wordIdx = indexOf(c.Words)
	labelIdx := indexOf(c.Labels)

	nL, nW := len(c.Labels), len(c.Words)
	c.Weights = make([][]float64, nL)
	for i := range c.Weights {
		c.Weights[i] = make([]float64, nW)
	}
	c.Bias = make([]float64, nL)

	// Precompute each row's feature counts and label.
	type example struct {
		x []float64
		y int
	}
	examples := make([]example, len(rows))
	for i, p := range rows {
		examples[i] = example{x: c.counts(p.Input), y: labelIdx[p.Intent]}
	}

	// Seeded SGD on the softmax cross-entropy. Zero-init
	// weights make the start deterministic; the seeded
	// shuffle makes every epoch's order deterministic too.
	rng := rand.New(rand.NewSource(trainSeed))
	grad := make([]float64, nW)
	for epoch := 0; epoch < trainEpochs; epoch++ {
		order := rng.Perm(len(examples))
		for _, idx := range order {
			ex := examples[idx]
			probs := softmax(c.logits(ex.x))
			for l := 0; l < nL; l++ {
				delta := probs[l]
				if l == ex.y {
					delta--
				}
				if delta == 0 {
					continue
				}
				for j := 0; j < nW; j++ {
					grad[j] = delta*ex.x[j] + trainL2*c.Weights[l][j]
				}
				for j := 0; j < nW; j++ {
					c.Weights[l][j] -= trainLR * grad[j]
				}
				c.Bias[l] -= trainLR * delta
			}
		}
	}
	return c, nil
}

// counts turns a line into feature counts over the
// classifier's vocabulary. Unknown words are ignored — they
// are evidence the classifier never saw, not evidence for
// any label.
func (c *Classifier) counts(line string) []float64 {
	x := make([]float64, len(c.Words))
	for _, w := range lineWords(line) {
		if j, ok := c.wordIdx[w]; ok {
			x[j]++
		}
	}
	return x
}

func (c *Classifier) logits(x []float64) []float64 {
	out := make([]float64, len(c.Labels))
	for l := range c.Labels {
		s := c.Bias[l]
		for j, v := range x {
			if v != 0 {
				s += c.Weights[l][j] * v
			}
		}
		out[l] = s
	}
	return out
}

func softmax(logits []float64) []float64 {
	max := math.Inf(-1)
	for _, v := range logits {
		if v > max {
			max = v
		}
	}
	out := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		out[i] = math.Exp(v - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Predict returns the most likely meaning of a line and the
// classifier's confidence in it. Below the threshold — or
// when the line shares no words with anything the classifier
// was trained on — the label is empty: the classifier
// abstains, and the caller should do nothing. The confidence
// is still reported (it is the best label's probability, or
// 0 when there was no evidence at all).
func (c *Classifier) Predict(line string) (string, float64) {
	x := c.counts(line)
	evidence := 0.0
	for _, v := range x {
		evidence += v
	}
	if evidence == 0 {
		return "", 0
	}
	probs := softmax(c.logits(x))
	best := 0
	for i := range probs {
		if probs[i] > probs[best] {
			best = i
		}
	}
	if probs[best] < c.Threshold {
		return "", probs[best]
	}
	return c.Labels[best], probs[best]
}

// Describe renders one line's prediction the way cmd/ask
// prints it: "lights_off (0.87)", or the abstention.
func (c *Classifier) Describe(line string) string {
	label, conf := c.Predict(line)
	if label == "" {
		return "(none confident enough)"
	}
	return fmt.Sprintf("%s (%.2f)", label, conf)
}

// Save writes the classifier as JSON — the sidecar that
// travels with a brain file (<model>.intents.json).
func (c *Classifier) Save(path string) error {
	raw, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

// Load reads a classifier written by Save. A file whose
// shapes do not agree with itself is an error, never a
// silently half-working classifier.
func Load(path string) (*Classifier, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Classifier
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("intent: bad sidecar %s: %w", path, err)
	}
	if len(c.Labels) == 0 || len(c.Weights) != len(c.Labels) || len(c.Bias) != len(c.Labels) {
		return nil, fmt.Errorf("intent: sidecar %s has inconsistent shapes (labels=%d weights=%d bias=%d)", path, len(c.Labels), len(c.Weights), len(c.Bias))
	}
	for i, row := range c.Weights {
		if len(row) != len(c.Words) {
			return nil, fmt.Errorf("intent: sidecar %s weight row %d has %d features, want %d", path, i, len(row), len(c.Words))
		}
	}
	if c.Threshold == 0 {
		c.Threshold = DefaultThreshold
	}
	c.wordIdx = indexOf(c.Words)
	return &c, nil
}

func keys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func indexOf(words []string) map[string]int {
	m := make(map[string]int, len(words))
	for i, w := range words {
		m[w] = i
	}
	return m
}
