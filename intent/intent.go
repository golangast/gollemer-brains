// Package intent ties a sentence to a MEANING the user
// defines: an intent label like "lights_off", carried on a
// training row's "intent" field (see the data package). A
// program using a brain can switch on the predicted label and
// run whatever command it binds to that meaning — the repo
// supplies the mechanism; the labels and the commands are
// the user's.
//
// Two mechanisms keep the classifier useful past the exact
// sentences it was trained on:
//
//   - Pieces. A line's features are its words AND the words'
//     subword pieces (character n-grams, or BPE pieces — the
//     scheme is chosen by measurement and recorded in the
//     sidecar). A word the classifier never saw still
//     registers through the pieces it shares with words it
//     did: "creat" is mostly "create", "remid" is mostly
//     "remind". Whole-word features stay; pieces add.
//   - Kinds. A row may carry a "kind" — the broad family of
//     its meaning ("make", "schedule", "talk"). Alongside
//     the label classifier, a second classifier learns the
//     kinds from the same features. A meaning the classifier
//     has never seen cannot get a label, but it can inherit
//     the right neighbourhood: PredictKind says what KIND of
//     thing the line is, even when Predict abstains.
//
// The classifiers are deliberately small: multinomial
// logistic regression (softmax) over the feature counts,
// trained by seeded SGD, so the same rows always produce
// the same classifiers. Standard library only.
//
// Abstention is a feature, not a failure: below the
// confidence threshold, Predict returns NO label. A program
// that acts on predictions will sometimes be handed a
// sentence the classifier barely understands, and a wrong
// command is worse than none. (With pieces, "shares no
// words with training" no longer means "no evidence" — a
// foreign word can still muster piece evidence. The
// threshold is the guard, as it always was.)
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

	"github.com/golangast/gollemer-brains/bpe"
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

// Scheme names a feature scheme: which features a line is
// scored on. The scheme is recorded in every sidecar Train
// writes, and Load featurizes exactly as the scheme says —
// a sidecar with no scheme record predates the mechanism
// and is rejected (see Load).
type Scheme string

const (
	// SchemeWords scores whole words only: the original
	// classifier, kept for comparison.
	SchemeWords Scheme = "words"
	// SchemeWordsNgrams adds each word's character n-grams
	// (length 3–5, "~"-prefixed) as features.
	SchemeWordsNgrams Scheme = "words+ngrams"
	// SchemeWordsBPE adds each word's BPE pieces (the repo's
	// bpe package, merges trained on the intent rows' words)
	// as features.
	SchemeWordsBPE Scheme = "words+bpe"
)

// DefaultScheme is the scheme Train uses, chosen by the
// pieces A/B recorded in the README ("Classifier: pieces
// and kinds"). The honest result: neither piece scheme
// earned the default. Always-on piece counts inflate the
// classifier's confidence — abstentions become confident
// errors — and both schemes regressed shipped fences the
// word classifier passes (held-out intent lines 30/30 raw,
// file Set P 20/20, small-talk 22/22). N-grams won exactly
// one measure (typo lines' raw top label, 10/10) and BPE
// one (held-out-family neighbourhood, 31/36), against
// losses on the collision traps and the fences. So words
// stay the default, and the piece schemes remain available
// through TrainWith for anyone whose own measurement, on
// their own data, says otherwise.
const DefaultScheme = SchemeWords

// bpeMerges caps the merges learned when the BPE scheme
// featurizes: the intent corpora are small, and a few
// hundred merges cover their vocabularies' shared parts.
const bpeMerges = 300

// Options selects non-default training behaviour. The zero
// value trains exactly as Train does.
type Options struct {
	// Scheme is the feature scheme; "" means DefaultScheme.
	Scheme Scheme
}

// Classifier is a trained intent classifier: a label
// softmax, and — when the training rows carried "kind"
// fields — a kind softmax beside it. The exported fields
// are its persisted form (see Save/Load): everything
// Predict and PredictKind need travels in one JSON sidecar
// file.
type Classifier struct {
	Labels    []string    `json:"labels"`    // sorted label names
	Words     []string    `json:"words"`     // feature order: sorted feature strings (words, plus pieces per Scheme)
	Weights   [][]float64 `json:"weights"`   // [label][feature]
	Bias      []float64   `json:"bias"`      // [label]
	Threshold float64     `json:"threshold"` // abstain below this confidence
	Rows      int         `json:"rows"`      // intent rows it was trained on

	Scheme string      `json:"scheme"`           // feature scheme (see Scheme); absent in pre-scheme sidecars
	Merges []bpe.Merge `json:"merges,omitempty"` // BPE merges, when Scheme is words+bpe

	Kinds       map[string]string `json:"kinds,omitempty"`       // label -> kind, for labels whose rows carried one
	KindLabels  []string          `json:"kindLabels,omitempty"`  // sorted kind names
	KindWeights [][]float64       `json:"kindWeights,omitempty"` // [kind][feature]
	KindBias    []float64         `json:"kindBias,omitempty"`    // [kind]

	SubKinds       map[string]string `json:"subkinds,omitempty"`       // label -> sub-kind, for labels whose rows carried one
	SubKindParents map[string]string `json:"subkindParents,omitempty"` // sub-kind -> kind: each sub-kind sits under exactly one kind

	featIdx map[string]int // feature index, rebuilt by Train and Load; not persisted
	bpeTok  *bpe.Tokenizer // rebuilt by Train and Load under the BPE scheme; not persisted
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

// piecesOf returns the subword features of one word under a
// scheme. N-gram features carry a "~" prefix so the feature
// space stays readable (a gram can never be mistaken for a
// word). BPE features are the tokenizer's vocabulary forms
// (first piece bare, continuations "##"-prefixed), minus
// single characters (a lone letter is not a piece in any
// useful sense) and minus the word itself (already a
// feature). Under SchemeWords there are no pieces.
func piecesOf(scheme Scheme, tok *bpe.Tokenizer, word string) []string {
	switch scheme {
	case SchemeWordsNgrams:
		runes := []rune(word)
		var out []string
		for n := 3; n <= 5; n++ {
			for i := 0; i+n <= len(runes); i++ {
				out = append(out, "~"+string(runes[i:i+n]))
			}
		}
		return out
	case SchemeWordsBPE:
		if tok == nil {
			return nil
		}
		var out []string
		for _, f := range tok.EncodeForms(word) {
			if len([]rune(strings.TrimPrefix(f, "##"))) < 2 || f == word {
				continue
			}
			out = append(out, f)
		}
		return out
	}
	return nil
}

// featurizer builds the feature space of a training corpus
// and turns lines into sparse feature counts over it. The
// same featurizer serves training and prediction, so a line
// is always scored on exactly the features the weights were
// trained over.
type featurizer struct {
	scheme Scheme
	tok    *bpe.Tokenizer // BPE scheme only
	merges []bpe.Merge    // BPE scheme only: persistable form of tok
	index  map[string]int
}

// newFeaturizer derives the feature space from the intent
// rows: every distinct word, plus every piece of every
// distinct word under the scheme.
func newFeaturizer(rows []data.Pair, scheme Scheme) *featurizer {
	f := &featurizer{scheme: scheme}
	if scheme == SchemeWordsBPE {
		var stream []string
		for _, p := range rows {
			stream = append(stream, lineWords(p.Input)...)
		}
		f.tok = bpe.Train(stream, bpeMerges)
		f.merges = f.tok.Merges()
	}
	set := map[string]bool{}
	for _, p := range rows {
		for _, w := range lineWords(p.Input) {
			if set[w] {
				continue
			}
			set[w] = true
			for _, piece := range piecesOf(scheme, f.tok, w) {
				set[piece] = true
			}
		}
	}
	features := keys(set)
	f.index = indexOf(features)
	return f
}

// loadFeaturizer rebuilds a featurizer from a sidecar's
// recorded scheme and feature list, for prediction.
func loadFeaturizer(scheme Scheme, merges []bpe.Merge, features []string) *featurizer {
	f := &featurizer{scheme: scheme, index: indexOf(features)}
	if scheme == SchemeWordsBPE {
		f.tok = bpe.FromMerges(merges)
		f.merges = merges
	}
	return f
}

// sparseVec is a feature-count vector in index form: the
// features a line actually has, with their counts. Training
// and prediction both work on sparse vectors — with pieces,
// the feature space is thousands wide and a line touches
// only dozens of them.
type sparseVec struct {
	idx []int
	val []float64
}

// counts turns a line into its sparse feature vector.
// Unknown words are not evidence for any label, but their
// PIECES may be: pieces are looked up in the same index, so
// an unseen word contributes whatever pieces it shares with
// trained words, and nothing else.
func (f *featurizer) counts(line string) sparseVec {
	m := map[int]float64{}
	for _, w := range lineWords(line) {
		if j, ok := f.index[w]; ok {
			m[j]++
		}
		for _, piece := range piecesOf(f.scheme, f.tok, w) {
			if j, ok := f.index[piece]; ok {
				m[j]++
			}
		}
	}
	out := sparseVec{idx: make([]int, 0, len(m)), val: make([]float64, 0, len(m))}
	for j := range m {
		out.idx = append(out.idx, j)
	}
	sort.Ints(out.idx)
	for _, j := range out.idx {
		out.val = append(out.val, m[j])
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

// KindMap derives the label -> kind map of a set of pairs
// and validates it: a label whose rows carry two different
// kinds is a data error (one meaning cannot live in two
// neighbourhoods), reported with the label and both kinds
// named. Labels whose rows carry no kind are absent from
// the map — kinds are optional, per label.
func KindMap(pairs []data.Pair) (map[string]string, error) {
	m := map[string]string{}
	for _, p := range Rows(pairs) {
		if p.Kind == "" {
			continue
		}
		if prev, ok := m[p.Intent]; ok && prev != p.Kind {
			return nil, fmt.Errorf("intent: label %q carries two kinds, %q and %q — a meaning belongs to one kind; fix the data", p.Intent, prev, p.Kind)
		}
		m[p.Intent] = p.Kind
	}
	return m, nil
}

// SubKindMap derives the label -> sub-kind and
// sub-kind -> kind maps of a set of pairs and validates
// them, with KindMap's discipline: a label carrying two
// sub-kinds is a data error, a sub-kind sitting under
// two kinds is a data error, and a sub-kind on a label
// that carries no kind is a data error — a sub-kind is
// a family WITHIN a kind, never a level of its own.
// Labels whose rows carry no sub-kind are absent.
func SubKindMap(pairs []data.Pair, kindMap map[string]string) (subs, parents map[string]string, err error) {
	subs = map[string]string{}
	parents = map[string]string{}
	for _, p := range Rows(pairs) {
		if p.SubKind == "" {
			continue
		}
		if prev, ok := subs[p.Intent]; ok && prev != p.SubKind {
			return nil, nil, fmt.Errorf("intent: label %q carries two sub-kinds, %q and %q — a meaning belongs to one sub-kind; fix the data", p.Intent, prev, p.SubKind)
		}
		subs[p.Intent] = p.SubKind
		kind := kindMap[p.Intent]
		if kind == "" {
			return nil, nil, fmt.Errorf("intent: label %q carries sub-kind %q but no kind — a sub-kind sits under a kind; fix the data", p.Intent, p.SubKind)
		}
		if prev, ok := parents[p.SubKind]; ok && prev != kind {
			return nil, nil, fmt.Errorf("intent: sub-kind %q sits under two kinds, %q and %q — a sub-kind belongs to one kind; fix the data", p.SubKind, prev, kind)
		}
		parents[p.SubKind] = kind
	}
	return subs, parents, nil
}

// Train fits a classifier on every pair that carries an
// intent, with the default options. It errors when there
// are no intent rows or only one label (a classifier needs
// a choice to learn).
func Train(pairs []data.Pair, threshold float64) (*Classifier, error) {
	return TrainWith(pairs, threshold, Options{})
}

// TrainWith is Train with options (see Options). When the
// rows carry kinds, a kind classifier is trained alongside
// the label classifier on the same features; a label with
// two kinds is an error (see KindMap).
func TrainWith(pairs []data.Pair, threshold float64, opts Options) (*Classifier, error) {
	scheme := opts.Scheme
	if scheme == "" {
		scheme = DefaultScheme
	}
	switch scheme {
	case SchemeWords, SchemeWordsNgrams, SchemeWordsBPE:
	default:
		return nil, fmt.Errorf("intent: unknown feature scheme %q", scheme)
	}
	rows := Rows(pairs)
	if len(rows) == 0 {
		return nil, fmt.Errorf("intent: no rows carry an \"intent\" field")
	}
	counts := CountByLabel(rows)
	if len(counts) < 2 {
		return nil, fmt.Errorf("intent: only one label (%v) — a classifier needs at least two meanings to choose between", keys(counts))
	}
	kindMap, err := KindMap(rows)
	if err != nil {
		return nil, err
	}
	subMap, subParents, err := SubKindMap(rows, kindMap)
	if err != nil {
		return nil, err
	}

	feat := newFeaturizer(rows, scheme)
	c := &Classifier{Threshold: threshold, Rows: len(rows), Scheme: string(scheme)}
	c.Labels = keys(counts)
	c.Words = keys(feat.index)
	c.featIdx = feat.index
	c.bpeTok = feat.tok
	c.Merges = feat.merges
	labelIdx := indexOf(c.Labels)

	xs := make([]sparseVec, len(rows))
	ys := make([]int, len(rows))
	for i, p := range rows {
		xs[i] = feat.counts(p.Input)
		ys[i] = labelIdx[p.Intent]
	}
	c.Weights, c.Bias = fit(xs, ys, nil, len(c.Labels), len(c.Words))

	// The kind classifier: every labeled row whose label
	// carries a kind contributes (line -> kind). With fewer
	// than two kinds there is no neighbourhood to learn, and
	// the classifier simply has none.
	//
	// Kind training is class-BALANCED: each row's gradient
	// is weighted so every kind contributes the same total,
	// however many rows it has. The kind classifier's whole
	// job is the weak-evidence case — a meaning too new for
	// a label — and there an unbalanced prior (the biggest
	// kind's share of the rows) is exactly the wrong answer:
	// it would call everything by the commonest kind's name.
	// The label classifier keeps the natural frequencies;
	// telling meanings apart is its job, and there the prior
	// is evidence too.
	kindSet := map[string]bool{}
	for _, k := range kindMap {
		kindSet[k] = true
	}
	if len(subMap) > 0 {
		c.SubKinds = subMap
		c.SubKindParents = subParents
	}
	if len(kindSet) >= 2 {
		c.Kinds = kindMap
		c.KindLabels = keys(kindSet)
		kindIdx := indexOf(c.KindLabels)
		kindRows := map[string]int{}
		for _, p := range rows {
			if kind, ok := kindMap[p.Intent]; ok {
				kindRows[kind]++
			}
		}
		var kxs []sparseVec
		var kys []int
		var kKinds []string
		for i, p := range rows {
			kind, ok := kindMap[p.Intent]
			if !ok {
				continue
			}
			kxs = append(kxs, xs[i])
			kys = append(kys, kindIdx[kind])
			kKinds = append(kKinds, kind)
		}
		// The balancing weight: total / (kinds * rows of this
		// kind), so each kind's weights sum to the same
		// total/kinds share.
		kws := make([]float64, len(kxs))
		for i, kind := range kKinds {
			kws[i] = float64(len(kxs)) / (float64(len(kindSet)) * float64(kindRows[kind]))
		}
		c.KindWeights, c.KindBias = fit(kxs, kys, kws, len(c.KindLabels), len(c.Words))
	}
	return c, nil
}

// fit trains a softmax model by seeded SGD on the
// cross-entropy, exactly as the original dense trainer did
// — same epochs, rate, and L2 — but over sparse vectors:
// a row's update touches only the features it has, and the
// L2 decay of the untouched weights is applied lazily, per
// label, through a decay factor (each label's stored
// weights are its true weights divided by that factor;
// skipped labels — a delta of exactly zero — keep their
// factor, as the dense loop skipped their update). The
// result matches the dense trainer's up to float rounding,
// at a cost proportional to what a line contains rather
// than to the whole feature space — which is what makes
// piece features affordable.
//
// ws holds per-example gradient weights (nil: all ones).
// The kind model trains with balancing weights (see
// TrainWith); the label model passes nil.
func fit(xs []sparseVec, ys []int, ws []float64, nLabels, nFeat int) ([][]float64, []float64) {
	weights := make([][]float64, nLabels)
	for i := range weights {
		weights[i] = make([]float64, nFeat)
	}
	bias := make([]float64, nLabels)
	decay := make([]float64, nLabels)
	for i := range decay {
		decay[i] = 1
	}
	const stepDecay = 1 - trainLR*trainL2

	rng := rand.New(rand.NewSource(trainSeed))
	logits := make([]float64, nLabels)
	for epoch := 0; epoch < trainEpochs; epoch++ {
		order := rng.Perm(len(xs))
		for _, idx := range order {
			x, y := xs[idx], ys[idx]
			for l := 0; l < nLabels; l++ {
				s := bias[l]
				for k, j := range x.idx {
					s += weights[l][j] * decay[l] * x.val[k]
				}
				logits[l] = s
			}
			probs := softmax(logits)
			w := 1.0
			if ws != nil {
				w = ws[idx]
			}
			for l := 0; l < nLabels; l++ {
				delta := probs[l]
				if l == y {
					delta--
				}
				delta *= w
				if delta == 0 {
					continue
				}
				decay[l] *= stepDecay
				inv := 1.0 / decay[l]
				for k, j := range x.idx {
					weights[l][j] -= trainLR * delta * x.val[k] * inv
				}
				bias[l] -= trainLR * delta
			}
		}
	}
	// Materialize the true weights: undo the lazy decay.
	for l := range weights {
		for j := range weights[l] {
			weights[l][j] *= decay[l]
		}
	}
	return weights, bias
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

// best returns the winning label of a softmax model over a
// line, with the abstention rule shared by labels and
// kinds: no feature evidence at all, or a best probability
// below the threshold, returns no label. The confidence is
// still reported (the best label's probability, or 0 when
// there was no evidence).
func best(labels []string, weights [][]float64, bias []float64, x sparseVec, threshold float64) (string, float64) {
	evidence := 0.0
	for _, v := range x.val {
		evidence += v
	}
	if evidence == 0 {
		return "", 0
	}
	logits := make([]float64, len(labels))
	for l := range labels {
		s := bias[l]
		for k, j := range x.idx {
			s += weights[l][j] * x.val[k]
		}
		logits[l] = s
	}
	probs := softmax(logits)
	b := 0
	for i := range probs {
		if probs[i] > probs[b] {
			b = i
		}
	}
	if probs[b] < threshold {
		return "", probs[b]
	}
	return labels[b], probs[b]
}

// Predict returns the most likely meaning of a line and the
// classifier's confidence in it. Below the threshold — or
// when the line shares no features with anything the
// classifier was trained on — the label is empty: the
// classifier abstains, and the caller should do nothing.
// The confidence is still reported (it is the best label's
// probability, or 0 when there was no evidence at all).
func (c *Classifier) Predict(line string) (string, float64) {
	return best(c.Labels, c.Weights, c.Bias, c.fcCounts(line), c.Threshold)
}

// LabelScore is one label's probability for a line, as
// RankedLabels reports it.
type LabelScore struct {
	Label string
	Prob  float64
}

// RankedLabels returns every label's probability for a
// line, best first — the full read Predict takes only the
// winner of. Callers that must reason about the RUNNER-UP
// (a near-tie between two meanings, the narrow read a
// fallback walk starts from) use this instead of guessing
// from the winner's confidence alone. The order is total
// and deterministic: probability descending, and exact
// probability ties broken by label name (Labels are stored
// sorted, so the tie-break is stable across processes).
// A line with no feature evidence at all returns nil —
// the same no-evidence rule best applies — rather than a
// ranking of priors.
func (c *Classifier) RankedLabels(line string) []LabelScore {
	x := c.fcCounts(line)
	evidence := 0.0
	for _, v := range x.val {
		evidence += v
	}
	if evidence == 0 {
		return nil
	}
	logits := make([]float64, len(c.Labels))
	for l := range c.Labels {
		s := c.Bias[l]
		for k, j := range x.idx {
			s += c.Weights[l][j] * x.val[k]
		}
		logits[l] = s
	}
	probs := softmax(logits)
	out := make([]LabelScore, len(c.Labels))
	for i, label := range c.Labels {
		out[i] = LabelScore{Label: label, Prob: probs[i]}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Prob != out[j].Prob {
			return out[i].Prob > out[j].Prob
		}
		return out[i].Label < out[j].Label
	})
	return out
}

// fcCounts is the classifier's featurizer applied to a line.
func (c *Classifier) fcCounts(line string) sparseVec {
	if c.featIdx == nil { // a Classifier built by hand, not Train/Load
		c.featIdx = indexOf(c.Words)
		if Scheme(c.Scheme) == SchemeWordsBPE && c.bpeTok == nil {
			c.bpeTok = bpe.FromMerges(c.Merges)
		}
	}
	f := &featurizer{scheme: Scheme(c.Scheme), tok: c.bpeTok, index: c.featIdx}
	return f.counts(line)
}

// HasKinds reports whether the classifier learned kinds
// (its training rows carried "kind" fields).
func (c *Classifier) HasKinds() bool {
	return len(c.KindLabels) > 0
}

// KindOf returns the kind a label belongs to, or "" when
// the label carries none (or is unknown).
func (c *Classifier) KindOf(label string) string {
	return c.Kinds[label]
}

// LabelKinds returns a copy of the label -> kind map the
// classifier was trained with (empty when it has no kinds).
func (c *Classifier) LabelKinds() map[string]string {
	out := make(map[string]string, len(c.Kinds))
	for k, v := range c.Kinds {
		out[k] = v
	}
	return out
}

// SubKindOf returns the sub-kind a label belongs to —
// the middle level of the kind tree — or "" when the
// label carries none (or is unknown).
func (c *Classifier) SubKindOf(label string) string {
	return c.SubKinds[label]
}

// LabelSubKinds returns a copy of the label -> sub-kind
// map the classifier was trained with (empty when its
// rows carried no sub-kinds).
func (c *Classifier) LabelSubKinds() map[string]string {
	out := make(map[string]string, len(c.SubKinds))
	for k, v := range c.SubKinds {
		out[k] = v
	}
	return out
}

// SubKindParent returns the kind a sub-kind sits under,
// or "" when the sub-kind is unknown.
func (c *Classifier) SubKindParent(sub string) string {
	return c.SubKindParents[sub]
}

// PredictKind returns the most likely KIND of a line — the
// neighbourhood its meaning lives in — with the same
// abstention rule as Predict. It is the answer available
// for a meaning too new to have a label: "" when the
// classifier has no kinds, has no evidence, or is not
// confident enough.
func (c *Classifier) PredictKind(line string) (string, float64) {
	if !c.HasKinds() {
		return "", 0
	}
	return best(c.KindLabels, c.KindWeights, c.KindBias, c.fcCounts(line), c.Threshold)
}

// Describe renders one line's prediction the way cmd/ask
// prints it: "lights_off (0.87)", or the abstention. When
// the classifier has kinds, the kind prediction is appended
// — "lights_off (0.87) · kind: find (0.91)" — and it is
// shown even when the label abstains, because the
// neighbourhood is exactly what an unseen meaning can
// still offer. Classifiers without kinds describe exactly
// as they always did.
func (c *Classifier) Describe(line string) string {
	label, conf := c.Predict(line)
	s := "(none confident enough)"
	if label != "" {
		s = fmt.Sprintf("%s (%.2f)", label, conf)
	}
	if c.HasKinds() {
		kind, kconf := c.PredictKind(line)
		ks := "(none confident enough)"
		if kind != "" {
			ks = fmt.Sprintf("%s (%.2f)", kind, kconf)
		}
		s += " · kind: " + ks
	}
	return s
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
// silently half-working classifier. A file with no feature
// scheme record was written before schemes existed: its
// weights were trained over whole words only, and scoring
// it with today's featurizer would quietly change what its
// predictions mean — so it is rejected loudly, with the
// remedy named, instead.
func Load(path string) (*Classifier, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Classifier
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("intent: bad sidecar %s: %w", path, err)
	}
	if c.Scheme == "" {
		return nil, fmt.Errorf("intent: sidecar %s records no feature scheme — it was written by an older version, and its predictions would not mean what they did; retrain the brain to upgrade its intent classifier", path)
	}
	switch Scheme(c.Scheme) {
	case SchemeWords, SchemeWordsNgrams:
	case SchemeWordsBPE:
		if len(c.Merges) == 0 {
			return nil, fmt.Errorf("intent: sidecar %s uses scheme %q but records no BPE merges", path, c.Scheme)
		}
	default:
		return nil, fmt.Errorf("intent: sidecar %s records unknown feature scheme %q", path, c.Scheme)
	}
	if len(c.Labels) == 0 || len(c.Weights) != len(c.Labels) || len(c.Bias) != len(c.Labels) {
		return nil, fmt.Errorf("intent: sidecar %s has inconsistent shapes (labels=%d weights=%d bias=%d)", path, len(c.Labels), len(c.Weights), len(c.Bias))
	}
	for i, row := range c.Weights {
		if len(row) != len(c.Words) {
			return nil, fmt.Errorf("intent: sidecar %s weight row %d has %d features, want %d", path, i, len(row), len(c.Words))
		}
	}
	if c.HasKinds() {
		if len(c.KindWeights) != len(c.KindLabels) || len(c.KindBias) != len(c.KindLabels) {
			return nil, fmt.Errorf("intent: sidecar %s has inconsistent kind shapes (kinds=%d weights=%d bias=%d)", path, len(c.KindLabels), len(c.KindWeights), len(c.KindBias))
		}
		for i, row := range c.KindWeights {
			if len(row) != len(c.Words) {
				return nil, fmt.Errorf("intent: sidecar %s kind weight row %d has %d features, want %d", path, i, len(row), len(c.Words))
			}
		}
		for label := range c.Kinds {
			found := false
			for _, l := range c.Labels {
				if l == label {
					found = true
					break
				}
			}
			if !found {
				return nil, fmt.Errorf("intent: sidecar %s maps unknown label %q to a kind", path, label)
			}
		}
	}
	for label, sub := range c.SubKinds {
		found := false
		for _, l := range c.Labels {
			if l == label {
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("intent: sidecar %s maps unknown label %q to a sub-kind", path, label)
		}
		parent, ok := c.SubKindParents[sub]
		if !ok {
			return nil, fmt.Errorf("intent: sidecar %s records no parent kind for sub-kind %q", path, sub)
		}
		if kind := c.Kinds[label]; kind != "" && kind != parent {
			return nil, fmt.Errorf("intent: sidecar %s puts label %q in kind %q but its sub-kind %q under kind %q", path, label, kind, sub, parent)
		}
	}
	if c.Threshold == 0 {
		c.Threshold = DefaultThreshold
	}
	c.featIdx = indexOf(c.Words)
	if Scheme(c.Scheme) == SchemeWordsBPE {
		c.bpeTok = bpe.FromMerges(c.Merges)
	}
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
