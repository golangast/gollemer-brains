// Package brain is the standalone glue around the seq2seq
// model: vocabulary building, encoding, generation (greedy and
// checked sampling), and loading a trained brain from its .gob
// file. The trainer lives in train.go.
//
// Everything here is carved from Gollemer's chat package
// (internal/ai/training/chat in github.com/golangast/gollemer),
// simplified where the chat product's machinery (sessions,
// fact seats, topic banks) did not belong in a reusable brain.
package brain

import (
	"fmt"
	"hash/fnv"
	"math/rand"
	"os"
	"regexp"
	"strings"

	"github.com/golangast/gollemer-brains/bpe"
	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/intent"
	"github.com/golangast/gollemer-brains/seq2seq"
	"github.com/golangast/gollemer-brains/tokenizer"
	"github.com/golangast/gollemer-brains/vocab"
)

// TokenizeLower splits text exactly the way the tokenizer does
// (whitespace + {}[],:\"?!. as single-char tokens), lowercased
// so the model is case-insensitive. Copied from the source
// trainer; it must stay in sync with the tokenizer package.
func TokenizeLower(text string) []string {
	var toks []string
	runes := []rune(strings.ToLower(text))
	i := 0
	for i < len(runes) {
		r := runes[i]
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			i++
			continue
		}
		if strings.ContainsRune(`{}[],:\"?!.`, r) {
			toks = append(toks, string(r))
			i++
			continue
		}
		start := i
		for i < len(runes) && runes[i] != ' ' && runes[i] != '\t' && runes[i] != '\n' && runes[i] != '\r' && !strings.ContainsRune(`{}[],:\"`, runes[i]) {
			i++
		}
		if tok := string(runes[start:i]); tok != "" {
			toks = append(toks, tok)
		}
	}
	return toks
}

// BuildVocab builds the vocabulary from pairs (the training
// split only — see data.Split) and the tokenizer over it.
func BuildVocab(pairs []data.Pair) (*vocab.Vocabulary, *tokenizer.Tokenizer, error) {
	v := vocab.NewVocabulary()
	for _, p := range pairs {
		for _, t := range TokenizeLower(p.Input) {
			v.AddToken(t)
		}
		for _, t := range TokenizeLower(p.Output) {
			v.AddToken(t)
		}
	}
	tok, err := tokenizer.NewTokenizer(v)
	if err != nil {
		return nil, nil, err
	}
	return v, tok, nil
}

// VocabFingerprint hashes the vocabulary so checkpoint resume
// can refuse a checkpoint trained on different data — a vocab
// index shift would silently corrupt the resumed weights.
// Copied from the source trainer.
func VocabFingerprint(v *vocab.Vocabulary) uint64 {
	h := fnv.New64a()
	fmt.Fprintf(h, "size=%d;", v.Size())
	for i := 0; i < v.Size(); i++ {
		fmt.Fprintf(h, "%d=%s;", i, v.GetWord(i))
	}
	return h.Sum64()
}

// EncodeInput tokenizes a line for the encoder. Inputs are
// normalized the same way at training and inference
// (seq2seq.NormalizeQuery): punctuation variants of one
// phrasing share one encoding.
func EncodeInput(tok *tokenizer.Tokenizer, q string) []int {
	ids, err := tok.Encode(seq2seq.NormalizeQuery(q))
	if err != nil {
		return nil
	}
	return ids
}

// EncodeTarget tokenizes a reply for the decoder: [BOS, ids..., EOS].
func EncodeTarget(tok *tokenizer.Tokenizer, v *vocab.Vocabulary, a string) []int {
	ids, err := tok.Encode(strings.ToLower(a))
	if err != nil {
		return nil
	}
	out := make([]int, 0, len(ids)+2)
	out = append(out, v.BosID)
	out = append(out, ids...)
	out = append(out, v.EosID)
	return out
}

// TidyDecode reattaches punctuation to the word before it, the
// way the source's probe output did.
func TidyDecode(s string) string {
	for _, p := range []string{"?", "!", ".", ",", ":", ";", "'"} {
		s = strings.ReplaceAll(s, " "+p, p)
	}
	s = strings.ReplaceAll(s, "  ", " ")
	return strings.TrimSpace(s)
}

// Brain is a trained layers model ready to answer lines.
type Brain struct {
	Model *seq2seq.LayersModel
	// Intents is the intent classifier trained alongside the
	// model (nil when the training data carried no "intent"
	// fields). It predicts the MEANING of a line — a label
	// the user defined — separately from the reply.
	Intents *intent.Classifier
	// FixTypos switches on the typo normalizer (see
	// normalize.go): before a line is encoded, an
	// out-of-vocabulary word with exactly one close
	// vocabulary neighbour is replaced by that neighbour.
	// Off by default; cmd/ask turns it on.
	FixTypos    bool
	tok         *tokenizer.Tokenizer
	typoTargets []string
	// bpeTok is the brain's BPE tokenizer, restored from
	// the brain file's header by LoadBrain; nil for a
	// word-tokenizer brain. When it is set, encodeLine
	// splits words into pieces and replies are glued back
	// together in renderText. The typo normalizer does not
	// apply: it repairs a word vocabulary's wall, and a
	// BPE brain has no wall — an unknown word decomposes
	// into pieces instead of going unknown.
	bpeTok *bpe.Tokenizer
}

// intentsPathFor returns the intent classifier's sidecar
// path for a brain file: <model>.intents.json. The
// classifier travels as a sidecar rather than inside the
// .gob so brains trained before intents existed load
// unchanged, and a brain without intents simply has no
// sidecar.
func intentsPathFor(modelPath string) string {
	return modelPath + ".intents.json"
}

// LoadBrain reads a .gob written by Train (or by the model's
// Save). The vocabulary travels inside the gob, so the
// tokenizer is rebuilt from the model itself — a .gob is all
// you need to run a brain.
func LoadBrain(path string) (*Brain, error) {
	// A BPE brain travels wrapped (see brainfile.go): the
	// header names the tokenizer and carries the merges.
	// unwrap validates the header loudly — a brain decoded
	// with the wrong tokenizer would garble silently.
	modelPath, kind, merges, cleanup, err := unwrapBrainFile(path)
	if err != nil {
		return nil, err
	}
	defer cleanup()
	// The tokenizer is not used during gob decode; it is only
	// stored on the model. Decode with a placeholder, then
	// install the real one built from the loaded vocabulary.
	empty, err := tokenizer.NewTokenizer(vocab.NewVocabulary())
	if err != nil {
		return nil, err
	}
	m, err := seq2seq.LoadLayers(modelPath, empty)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.NewTokenizer(m.Base.OutputVocab)
	if err != nil {
		return nil, err
	}
	m.Base.Tokenizer = tok
	br := &Brain{Model: m, tok: tok}
	if kind == "bpe" {
		br.bpeTok = bpe.FromMerges(merges)
	}
	// The intent sidecar is optional — most brains have
	// none — but when it is present it must load: a corrupt
	// sidecar is an error, never a silently dropped
	// classifier.
	if _, err := os.Stat(intentsPathFor(path)); err == nil {
		clf, err := intent.Load(intentsPathFor(path))
		if err != nil {
			return nil, err
		}
		br.Intents = clf
	}
	return br, nil
}

// encodeLine turns a raw line into the ids + surfaces the
// prediction path consumes, exactly as training encoded them
// — plus, when FixTypos is on, the typo normalizer's
// corrections (normalize.go).
func (b *Brain) encodeLine(line string) ([]int, []string) {
	v := b.Model.Base.OutputVocab
	if b.bpeTok != nil {
		return bpeEncodeText(b.bpeTok, v, TokenizeLower(seq2seq.NormalizeQuery(line)))
	}
	words := b.fixTokens(TokenizeLower(seq2seq.NormalizeQuery(line)))
	ids := make([]int, len(words))
	for i, w := range words {
		ids[i] = v.GetTokenID(w)
	}
	return ids, append([]string{}, words...)
}

// renderText turns a raw prediction (vocabulary forms
// joined by spaces) into the reply text: for a BPE brain
// the "##" continuation pieces are glued back into words
// first, then the usual punctuation tidying applies.
func (b *Brain) renderText(raw string) string {
	if b.bpeTok != nil {
		raw = bpe.Join(strings.Fields(raw))
	}
	return TidyDecode(raw)
}

// Answer generates the greedy reply for one line.
func (b *Brain) Answer(line string) (string, error) {
	ids, surfs := b.encodeLine(line)
	pred, err := b.Model.PredictLayers(ids, surfs, seq2seq.LayersPredictOptions{MaxLen: 40})
	if err != nil {
		return "", err
	}
	return b.renderText(pred.Text), nil
}

const (
	// The decode policy measured in the source project:
	// temperature 0.8, at most 8 sampled draws to rescue a
	// greedy reply that dropped the line's subject.
	sampleTemp  = 0.8
	sampleDraws = 8
)

var wordReCache = map[string]*regexp.Regexp{}

func wordRe(word string) *regexp.Regexp {
	if re, ok := wordReCache[word]; ok {
		return re
	}
	re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(word) + `\b`)
	wordReCache[word] = re
	return re
}

// ReplyCarries is the checked-sampling test, simplified from
// the source: there, the line's topic came from a bank of
// phrases the model was trained on, falling back to the
// line's content words. The standalone keeps the fallback as
// the whole rule — a reply carries the line when it contains
// at least one of the line's content words (word boundaries,
// case-insensitive). An empty reply never carries; a line
// with no content words is carried by nothing, so its greedy
// reply always stands.
func ReplyCarries(line, reply string) bool {
	if strings.TrimSpace(reply) == "" {
		return false
	}
	for _, w := range seq2seq.ContentWordsOf(lineWords(line)) {
		if wordRe(w).MatchString(reply) {
			return true
		}
	}
	return false
}

// lineWords splits a line the way the source's checker read
// its inputs: lowercased fields with edge punctuation trimmed.
func lineWords(line string) []string {
	fields := strings.Fields(strings.ToLower(line))
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if f = strings.Trim(f, ".,!?;:\"'()[]"); f != "" {
			out = append(out, f)
		}
	}
	return out
}

// AnswerChecked generates a reply with checked sampling: the
// greedy reply stands when it carries the line; otherwise up
// to sampleDraws replies are drawn at temperature sampleTemp
// and the first carrying one wins. Sampling only rescues
// replies that dropped the subject — it never replaces a
// greedy reply that was already on-topic.
func (b *Brain) AnswerChecked(line string, rng *rand.Rand) (string, error) {
	greedy, err := b.Answer(line)
	if err != nil {
		return "", err
	}
	if ReplyCarries(line, greedy) {
		return greedy, nil
	}
	ids, surfs := b.encodeLine(line)
	for i := 0; i < sampleDraws; i++ {
		pred, err := b.Model.PredictLayersSampled(ids, surfs, seq2seq.LayersPredictOptions{MaxLen: 40}, sampleTemp, rng)
		if err != nil {
			return greedy, nil
		}
		if s := b.renderText(pred.Text); ReplyCarries(line, s) {
			return s, nil
		}
	}
	return greedy, nil
}
