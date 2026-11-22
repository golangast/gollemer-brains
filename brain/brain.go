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
	"regexp"
	"strings"

	"github.com/golangast/gollemer-brains/data"
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
	tok   *tokenizer.Tokenizer
}

// LoadBrain reads a .gob written by Train (or by the model's
// Save). The vocabulary travels inside the gob, so the
// tokenizer is rebuilt from the model itself — a .gob is all
// you need to run a brain.
func LoadBrain(path string) (*Brain, error) {
	// The tokenizer is not used during gob decode; it is only
	// stored on the model. Decode with a placeholder, then
	// install the real one built from the loaded vocabulary.
	empty, err := tokenizer.NewTokenizer(vocab.NewVocabulary())
	if err != nil {
		return nil, err
	}
	m, err := seq2seq.LoadLayers(path, empty)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.NewTokenizer(m.Base.OutputVocab)
	if err != nil {
		return nil, err
	}
	m.Base.Tokenizer = tok
	return &Brain{Model: m, tok: tok}, nil
}

// encodeLine turns a raw line into the ids + surfaces the
// prediction path consumes, exactly as training encoded them.
func (b *Brain) encodeLine(line string) ([]int, []string) {
	v := b.Model.Base.OutputVocab
	words := TokenizeLower(seq2seq.NormalizeQuery(line))
	ids := make([]int, len(words))
	for i, w := range words {
		ids[i] = v.GetTokenID(w)
	}
	return ids, append([]string{}, words...)
}

// Answer generates the greedy reply for one line.
func (b *Brain) Answer(line string) (string, error) {
	ids, surfs := b.encodeLine(line)
	pred, err := b.Model.PredictLayers(ids, surfs, seq2seq.LayersPredictOptions{MaxLen: 40})
	if err != nil {
		return "", err
	}
	return TidyDecode(pred.Text), nil
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
		if s := TidyDecode(pred.Text); ReplyCarries(line, s) {
			return s, nil
		}
	}
	return greedy, nil
}
