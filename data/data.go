// Package data loads training pairs for the brains.
//
// The format is the whole contract: one JSON object per line
// (JSONL), with "input" and "output" string fields:
//
//	{"input": "hello", "output": "Hi! How are you?"}
//
// A row may also carry an "intent" field — a meaning label
// the user defines (see the intent package):
//
//	{"input": "turn off the lights", "intent": "lights_off"}
//
// An intent row may further carry a "kind" field: the broad
// family its meaning belongs to ("make", "schedule", "talk",
// ...). Kinds give the intent classifier a neighbourhood to
// fall back on for meanings it has never seen. A kind only
// annotates an intent — it never replaces one, and a row
// still needs an output or an intent to teach anything.
//
// An intent row does not need an "output": a row with an
// input and an intent but no output is a LABEL-ONLY row, a
// pure command example. The seq2seq trainer skips label-only
// rows (there is no reply to learn); the intent classifier
// trains on every row that carries an intent, with or
// without an output. A row with neither an output nor an
// intent teaches nothing and is rejected as malformed.
//
// Extra fields (for example "domain", which Gollemer's own
// dataset carries) are ignored, so a Gollemer chat_pairs.jsonl
// file drops in unchanged.
package data

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
)

// Pair is one training example: a line the brain may be given,
// and the reply it should learn to produce. Intent is an
// optional meaning label (the intent package's classifier
// trains on it); a Pair with an Intent but no Output is a
// label-only command example. Kind is the optional broad
// family of the Intent ("make", "talk", ...): the classifier
// learns it alongside the label, and the seq2seq trainer
// ignores it exactly as it ignores Intent.
type Pair struct {
	Input  string `json:"input"`
	Output string `json:"output"`
	Intent string `json:"intent,omitempty"`
	Kind   string `json:"kind,omitempty"`
}

// LoadPairs reads a JSONL pairs file. Blank lines are skipped.
// A malformed line or an empty input is an error that names
// the line number — a data file you cannot debug is a data
// file you will not fix. An empty output is an error too,
// unless the row carries an intent: then it is a label-only
// command example and loads fine.
func LoadPairs(path string) ([]Pair, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var pairs []Pair
	sc := bufio.NewScanner(f)
	// Pair lines are short, but a generous buffer keeps a long
	// pasted line from failing the load with a scanner error
	// that has nothing to do with the data's shape.
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var p Pair
		if err := json.Unmarshal([]byte(line), &p); err != nil {
			return nil, fmt.Errorf("%s line %d: invalid JSON: %w", path, lineNo, err)
		}
		if strings.TrimSpace(p.Input) == "" {
			return nil, fmt.Errorf("%s line %d: empty \"input\"", path, lineNo)
		}
		p.Intent = strings.TrimSpace(p.Intent)
		p.Kind = strings.TrimSpace(p.Kind)
		if strings.TrimSpace(p.Output) == "" && p.Intent == "" {
			return nil, fmt.Errorf("%s line %d: empty \"output\" (a row needs an output, or an \"intent\" to be a label-only command example)", path, lineNo)
		}
		pairs = append(pairs, p)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(pairs) == 0 {
		return nil, fmt.Errorf("%s: no pairs found", path)
	}
	return pairs, nil
}

// Split divides pairs into a training set and a held-out set
// with the source trainer's deterministic 80/20 split (fixed
// seed, shuffled once): every run of the same file splits the
// same way, so results are reproducible and the held-out pairs
// are a stable yardstick. The vocabulary is built from the
// training split only — words that appear only in held-out
// pairs stay unknown, which is exactly what the held-out set
// is for.
func Split(pairs []Pair) (train, held []Pair) {
	rng := rand.New(rand.NewSource(20260926))
	order := rng.Perm(len(pairs))
	nTrain := int(0.8 * float64(len(pairs)))
	train = make([]Pair, 0, nTrain)
	held = make([]Pair, 0, len(pairs)-nTrain)
	for i, idx := range order {
		if i < nTrain {
			train = append(train, pairs[idx])
		} else {
			held = append(held, pairs[idx])
		}
	}
	return train, held
}
