// Package data loads training pairs for the brains.
//
// The format is the whole contract: one JSON object per line
// (JSONL), with "input" and "output" string fields:
//
//	{"input": "hello", "output": "Hi! How are you?"}
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
// and the reply it should learn to produce.
type Pair struct {
	Input  string `json:"input"`
	Output string `json:"output"`
}

// LoadPairs reads a JSONL pairs file. Blank lines are skipped.
// A malformed line, an empty input, or an empty output is an
// error that names the line number — a data file you cannot
// debug is a data file you will not fix.
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
		if strings.TrimSpace(p.Output) == "" {
			return nil, fmt.Errorf("%s line %d: empty \"output\"", path, lineNo)
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
