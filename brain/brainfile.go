package brain

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer-brains/bpe"
	"github.com/golangast/gollemer-brains/seq2seq"
)

// Brain file format. A word-tokenizer brain is exactly what
// it has always been: the bare model gob, no wrapper, so
// every brain written before the BPE tokenizer existed
// loads unchanged. A BPE brain needs its merge list to
// travel with the model — without it the piece vocabulary
// is meaningless — so it is wrapped:
//
//	line 1: the magic string below
//	line 2: a JSON header {"kind":"bpe","merges":[...]}
//	rest:   the model gob, byte for byte
//
// The wrapper is detected by the magic line, never guessed
// from content, and an unknown kind is a loud error: a
// brain loaded with the wrong tokenizer would not fail, it
// would silently garble, which is worse.
const brainFileMagic = "gollemer-brains brain file v1"

type brainFileHeader struct {
	Kind   string      `json:"kind"`
	Merges []bpe.Merge `json:"merges,omitempty"`
}

// saveBrainFile writes model to path. For the word
// tokenizer it is model.Save, unchanged. For BPE it writes
// the wrapped format above (via a temp file + rename, like
// the checkpoints).
func saveBrainFile(path string, model *seq2seq.LayersModel, kind string, merges []bpe.Merge) error {
	if kind != "bpe" {
		return model.Save(path)
	}
	if len(merges) == 0 {
		return fmt.Errorf("refusing to save a bpe brain with no merges")
	}
	tmpModel := path + ".model.tmp"
	if err := model.Save(tmpModel); err != nil {
		return err
	}
	defer os.Remove(tmpModel)
	gobBytes, err := os.ReadFile(tmpModel)
	if err != nil {
		return err
	}
	hdr, err := json.Marshal(brainFileHeader{Kind: "bpe", Merges: merges})
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(f, "%s\n%s\n", brainFileMagic, hdr); err != nil {
		f.Close()
		return err
	}
	if _, err := f.Write(gobBytes); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// unwrapBrainFile inspects a brain file. A legacy file (no
// magic line) is returned as-is with kind "word". A wrapped
// file has its inner gob written to a temp file next to it
// — LoadLayers reads paths — and the caller must call
// cleanup when done loading. A wrapped file whose header
// is malformed, names an unknown kind, or claims bpe with
// no merges is an error, never a silent fallback.
func unwrapBrainFile(path string) (modelPath string, kind string, merges []bpe.Merge, cleanup func(), err error) {
	cleanup = func() {}
	f, err := os.Open(path)
	if err != nil {
		return "", "", nil, cleanup, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	first, err := r.ReadString('\n')
	if err != nil || strings.TrimRight(first, "\n") != brainFileMagic {
		return path, "word", nil, cleanup, nil
	}
	hdrLine, err := r.ReadString('\n')
	if err != nil {
		return "", "", nil, cleanup, fmt.Errorf("%s: wrapped brain file ends inside its header", path)
	}
	var hdr brainFileHeader
	if err := json.Unmarshal([]byte(strings.TrimSpace(hdrLine)), &hdr); err != nil {
		return "", "", nil, cleanup, fmt.Errorf("%s: malformed brain file header: %w", path, err)
	}
	switch hdr.Kind {
	case "bpe":
		if len(hdr.Merges) == 0 {
			return "", "", nil, cleanup, fmt.Errorf("%s: brain file says tokenizer \"bpe\" but carries no merges", path)
		}
	case "word":
		// A wrapped word brain is legal (the header says
		// what the gob already implies); nothing extra
		// is needed to load it.
	default:
		return "", "", nil, cleanup, fmt.Errorf("%s: unknown tokenizer kind %q in brain file header", path, hdr.Kind)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".brain-unwrap-*")
	if err != nil {
		return "", "", nil, cleanup, err
	}
	if _, err := r.WriteTo(tmp); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return "", "", nil, cleanup, err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return "", "", nil, cleanup, err
	}
	name := tmp.Name()
	return name, hdr.Kind, hdr.Merges, func() { os.Remove(name) }, nil
}
