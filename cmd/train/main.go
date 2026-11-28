// Command train trains a brain on a JSONL pairs file.
//
//	go run ./cmd/train -data examples/pairs.jsonl -out brain.gob -epochs 200 -layers copy
//
// Add -augment 0.3 to also train on lightly mangled variants
// of the inputs (a dropped filler word, a typo, swapped
// neighbours) whose replies stay correct — it teaches the
// brain that meaning survives sloppy typing. Off by default.
// -synonyms 0.3 does the same for word CHOICE: one word of
// an input is swapped for a known synonym (like -> enjoy),
// same correct reply.
//
// If the data carries "intent" fields, an intent classifier
// is trained on those rows too (label-only rows — an intent
// with no output — exist for exactly this) and written next
// to the brain as <out>.intents.json; the run's output lists
// the labels it found and the rows per label. If the intent
// rows also carry "kind" fields (the broad family of each
// meaning), a kind classifier is trained alongside and the
// output lists the kinds with their labels.
//
// The output .gob is written once, at the end, from the best
// weights seen. While training runs, checkpoints live next to
// it (<out>.ckpt / <out>.latest with JSON sidecars); rerun
// with -resume to continue a killed run instead of starting
// over. See the README for the data format and the checkpoint
// discipline.
//
// Training a whole domain folder is one command:
//
//	go run ./cmd/train -dir path/to/domain
//
// -dir names a folder holding that domain's own pairs.jsonl;
// the brain is written into the same folder as brain.gob
// (with its .intents.json sidecar), using the standard
// recipe — 60 epochs, the copy layer, light augment and
// synonym passes, checkpoints every 10 epochs — the recipe
// this repo's own brains are trained with. No path in the
// repo is assumed: the folder can live anywhere, and -data,
// -out, and the recipe flags still override the -dir
// defaults individually when given.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer-brains/brain"
	"github.com/golangast/gollemer-brains/data"
)

// The standard recipe: the flag values -dir trains with
// unless a flag is given explicitly. They are the values
// the repo's shipped brains are trained with (the chat
// recipe), so a domain trained with one command is
// trained the way the repo trains its own.
const (
	stdEpochs    = 60
	stdLayers    = "copy"
	stdAugment   = 0.3
	stdSynonyms  = 0.3
	stdCkptEvery = 10
)

func main() {
	dirFlag := flag.String("dir", "", "domain folder to train in one command: reads <dir>/pairs.jsonl, writes <dir>/brain.gob, standard recipe (see above)")
	dataFlag := flag.String("data", "", "path to the JSONL pairs file (required without -dir)")
	outFlag := flag.String("out", "brain.gob", "where to write the trained brain")
	epochsFlag := flag.Int("epochs", 200, "total epochs to reach")
	layersFlag := flag.String("layers", "", "comma-separated layers to switch on (supported: char, copy)")
	resumeFlag := flag.Bool("resume", false, "resume from the checkpoints next to -out instead of starting at epoch 1")
	augmentFlag := flag.Float64("augment", 0, "probability that a pair's input is mutated in each epoch (0 = off, 0.3 = light)")
	synonymsFlag := flag.Float64("synonyms", 0, "probability that a pair's input swaps one word for a known synonym in each epoch (0 = off, 0.3 = light)")
	tokenizerFlag := flag.String("tokenizer", "word", "text tokenizer: word (whole words, the original) or bpe (byte-pair pieces, no vocabulary wall)")
	bpeMergesFlag := flag.Int("bpe-merges", 700, "maximum merges the BPE tokenizer learns (only with -tokenizer bpe)")
	ckptEveryFlag := flag.Int("ckpt-every", 1, "epochs between <out>.latest checkpoint writes (1 = every epoch; higher trades a few replayed epochs on resume for less checkpoint I/O)")
	flag.Parse()

	// -dir resolves the one-command defaults: the folder's
	// own pairs file in, the folder's own brain out, and
	// the standard recipe — each only where the matching
	// flag was not given explicitly.
	if *dirFlag != "" {
		given := map[string]bool{}
		flag.Visit(func(f *flag.Flag) { given[f.Name] = true })
		if !given["data"] {
			*dataFlag = filepath.Join(*dirFlag, "pairs.jsonl")
		}
		if !given["out"] {
			*outFlag = filepath.Join(*dirFlag, "brain.gob")
		}
		if !given["epochs"] {
			*epochsFlag = stdEpochs
		}
		if !given["layers"] {
			*layersFlag = stdLayers
		}
		if !given["augment"] {
			*augmentFlag = stdAugment
		}
		if !given["synonyms"] {
			*synonymsFlag = stdSynonyms
		}
		if !given["ckpt-every"] {
			*ckptEveryFlag = stdCkptEvery
		}
		fmt.Printf("train: domain folder %s — data %s, out %s, standard recipe (epochs %d, layers %s, augment %.1f, synonyms %.1f)\n",
			*dirFlag, *dataFlag, *outFlag, *epochsFlag, *layersFlag, *augmentFlag, *synonymsFlag)
	}

	if *dataFlag == "" {
		fmt.Fprintln(os.Stderr, "train: -data is required (or give -dir for a domain folder)")
		flag.Usage()
		os.Exit(2)
	}
	pairs, err := data.LoadPairs(*dataFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, "train:", err)
		os.Exit(1)
	}
	var layers []string
	for _, name := range strings.Split(*layersFlag, ",") {
		if name = strings.TrimSpace(name); name != "" {
			layers = append(layers, name)
		}
	}
	if err := brain.Train(pairs, *outFlag, brain.TrainOptions{
		Layers:          layers,
		Epochs:          *epochsFlag,
		Resume:          *resumeFlag,
		Augment:         *augmentFlag,
		Synonyms:        *synonymsFlag,
		Tokenizer:       *tokenizerFlag,
		BPEMerges:       *bpeMergesFlag,
		CheckpointEvery: *ckptEveryFlag,
	}); err != nil {
		fmt.Fprintln(os.Stderr, "train:", err)
		os.Exit(1)
	}
}
