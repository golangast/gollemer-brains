// Command train trains a brain on a JSONL pairs file.
//
//	go run ./cmd/train -data examples/pairs.jsonl -out brain.gob -epochs 200 -layers copy
//
// Add -augment 0.3 to also train on lightly mangled variants
// of the inputs (a dropped filler word, a typo, swapped
// neighbours) whose replies stay correct — it teaches the
// brain that meaning survives sloppy typing. Off by default.
//
// If the data carries "intent" fields, an intent classifier
// is trained on those rows too (label-only rows — an intent
// with no output — exist for exactly this) and written next
// to the brain as <out>.intents.json; the run's output lists
// the labels it found and the rows per label.
//
// The output .gob is written once, at the end, from the best
// weights seen. While training runs, checkpoints live next to
// it (<out>.ckpt / <out>.latest with JSON sidecars); rerun
// with -resume to continue a killed run instead of starting
// over. See the README for the data format and the checkpoint
// discipline.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/golangast/gollemer-brains/brain"
	"github.com/golangast/gollemer-brains/data"
)

func main() {
	dataFlag := flag.String("data", "", "path to the JSONL pairs file (required)")
	outFlag := flag.String("out", "brain.gob", "where to write the trained brain")
	epochsFlag := flag.Int("epochs", 200, "total epochs to reach")
	layersFlag := flag.String("layers", "", "comma-separated layers to switch on (supported: char, copy)")
	resumeFlag := flag.Bool("resume", false, "resume from the checkpoints next to -out instead of starting at epoch 1")
	augmentFlag := flag.Float64("augment", 0, "probability that a pair's input is mutated in each epoch (0 = off, 0.3 = light)")
	flag.Parse()

	if *dataFlag == "" {
		fmt.Fprintln(os.Stderr, "train: -data is required")
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
		Layers:  layers,
		Epochs:  *epochsFlag,
		Resume:  *resumeFlag,
		Augment: *augmentFlag,
	}); err != nil {
		fmt.Fprintln(os.Stderr, "train:", err)
		os.Exit(1)
	}
}
