// Command train trains a brain on a JSONL pairs file.
//
//	go run ./cmd/train -data examples/pairs.jsonl -out brain.gob -epochs 200 -layers copy
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
		Layers: layers,
		Epochs: *epochsFlag,
		Resume: *resumeFlag,
	}); err != nil {
		fmt.Fprintln(os.Stderr, "train:", err)
		os.Exit(1)
	}
}
