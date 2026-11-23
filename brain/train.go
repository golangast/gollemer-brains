package brain

// The standalone trainer, carved from Gollemer's layers trainer
// (internal/ai/training/chat/layers_train.go in the source) with
// its loop discipline intact:
//
//   - batch 32, Adam at lr 1e-3, gradient clip 1.0, LR halved
//     every 40 epochs;
//   - divergence rollback: a NaN/Inf epoch, or an epoch after
//     epoch 5 landing above 3x the best loss, restores the best
//     weights and halves the LR; more than 5 rollbacks stops
//     the run;
//   - best-on-train-loss selection: the production .gob is
//     written ONCE, at the end, from the best weights seen;
//   - checkpoints next to the output file: <out>.ckpt (+ .json
//     sidecar) holds the best weights, refreshed on improvement
//     with a 2-epoch minimum spacing; <out>.latest (+ .json) is
//     rewritten every epoch with the current weights, so a
//     killed run resumes where it stopped instead of replaying
//     from the last improvement — TrainOptions.CheckpointEvery
//     can space those .latest writes out (every N epochs),
//     trading a few replayed epochs for less checkpoint I/O;
//   - resume refuses a checkpoint whose vocabulary fingerprint
//     or layer config differs from the current run's — a vocab
//     index shift would silently corrupt the weights;
//   - early stop after 30 epochs without improvement.
//
// What the carve dropped, and why: the source's synonym
// augmentation (chat-dataset machinery), the frame/class/turn/
// memory layers and their corpus derivations (they are computed
// by the chat package's fact seat and intent labeller), and the
// session probes. The source's mutation augmentation IS here,
// simplified to a per-epoch re-roll: see mutate.go. The layers
// that carve cleanly — char and copy — are supported; see
// SupportedLayers.

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/golangast/gollemer-brains/bpe"
	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/intent"
	"github.com/golangast/gollemer-brains/nn"
	"github.com/golangast/gollemer-brains/seq2seq"
	"github.com/golangast/gollemer-brains/tokenizer"
	"github.com/golangast/gollemer-brains/vocab"
)

const (
	batchSize = 32
	baseLR    = 1e-3
	gradClip  = 1.0
	embedDim  = 128
	hiddenDim = 256
)

// SupportedLayers are the layer names the standalone trainer
// accepts: the ones that need no chat-side corpus derivations.
// "copy" is the mechanism the source project's best brain was
// adopted with; "char" is the character-trigram layer.
var SupportedLayers = []string{"char", "copy"}

// droppedLayers are real layers in the source whose derivations
// live in the chat product; naming one is a clear error, not a
// silent no-op.
var droppedLayers = []string{"class", "frame", "frame-top", "frame-slots", "frame-soft", "turn", "memory"}

// ckptMeta is the sidecar written next to every checkpoint. It
// records exactly where a killed run stopped so -resume can
// pick up with the same learning rate, best-loss tracking, and
// vocabulary it had. Copied from the source trainer.
type ckptMeta struct {
	Epoch              int     `json:"epoch"`
	Best               float64 `json:"best"`
	LastCheckpointBest float64 `json:"last_checkpoint_best"`
	LR                 float32 `json:"lr"`
	EpochsNoImprove    int     `json:"epochs_no_improve"`
	VocabFP            uint64  `json:"vocab_fp"`
}

// TrainOptions configures one training run.
type TrainOptions struct {
	Layers []string // layer names to switch on (see SupportedLayers)
	Epochs int      // total epochs to reach (resume counts from the checkpoint)
	Resume bool     // continue from <out>.ckpt/.latest when usable
	// Augment is the probability that a pair's input is
	// replaced by a mutated variant in any given epoch (see
	// mutate.go). 0 (the default) is off: training is exactly
	// what it was before the augmentation existed. Each epoch
	// re-rolls, so over a run a pair is seen both clean and
	// mangled.
	Augment float64
	// Synonyms is the probability that a pair's input has
	// one word swapped for a known synonym in any given
	// epoch (see synonyms.go). 0 (the default) is off. It
	// composes with Augment: the two rolls are independent.
	Synonyms float64
	// Tokenizer selects the text<->id layer: "" or "word"
	// is the original whole-word tokenizer; "bpe" trains a
	// byte-pair-encoding tokenizer on the training split
	// and builds the vocabulary from its pieces (see the
	// bpe package). The choice is recorded in the saved
	// brain file; LoadBrain restores the matching one and
	// refuses a file it cannot interpret.
	Tokenizer string
	// BPEMerges caps how many merges BPE learns (only used
	// when Tokenizer is "bpe"). 0 means the default, 700.
	BPEMerges int
	// CheckpointEvery spaces the <out>.latest checkpoint
	// writes: 0 or 1 (the default) rewrites it every epoch,
	// exactly as before; N > 1 rewrites it every N epochs,
	// so a killed run replays at most N-1 epochs on resume.
	// The best checkpoint (<out>.ckpt) is not affected — it
	// is already gated on improvement.
	CheckpointEvery int
}

// ckptPathsFor returns the checkpoint paths living next to the
// output model: best (.ckpt) and latest (.latest), each with a
// JSON sidecar.
func ckptPathsFor(out string) (ckptPath, metaPath, latestPath, latestMetaPath string) {
	return out + ".ckpt", out + ".ckpt.json", out + ".latest", out + ".latest.json"
}

// latestDue reports whether the <out>.latest checkpoint is
// written at this epoch under the given cadence (1 = every
// epoch, the historical behaviour). saveCheckpoint writes
// the weights and the sidecar together, so the pair stays
// consistent at any cadence; a sparser cadence only means a
// resumed run replays a few more epochs.
func latestDue(epoch, every int) bool {
	if every < 1 {
		every = 1
	}
	return epoch%every == 0
}

func saveCheckpoint(model *seq2seq.LayersModel, meta ckptMeta, ckptPath, metaPath string) error {
	if err := os.MkdirAll(filepath.Dir(ckptPath), 0o755); err != nil {
		return err
	}
	tmp := ckptPath + ".tmp"
	if err := model.Save(tmp); err != nil {
		return err
	}
	if err := os.Rename(tmp, ckptPath); err != nil {
		return err
	}
	data, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	return os.WriteFile(metaPath, data, 0o644)
}

func loadCheckpoint(tok *tokenizer.Tokenizer, curFP uint64, ckptPath, metaPath string) (*seq2seq.LayersModel, ckptMeta, error) {
	var meta ckptMeta
	raw, err := os.ReadFile(metaPath)
	if err != nil {
		return nil, meta, fmt.Errorf("no checkpoint sidecar: %w", err)
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		return nil, meta, fmt.Errorf("bad checkpoint sidecar: %w", err)
	}
	if meta.VocabFP != curFP {
		return nil, meta, fmt.Errorf("checkpoint vocab fingerprint %d != current %d; refusing to resume across data changes", meta.VocabFP, curFP)
	}
	model, err := seq2seq.LoadLayers(ckptPath, tok)
	if err != nil {
		return nil, meta, err
	}
	return model, meta, nil
}

// copyWeightsInto copies src's parameter values into dst in
// place, mirroring the source's resume weight transplant.
func copyWeightsInto(dst, src *seq2seq.LayersModel) error {
	dp := dst.Parameters()
	sp := src.Parameters()
	if len(dp) != len(sp) {
		return fmt.Errorf("parameter count mismatch on resume (%d vs %d)", len(dp), len(sp))
	}
	for i := range dp {
		if len(dp[i].Data) != len(sp[i].Data) {
			return fmt.Errorf("parameter %d size mismatch on resume", i)
		}
		copy(dp[i].Data, sp[i].Data)
	}
	return nil
}

// encodedPair is one training pair with everything the batch
// needs precomputed (the source's layersEncodedPair minus the
// frame teacher labels, which the dropped frame layer owned).
type encodedPair struct {
	input       []int
	inputSurfs  []string
	target      []int
	targetSurfs []string
	// srcInput is the pair's original input text, kept so the
	// mutation augmentation can re-roll a mangled variant of
	// it each epoch (mutate.go). Unused when Augment is 0.
	srcInput string
}

// encodeInputSide encodes an input line into the ids and
// surface forms the encoder consumes — the input half of
// encodePair, shared with the mutation re-encode.
func encodeInputSide(input string, v *vocab.Vocabulary, tok *tokenizer.Tokenizer) ([]int, []string) {
	words := TokenizeLower(seq2seq.NormalizeQuery(input))
	ids := EncodeInput(tok, input)
	var surfs []string
	if len(words) == len(ids) {
		surfs = append([]string{}, words...)
	} else {
		surfs = make([]string, len(ids))
		for i, id := range ids {
			surfs[i] = v.GetWord(id)
		}
	}
	return ids, surfs
}

// encoder is the text<->id layer training goes through:
// the word tokenizer, or BPE when bpe is non-nil. The
// seq2seq model only ever sees ids; which tokenizer made
// them is entirely this type's business.
type encoder struct {
	v   *vocab.Vocabulary
	tok *tokenizer.Tokenizer
	bpe *bpe.Tokenizer
}

// inputSide encodes an input line for training.
func (e *encoder) inputSide(input string) ([]int, []string) {
	if e.bpe != nil {
		return bpeEncodeText(e.bpe, e.v, TokenizeLower(seq2seq.NormalizeQuery(input)))
	}
	return encodeInputSide(input, e.v, e.tok)
}

// target encodes a reply for training: [BOS, ids..., EOS],
// like EncodeTarget.
func (e *encoder) target(output string) []int {
	if e.bpe != nil {
		ids, _ := bpeEncodeText(e.bpe, e.v, TokenizeLower(strings.ToLower(output)))
		out := make([]int, 0, len(ids)+2)
		out = append(out, e.v.BosID)
		out = append(out, ids...)
		out = append(out, e.v.EosID)
		return out
	}
	return EncodeTarget(e.tok, e.v, output)
}

// bpeEncodeText splits pre-tokenized words into BPE pieces
// and maps the forms to ids. A form the vocabulary lacks
// (a piece combination training never produced) maps to
// the unknown id, exactly as an unknown word does in word
// mode.
func bpeEncodeText(bt *bpe.Tokenizer, v *vocab.Vocabulary, words []string) ([]int, []string) {
	var ids []int
	var surfs []string
	for _, w := range words {
		for _, form := range bt.EncodeForms(w) {
			ids = append(ids, v.GetTokenID(form))
			surfs = append(surfs, form)
		}
	}
	return ids, surfs
}

// mergesOf returns a BPE tokenizer's merges, or nil for
// the word tokenizer (whose brains save unwrapped).
func mergesOf(bt *bpe.Tokenizer) []bpe.Merge {
	if bt == nil {
		return nil
	}
	return bt.Merges()
}

// encodePair encodes one pair for training. Pairs whose target
// encodes to fewer than 2 tokens (BOS+EOS) or more than 60 are
// skipped, exactly as in the source.
func encodePair(p data.Pair, e *encoder) (encodedPair, bool) {
	ids, surfs := e.inputSide(p.Input)
	tg := e.target(p.Output)
	if len(ids) == 0 || len(tg) < 2 || len(tg) > 60 {
		return encodedPair{}, false
	}
	ep := encodedPair{target: tg, srcInput: p.Input}
	ep.inputSurfs = surfs
	ep.input = ids
	ep.targetSurfs = make([]string, len(tg))
	for i, id := range tg {
		ep.targetSurfs[i] = e.v.GetWord(id)
	}
	return ep, true
}

// makeBatches pads a length-sorted slice of encoded pairs into
// batches, mirroring the source's makeLayersBatches.
func makeBatches(enc []encodedPair, size, padID int) []seq2seq.LayersBatch {
	var out []seq2seq.LayersBatch
	for i := 0; i < len(enc); i += size {
		end := i + size
		if end > len(enc) {
			end = len(enc)
		}
		grp := enc[i:end]
		maxIn, maxTg := 0, 0
		for _, p := range grp {
			if len(p.input) > maxIn {
				maxIn = len(p.input)
			}
			if len(p.target) > maxTg {
				maxTg = len(p.target)
			}
		}
		b := seq2seq.LayersBatch{PadID: padID}
		for _, p := range grp {
			in := make([]int, maxIn)
			copy(in, p.input)
			inS := make([]string, maxIn)
			copy(inS, p.inputSurfs)
			for k := len(p.input); k < maxIn; k++ {
				in[k] = padID
				inS[k] = ""
			}
			tg := make([]int, maxTg)
			copy(tg, p.target)
			tgS := make([]string, maxTg)
			copy(tgS, p.targetSurfs)
			for k := len(p.target); k < maxTg; k++ {
				tg[k] = padID
				tgS[k] = ""
			}
			b.InputIDs = append(b.InputIDs, in)
			b.InputSurfaces = append(b.InputSurfaces, inS)
			b.TargetIDs = append(b.TargetIDs, tg)
			b.TargetSurfs = append(b.TargetSurfs, tgS)
			b.TopCodes = append(b.TopCodes, -1)
			b.ActLabels = append(b.ActLabels, 0)
			b.ToneLabels = append(b.ToneLabels, 0)
			b.PrevTop = append(b.PrevTop, -1)
		}
		out = append(out, b)
	}
	return out
}

// Train trains a brain on pairs and writes it to outPath.
// pairs is the FULL dataset: Train applies the deterministic
// 80/20 split itself (data.Split), builds the vocabulary from
// the training split, and prints a few held-out predictions at
// the end.
func Train(pairs []data.Pair, outPath string, opts TrainOptions) error {
	cfg := seq2seq.DefaultLayersConfig()
	for _, name := range opts.Layers {
		switch {
		case contains(SupportedLayers, name):
			if err := cfg.Set(name, true); err != nil {
				return err
			}
		case contains(droppedLayers, name):
			return fmt.Errorf("layer %q is not in the standalone extraction: its corpus derivations live in Gollemer's chat package (see README)", name)
		default:
			return fmt.Errorf("unknown layer %q (supported: %s)", name, strings.Join(SupportedLayers, ", "))
		}
	}
	if opts.Augment < 0 || opts.Augment > 1 {
		return fmt.Errorf("augment probability %v is outside [0, 1]", opts.Augment)
	}
	if opts.Synonyms < 0 || opts.Synonyms > 1 {
		return fmt.Errorf("synonyms probability %v is outside [0, 1]", opts.Synonyms)
	}
	kind := opts.Tokenizer
	if kind == "" {
		kind = "word"
	}
	if kind != "word" && kind != "bpe" {
		return fmt.Errorf("unknown tokenizer %q (want \"word\" or \"bpe\")", opts.Tokenizer)
	}
	if opts.BPEMerges < 0 {
		return fmt.Errorf("bpe merges %d is negative", opts.BPEMerges)
	}
	epochs := opts.Epochs
	if epochs <= 0 {
		epochs = 200
	}

	// Label-only rows (an intent, no output) are command
	// examples, not chat pairs: there is no reply for the
	// seq2seq to learn, so they stay out of its split and
	// vocabulary. They still train the intent classifier at
	// the end of the run (see trainIntents).
	chatPairs := make([]data.Pair, 0, len(pairs))
	for _, p := range pairs {
		if strings.TrimSpace(p.Output) != "" {
			chatPairs = append(chatPairs, p)
		}
	}
	labelOnly := len(pairs) - len(chatPairs)
	if len(chatPairs) < 10 {
		return fmt.Errorf("need at least 10 pairs with outputs to train, have %d (label-only intent rows do not count)", len(chatPairs))
	}

	trainPairs, heldPairs := data.Split(chatPairs)
	var v *vocab.Vocabulary
	var tok *tokenizer.Tokenizer
	var bpeTok *bpe.Tokenizer
	var err error
	if kind == "bpe" {
		// BPE learns its merges from the training split's
		// own word stream (inputs and outputs, duplicates
		// kept: frequency is the signal), and the
		// vocabulary is built from the pieces it can
		// produce — each in a bare and a "##" continuation
		// form.
		merges := opts.BPEMerges
		if merges == 0 {
			merges = 700
		}
		var words []string
		for _, p := range trainPairs {
			words = append(words, TokenizeLower(p.Input)...)
			words = append(words, TokenizeLower(p.Output)...)
		}
		bpeTok = bpe.Train(words, merges)
		v = vocab.NewVocabulary()
		for _, piece := range bpeTok.Pieces() {
			v.AddToken(piece)
			v.AddToken("##" + piece)
		}
		tok, err = tokenizer.NewTokenizer(v)
		if err != nil {
			return err
		}
		log.Printf("[train] tokenizer: bpe — %d merges learned, %d pieces", len(bpeTok.Merges()), len(bpeTok.Pieces()))
	} else {
		v, tok, err = BuildVocab(trainPairs)
		if err != nil {
			return err
		}
	}
	enc := &encoder{v: v, tok: tok, bpe: bpeTok}
	log.Printf("[train] dataset: %d pairs -> %d train, %d held out; vocab size=%d", len(chatPairs), len(trainPairs), len(heldPairs), v.Size())
	if labelOnly > 0 {
		log.Printf("[train] %d of the %d rows are label-only (intent, no output): skipped for the brain, used by the intent classifier", labelOnly, len(pairs))
	}

	encoded := make([]encodedPair, 0, len(trainPairs))
	for _, p := range trainPairs {
		ep, ok := encodePair(p, enc)
		if !ok {
			continue
		}
		encoded = append(encoded, ep)
	}
	if len(encoded) == 0 {
		return fmt.Errorf("no encodable training pairs")
	}
	sort.Slice(encoded, func(i, j int) bool { return len(encoded[i].input) < len(encoded[j].input) })

	curFP := VocabFingerprint(v)
	ckptPath, metaPath, latestPath, latestMetaPath := ckptPathsFor(outPath)
	latestEvery := opts.CheckpointEvery
	if latestEvery < 1 {
		latestEvery = 1
	}

	startEpoch := 1
	lr := float32(baseLR)
	best := float32(math.Inf(1))
	epochsNoImprove := 0
	lastCheckpointBest := float32(math.Inf(1))
	lastCheckpointEpoch := 0
	var model *seq2seq.LayersModel
	var pendingLatest *seq2seq.LayersModel

	checkCfg := func(m *seq2seq.LayersModel) error {
		if m.Config != cfg {
			return fmt.Errorf("checkpoint config %+v does not match requested %+v; refusing to resume across configurations", m.Config, cfg)
		}
		return nil
	}

	if opts.Resume {
		bm, bmeta, berr := loadCheckpoint(tok, curFP, ckptPath, metaPath)
		lm, lmeta, lerr := loadCheckpoint(tok, curFP, latestPath, latestMetaPath)
		if berr == nil {
			berr = checkCfg(bm)
		}
		if lerr == nil {
			lerr = checkCfg(lm)
		}
		switch {
		case lerr == nil && (berr != nil || lmeta.Epoch > bmeta.Epoch):
			startEpoch = lmeta.Epoch + 1
			lr = lmeta.LR
			best = float32(lmeta.Best)
			epochsNoImprove = lmeta.EpochsNoImprove
			if berr == nil {
				lastCheckpointBest = float32(bmeta.LastCheckpointBest)
				lastCheckpointEpoch = bmeta.Epoch
			} else {
				lastCheckpointBest = float32(lmeta.LastCheckpointBest)
			}
			if berr == nil && !(lmeta.Best < lmeta.LastCheckpointBest) {
				model = bm
				pendingLatest = lm
			} else {
				model = lm
			}
			log.Printf("[train] RESUMED from latest checkpoint: completed epoch=%d best=%.6f lr=%.6f no-improve=%d (continuing at epoch %d)",
				lmeta.Epoch, lmeta.Best, lmeta.LR, lmeta.EpochsNoImprove, startEpoch)
		case berr == nil:
			model = bm
			startEpoch = bmeta.Epoch + 1
			lr = bmeta.LR
			best = float32(bmeta.Best)
			lastCheckpointBest = float32(bmeta.LastCheckpointBest)
			lastCheckpointEpoch = bmeta.Epoch
			epochsNoImprove = bmeta.EpochsNoImprove
			log.Printf("[train] RESUMED from checkpoint: completed epoch=%d best=%.6f lr=%.6f (continuing at epoch %d)",
				bmeta.Epoch, bmeta.Best, bmeta.LR, startEpoch)
		default:
			log.Printf("[train] -resume given but checkpoint unusable (%v); starting fresh from epoch 1", berr)
		}
	}
	if model == nil {
		cfg.EmbedDim, cfg.HiddenDim = embedDim, hiddenDim
		base, err := seq2seq.NewSeq2Seq(v.Size(), v.Size(), embedDim, hiddenDim, tok, v)
		if err != nil {
			return err
		}
		model, err = seq2seq.NewLayersModel(base, cfg, seq2seq.LayersDerivations{})
		if err != nil {
			return err
		}
	} else {
		cfg = model.Config
	}
	opt := nn.NewOptimizer(model.Parameters(), baseLR, gradClip)
	adam, ok := opt.(*nn.Adam)
	if !ok {
		return fmt.Errorf("optimizer is not *nn.Adam")
	}
	adam.SetLearningRate(lr)
	bestParams := adam.SnapshotParameters()
	if pendingLatest != nil {
		if err := copyWeightsInto(model, pendingLatest); err != nil {
			return err
		}
		pendingLatest = nil
	}
	rollbacks := 0

	batches := makeBatches(encoded, batchSize, v.PaddingTokenID)
	log.Printf("[train] %d encoded pairs, %d batches/epoch, lr=%.6f, layers=%v, epochs<=%d",
		len(encoded), len(batches), lr, opts.Layers, epochs)

	// The augmentation stream exists only when an
	// augmentation is on; with both at 0 the loop below
	// trains on the prebuilt batches exactly as it always
	// has.
	var augRng *rand.Rand
	if opts.Augment > 0 || opts.Synonyms > 0 {
		augRng = rand.New(rand.NewSource(augmentSeed))
	}
	if opts.Augment > 0 {
		log.Printf("[train] mutation augmentation on: each pair's input is re-rolled every epoch with probability %.2f (seed %d)", opts.Augment, augmentSeed)
	}
	if opts.Synonyms > 0 {
		log.Printf("[train] synonym augmentation on: each pair's input may swap one word for a known synonym every epoch with probability %.2f", opts.Synonyms)
	}

	for epoch := startEpoch; epoch <= epochs; epoch++ {
		if epoch > 1 && (epoch-1)%40 == 0 {
			lr *= 0.5
			adam.SetLearningRate(lr)
		}
		epochBatches := batches
		if augRng != nil {
			epochBatches = makeBatches(rerollEncoded(encoded, augRng, opts.Augment, opts.Synonyms, enc), batchSize, v.PaddingTokenID)
		}
		var epochLoss float64
		for _, b := range epochBatches {
			adam.ZeroGrad()
			loss, err := model.TrainBatchLayers(b)
			if err != nil {
				return fmt.Errorf("epoch %d: %w", epoch, err)
			}
			epochLoss += float64(loss)
			adam.ClipGradients()
			adam.Step()
		}
		avg := float32(epochLoss / float64(len(epochBatches)))

		if math.IsNaN(float64(avg)) || math.IsInf(float64(avg), 0) || (epoch > 5 && avg > best*3) {
			rollbacks++
			log.Printf("[train] DIVERGENCE at epoch=%d loss=%.6f (best=%.6f); rolling back (lr %.6f -> %.6f)",
				epoch, avg, best, lr, lr*0.5)
			adam.RestoreParameters(bestParams)
			lr *= 0.5
			adam.SetLearningRate(lr)
			if rollbacks > 5 {
				log.Printf("[train] too many rollbacks; stopping")
				break
			}
			continue
		}

		if avg < best {
			best = avg
			bestParams = adam.SnapshotParameters()
			epochsNoImprove = 0
		} else {
			epochsNoImprove++
		}
		log.Printf("[train] epoch=%d avg_loss=%.6f best=%.6f lr=%.6f", epoch, avg, best, lr)

		if best < lastCheckpointBest && epoch-lastCheckpointEpoch >= 2 {
			adam.RestoreParameters(bestParams)
			meta := ckptMeta{
				Epoch:              epoch,
				Best:               float64(best),
				LastCheckpointBest: float64(best),
				LR:                 lr,
				EpochsNoImprove:    epochsNoImprove,
				VocabFP:            curFP,
			}
			if err := saveCheckpoint(model, meta, ckptPath, metaPath); err != nil {
				return err
			}
			lastCheckpointBest = best
			lastCheckpointEpoch = epoch
			log.Printf("[train] checkpointed best model (loss=%.6f) to %s", best, ckptPath)
		}
		if latestDue(epoch, latestEvery) {
			latestLCB := lastCheckpointBest
			if math.IsInf(float64(latestLCB), 1) {
				latestLCB = best
			}
			latestMeta := ckptMeta{
				Epoch:              epoch,
				Best:               float64(best),
				LastCheckpointBest: float64(latestLCB),
				LR:                 lr,
				EpochsNoImprove:    epochsNoImprove,
				VocabFP:            curFP,
			}
			if err := saveCheckpoint(model, latestMeta, latestPath, latestMetaPath); err != nil {
				return err
			}
		}
		if epochsNoImprove >= 30 {
			log.Printf("[train] early stopping: 30 epochs without improvement")
			break
		}
	}

	adam.RestoreParameters(bestParams)
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil && filepath.Dir(outPath) != "." {
		return err
	}
	if err := saveBrainFile(outPath, model, kind, mergesOf(bpeTok)); err != nil {
		return err
	}
	log.Printf("[train] saved best model (loss=%.6f) to %s", best, outPath)
	for _, p := range []string{ckptPath, metaPath, latestPath, latestMetaPath} {
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			log.Printf("[train] warning: could not remove %s: %v", p, err)
		}
	}

	// Held-out spot check: a few pairs the model never trained
	// on, with the reference beside the prediction.
	br := &Brain{Model: model, tok: tok, bpeTok: bpeTok}
	n := len(heldPairs)
	if n > 5 {
		n = 5
	}
	for i := 0; i < n; i++ {
		ans, err := br.Answer(heldPairs[i].Input)
		if err != nil {
			continue
		}
		log.Printf("[train] held-out Q: %s\n[train] held-out A: %s (reference: %s)", heldPairs[i].Input, ans, heldPairs[i].Output)
	}
	return trainIntents(pairs, outPath)
}

// trainIntents fits the intent classifier on every row that
// carries an intent (label-only rows included — they exist
// for exactly this) and writes it as the brain's sidecar,
// <out>.intents.json, where LoadBrain picks it up. When the
// data carries no intents, any sidecar an earlier run left
// at this path is removed: a stale classifier must never be
// mistaken for this brain's.
func trainIntents(pairs []data.Pair, outPath string) error {
	sidecar := intentsPathFor(outPath)
	counts := intent.CountByLabel(pairs)
	if len(counts) == 0 {
		if err := os.Remove(sidecar); err != nil && !os.IsNotExist(err) {
			log.Printf("[train] warning: could not remove stale intent sidecar %s: %v", sidecar, err)
		}
		return nil
	}
	clf, err := intent.Train(pairs, intent.DefaultThreshold)
	if err != nil {
		return err
	}
	if err := clf.Save(sidecar); err != nil {
		return fmt.Errorf("saving intent classifier: %w", err)
	}
	labels := make([]string, 0, len(counts))
	total := 0
	for l, n := range counts {
		labels = append(labels, l)
		total += n
	}
	sort.Strings(labels)
	log.Printf("[train] intents: %d labels from %d rows -> %s", len(labels), total, sidecar)
	for _, l := range labels {
		log.Printf("[train] intent %-16s %d rows", l, counts[l])
	}
	if clf.HasKinds() {
		byKind := map[string][]string{}
		for label, kind := range clf.LabelKinds() {
			byKind[kind] = append(byKind[kind], label)
		}
		kinds := make([]string, 0, len(byKind))
		for k := range byKind {
			sort.Strings(byKind[k])
			kinds = append(kinds, k)
		}
		sort.Strings(kinds)
		log.Printf("[train] kinds: %d kinds over %d labels", len(kinds), len(clf.LabelKinds()))
		for _, k := range kinds {
			log.Printf("[train] kind %-16s %s", k, strings.Join(byKind[k], ", "))
		}
	}
	return nil
}

func contains(xs []string, x string) bool {
	for _, s := range xs {
		if s == x {
			return true
		}
	}
	return false
}
