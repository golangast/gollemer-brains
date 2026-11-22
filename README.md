# gollemer-brains

A tiny, pure-Go neural seq2seq brain you can train on your own
data — plus the simplest possible way to feed it that data.

This is the reusable core carved out of
[Gollemer](https://github.com/golangast/gollemer), a beginner-first
Go chat tool whose social brain is a small LSTM seq2seq model
trained from scratch (no frameworks, no rented models, standard
library only). Gollemer grew a whole product around that brain;
this repo is just the brain, its trainer, and a demo REPL, so
anyone can train their own.

**License: GPL-3.0** (see LICENSE).

## Quickstart

You need Go (this repo builds with Go 1.26; the code is pure
standard library — no dependencies to download).

Train the bundled example dataset (204 pairs of tiny small talk):

```sh
go run ./cmd/train -data examples/pairs.jsonl -out brain.gob -epochs 60 -layers copy
```

Real output from that run (abridged — one line per epoch in the
actual log):

```
[train] dataset: 204 pairs -> 163 train, 41 held out; vocab size=502
[train] 163 encoded pairs, 6 batches/epoch, lr=0.001000, layers=[copy], epochs<=60
[train] epoch=1 avg_loss=5.991518 best=5.991518 lr=0.001000
...
[train] epoch=60 avg_loss=1.336682 best=1.336682 lr=0.000500
[train] saved best model (loss=1.336682) to brain.gob
[train] held-out Q: i like painting
[train] held-out A: painting is wonderful. what do you enjoy most about it? (reference: Painting is wonderful. What do you enjoy most about it?)
```

A few dozen epochs on the toy set already shows learning (the
loss falls and replies become readable); the default of 200
epochs is the full effect. On a busy shared VM the 60-epoch run
above took about 17 minutes; on a quiet desktop it is much
faster. Bigger datasets take proportionally longer per epoch.

Then talk to it:

```sh
go run ./cmd/ask -model brain.gob
```

```
brain loaded. type a line, or 'quit' to exit.
> hello
hello! how you today?
> do you like gardening
gardening, i like gardening a lot. it is one of my favorite topics.
> hey, how is your day going
i day going not going brain, trained on a small file example pairs.
```

That last line is in here on purpose: it is an *unseen
paraphrase*, and the tiny brain garbles it. That is the honest
state of a model this small trained on 204 pairs — strong on
what it was taught and near neighbours, weak on genuinely new
phrasings. More data and more epochs are the cure, in that
order.

## Your own data

One JSONL file. One pair per line:

```json
{"input": "hello", "output": "Hello! How are you today?"}
```

That is the whole contract. Extra fields are ignored, so
Gollemer's own `chat_pairs.jsonl` (which carries a `"domain"`
field) drops in unchanged. Malformed lines are rejected with
the line number named; empty inputs or outputs are rejected
the same way.

Loading in code is one call:

```go
pairs, err := data.LoadPairs("mine.jsonl")
```

Training splits your file deterministically, 80% train / 20%
held out (fixed seed — same file, same split, every run). The
vocabulary is built from the training split only, so words
that appear only in held-out pairs stay unknown: the held-out
set is a real yardstick, and the trainer prints a few held-out
predictions next to their references when it finishes.

## Checkpoint-resume (train like you mean it)

Real training runs get killed — laptops close, VMs reboot.
This trainer is built for that, mirroring the discipline the
source project learned the hard way:

- While training, two checkpoints live next to your `-out`
  file: `<out>.ckpt` (+ `.ckpt.json` sidecar) holds the **best**
  weights so far, refreshed on improvement; `<out>.latest`
  (+ sidecar) is rewritten **every epoch** with the current
  weights, so a killed run never replays finished epochs.
- The production `.gob` is written **once, at the end**, from
  the best weights — a killed or diverged run can never leave
  a half-trained model in its place. Checkpoints are removed
  when a run completes.
- Rerun with `-resume` to continue. Resume **refuses** a
  checkpoint whose vocabulary fingerprint or layer config
  differs from the current run — change your data and you get
  a clean refusal and a fresh start, never silently corrupted
  weights.
- The loop itself: batch 32, Adam (lr 1e-3, halved every 40
  epochs), gradient clip 1.0, divergence rollback to the best
  weights with a halved LR (stops after 5 rollbacks), and an
  early stop after 30 epochs without improvement.

## Layers

The model is a layers stack around the base seq2seq. The
standalone trainer supports the layers that need no extra
corpus machinery:

- `copy` — a pointer-generator copy mechanism: the decoder can
  copy words straight from your input line. This is the layer
  the source project's best brain was adopted with; it is what
  lets replies name the topic you mentioned even in
  combinations training never showed. (It also produces the
  occasional topic-echo at the start of a reply, as in the
  gardening answer above.)
- `char` — a character-trigram layer that gives rare and
  misspelled words a foothold.

```sh
go run ./cmd/train -data mine.jsonl -layers copy
go run ./cmd/train -data mine.jsonl -layers char,copy
```

**Left behind, deliberately:** the source's `class`, `frame`,
`turn`, and `memory` layers. Their training signal is computed
by machinery that lives in Gollemer's chat product (a fact
extractor, an intent labeller, reply clustering), so they do
not carve into a standalone brain honestly. Naming one here is
a clear error that says so. Also left behind: the whole chat
product itself (routing, tools, memory, receipts), the
transformer and pointer-generator probe models, and the
source's synonym/mutation data augmentation.

## Generation: greedy + checked sampling

`cmd/ask` answers with **checked sampling**, the decode policy
the source project measured before adopting: the greedy reply
stands when it carries the line's topic; otherwise up to 8
replies are sampled at temperature 0.8 and the first
on-topic one wins. In the source that policy measurably beat
plain greedy on unseen combinations.

One simplification, stated plainly: in the source, "topic"
came from a bank of phrases the model was trained on, falling
back to the line's content words. The standalone keeps only
the fallback — a reply carries the line when it contains at
least one of the line's content words (`brain.ReplyCarries`).
The check guarantees the topic is present, not that the
sentence is good; a rescued reply can still be rough.

## Honest limits

- This is a **tiny CPU model** (128-dim embeddings, 256-dim
  hidden state). It learns the shape of your data — phrasings,
  topics, combinations of things it has seen. It will not
  reason, and it will not generalize like a large model.
- The **vocabulary is closed**: it is built from your training
  pairs, and any other word is an unknown token. Unseen words
  are the hard wall, not a bug.
- Short, ambiguous lines are its thinnest ground. Near-identical
  inputs with different outputs blur together.
- Training is single-threaded Go on CPU. Fine for thousands of
  pairs; not a datacenter trainer.

## Layout

```
tensor/     tensor math (pure Go; optional SIMD assembly included)
nn/         LSTM, embeddings, MoE, Adam — the neural building blocks
vocab/      the vocabulary
tokenizer/  word tokenization, in sync with the trainer's splitter
seq2seq/    the model: encoder/decoder, BPTT training, layers
            (copy/char/frame/...), gob save/load, prediction
data/       the JSONL pairs loader and the 80/20 split
brain/      standalone glue: vocab building, the trainer loop
            with checkpoint-resume, greedy + checked sampling
cmd/train   train a brain: -data -out -epochs -layers -resume
cmd/ask     a demo REPL for a trained brain (not a chat product)
examples/   pairs.jsonl — 204 toy pairs to start from
```

## Provenance

Carved from [github.com/golangast/gollemer](https://github.com/golangast/gollemer)
(the `internal/ai/neural/...` packages and the layers trainer
in its chat package) in November 2026, and relicensed under
GPL-3.0 for this standalone form. It is a copy, not a live
mirror: improvements in Gollemer do not flow here automatically.
Three carved files were gofmt'd during the extraction (two were
already unformatted upstream).
