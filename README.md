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

Rows may also carry an `"intent"` field — a meaning label of
your own — and an intent row may omit `"output"` entirely.
That is the command mechanism; see "Tie sentences to
commands" below.

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
source's synonym data augmentation. (Its mutation
augmentation did make the trip — next section.)

## Mutation augmentation

People do not type like training data. `-augment` teaches
the brain that a line's meaning survives sloppy typing:
each epoch, every pair's input is re-rolled, and with the
probability you set it is replaced — for that epoch only —
by a lightly mangled variant: one non-content word dropped,
two adjacent words swapped, one small typo in a longer word
(a letter deleted, swapped, or duplicated, never the first
letter), or a trailing punctuation mark or filler word
dropped. The reply is never touched: mangled line in,
correct reply out, so the model learns the mangling does
not change the meaning. The stream is seeded, so runs are
reproducible; `0` (the default) is off, and training is then
exactly what it was before the augmentation existed.

```sh
go run ./cmd/train -data examples/pairs.jsonl -out brain.gob -epochs 60 -layers copy -augment 0.3
```

Measured on the bundled example set (204 pairs, 60 epochs,
`copy` layer), against 36 held-out paraphrases written for
the test — rewordings, shortened lines, and typo'd lines
whose phrasings appear nowhere in the corpus. Both brains
started from identical weights (same seed, one process), so
augmentation is the only difference. A reply carries the
topic when it contains the line's topic word:

| run | held-out, greedy | held-out, checked sampling | trained lines, checked | final avg_loss |
| --- | --- | --- | --- | --- |
| `-augment 0` | 11/36 | 17/36 | 9/12 | 1.338902 |
| `-augment 0.3` | 13/36 | 18/36 | 10/12 | 1.065940 |

Checked sampling is the `cmd/ask` decode policy. The
trained-lines column is a sanity check on 12 lines taken
verbatim from the training split: augmentation did not wreck
what the brain had memorized. On the looser any-content-word
rule (`brain.ReplyCarries`), checked sampling scored 24/36
without augmentation and 28/36 with it. One caveat on the
loss column: the augmented run's loss is averaged over
batches that include mangled inputs, so the two loss numbers
are not measured on identical data.

Real exchanges (greedy replies, before → after):

```
> got any thoughts on cycling
hi there! good to see you.  →  cycling is wonderful. what do you enjoy most about it?
> gardening sounds relaxing
i am a tiny brain, trained on a small file example pairs.  →  gardening is wonderful. what do you enjoy most about it?
> thoughts on yoga
hi there! good to see you.  →  yoga is wonderful. what do you enjoy most about it?
> seen any good movies lately
hi there! good to see you.  →  hello! how are you today?
> tell me about cooknig
you the mind. a trail, some, and fresh air fix many  →  (empty reply)
```

The last two are in here on purpose: augmentation is a
nudge, not a cure. The wins are real but modest, and they
landed on reworded and shortened lines — the six typo'd
lines in the set put their topic word in the reply 0/6
times for *both* brains. A typo'd topic word is an unknown
token, and at this scale the model still cannot bridge it.
More data remains the bigger lever.

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

## Tie sentences to commands

A reply is not always what you want from a line — sometimes
you want to KNOW WHAT IT MEANS and act on it. Add an
`"intent"` field to a row: a meaning label you define.

```json
{"input": "turn off the lights", "intent": "lights_off"}
{"input": "lights out", "intent": "lights_off"}
{"input": "turn the lights off please", "output": "Turning the lights off.", "intent": "lights_off"}
```

Most rows in a command set are **label-only** — an input and
an intent, no output. A command set should not need chat
replies invented for it. The brain's trainer skips
label-only rows (there is no reply to learn); the intent
classifier trains on every row that carries an intent, with
or without an output. `examples/commands.jsonl` is a working
demo: 6 meanings (lights_on, lights_off, play_music, stop,
set_timer, weather_ask) × 10 phrasings, most of them
label-only.

Training states plainly what it picked up:

```
[train] dataset: 18 pairs -> 14 train, 4 held out; vocab size=53
[train] 42 of the 60 rows are label-only (intent, no output): skipped for the brain, used by the intent classifier
[train] intents: 6 labels from 60 rows -> brain.gob.intents.json
[train] intent lights_off       10 rows
...
```

The classifier — a small bag-of-words softmax in the
`intent` package — is written next to the brain as
`<model>.intents.json` and loaded automatically by
`LoadBrain` when present. Your program switches on the
label and runs whatever it binds to that meaning:

```go
b, err := brain.LoadBrain("brain.gob")
if err != nil {
	log.Fatal(err)
}
line := "turn off the lights"
if b.Intents != nil {
	switch label, _ := b.Intents.Predict(line); label {
	case "lights_off":
		hallwayLight.Off()
	case "lights_on":
		hallwayLight.On()
	case "":
		// abstained — no confident meaning, do nothing
	}
}
```

**The abstention rule:** below a confidence of 0.5 (the
default threshold, configurable), `Predict` returns no
label at all — and a line that shares no words with any
training row gets none either. That is deliberate: a
program that acts on predictions will be handed sentences
the classifier barely understands, and a wrong command is
worse than none.

`cmd/ask -intents` shows the mechanism:

```
> turn off the lights
the lights off.
intent: lights_off (1.00)
> tell me a joke
cannot started. see here sky tell but i hope weather is kind.
intent: set_timer (0.50)
```

Measured on the demo set's classifier against 24 held-out
lines — 18 unseen phrasings of the six meanings, plus 6
lines that belong to no meaning:

| held-out lines | result |
| --- | --- |
| 18 meaning lines | **12 correct**, 4 wrong, 2 abstained |
| 6 no-meaning lines | 2 abstained, **4 false-fired** |

Verbatim, one of each: "could you switch on the lights" →
`lights_on (1.00)`, correct. "no more music please" →
`play_music (0.98)`, wrong — it means stop, but a
bag-of-words cannot see the "no"; the words *music please*
are the strongest evidence it knows. "how about a tune" →
abstained (best guess 0.36). And the honest weak spot: the
no-meaning lines false-fired more often than they
abstained — "what is your name" → `weather_ask (0.70)`,
because *what is* is all the evidence the classifier has
and in training those words only ever meant weather. More
rows per meaning, with more varied wording, is the lever;
the classifier itself is exactly as smart as a bag of
words gets.

## Runnable example: talk to your filesystem

`examples/filecmd` puts the intent mechanism to work end
to end. You type plain sentences; files are created,
deleted, and listed in a `demo-files/` folder — and there
is no sentence parsing in the program at all. The trained
intent classifier says *what the line means*; a second
trained brain, the same seq2seq, acts as the argument
extractor and says *which filename*; the host program
contributes exactly three things: one small binding
function per meaning, a sanitizer for the extractor's
answer, and the safety gates. Nothing outside
`demo-files/` is ever touched.

The data is `examples/filecommands.jsonl`: three meanings
(`create_file`, `delete_file`, `list_files`) × 12
phrasings. The create/delete rows also carry an `"output"`
— the bare filename in the sentence — which is what
trains the extractor; the list rows are label-only.

```sh
go run ./cmd/train -data examples/filecommands.jsonl -out filebrain.gob -epochs 300 -layers char,copy -augment 0.5
go run ./examples/filecmd -model filebrain.gob
```

```
> make me a brand new file named budget.txt
intent: create_file (1.00), file: budget.txt
created demo-files/budget.txt
> get rid of ideas.txt
intent: delete_file (0.77) — not confident enough to delete (this demo wants 0.80); nothing done.
```

(Both exchanges are from the measured run below —
including the delete the confidence gate refused.)

The safety rules are the host's, on purpose: the
extractor's answer is accepted only as a single plain
filename (no separators, no `..`, checked again against
the sandbox folder); `delete_file` needs a confidence of
at least 0.80 *and* a typed yes at a prompt that names
the file; a line with no confident meaning does nothing
and says so.

**Measured, not promised.** The try-these list below was
run against a brain trained with exactly the command
above; every phrasing is absent from the training file.
Scored by what the program actually did:

| try-these lines (unseen) | right action | wrong action | abstained |
| --- | --- | --- | --- |
| 5 create phrasings | 2 | 3 | 0 |
| 4 delete phrasings | 0 | 3 | 1 (the 0.80 gate) |
| 3 list phrasings | 3 | 0 | 0 |
| 3 lines with no meaning in the set | 1 | 2 | 0 |
| **total, 15 lines** | **6** | **8** | **1** |

Read the two halves separately, because they are not
equally strong. The intent classifier landed the right
meaning on **12 of 12** meaning lines, unseen phrasings
included — "brand new" is a word it never trained on, and
"make me a brand new file named budget.txt" still came
back `create_file (1.00)` with the right file. The
extractor is the weak half: it produced the right
filename in 2 of 5 creates and 0 of 3 deletes it attempted
(the fourth delete was stopped by the confidence gate
before extraction). The trained control line made the
point plainly — "create a file called notes.txt" was
understood perfectly and then created **todo.md**.

A verification run after the fact found the extractor's
hard boundary: "make me a brand new file named
summit.txt" — a filename that appears nowhere in
training — came back `create_file (1.00)` with the file
**budget.txt**. The intent half did not care that the
name was new; the extractor collapsed to a favourite,
because a word-level model has no token for a word it
never trained on, so there is literally nothing for it
to copy. Names from the training set extract; brand-new
names are this design's wall (the BPE chapter below is
the piece that speaks to it).

Failures, verbatim from the run:

- "please remove budget.txt" → extracted `ideas.txt`;
  the confirmation prompt *named* ideas.txt, and the
  scripted "y" deleted the wrong file. A person reading
  the prompt would have caught it — that is what the
  prompt is for — but the machine did not.
- "tell me a joke" → `create_file (0.78)`, extracted
  `recipes.txt`, and created it. A false-fire with a real
  (sandboxed) side effect: the abstention threshold is a
  floor, not a guarantee, exactly as the intent chapter
  warns.
- "i don't need todo.md anymore, delete it" →
  `delete_file (0.99)`, extracted `workout.md`, which did
  not exist, so nothing happened.

And the pair the example exists for: the trained line
"list my files" and the never-trained "what files do i
have" produced the identical, correct action — the same
listing of the folder's real contents. Unseen phrasings
of a trained meaning: yes, that works, measured above.
A brand-new *meaning* is the boundary: it needs its first
example rows plus one binding function in the host —
meanings are declared, not discovered.

## Improving unseen answers

The chapters above describe a brain that is strong on what
it was taught and weak on everything else. This section is
the record of attacking that weakness directly — four
steps, each one committed separately and **measured before
it counted**, all against the same yardsticks: 36 held-out
paraphrases of the example corpus (30 clean rewordings +
typo'd lines), written before any of the steps and never
trained on. The mutation augmentation chapter above was
the warm-up; its own numbers live there. A reply "carries"
when it contains the line's topic word.

### Step A — the typo normalizer (ask-side, no retraining)

A typo'd word is an unknown token, and one unknown token
can derail a whole reply — the augmentation chapter's
brains scored 0/6 on the typo'd lines no matter what the
training did. So don't fix the model; fix the input.
Before a line is encoded, any word that is out of the
vocabulary, at least 4 letters, and has **exactly one**
close vocabulary neighbour (Damerau–Levenshtein: 1 edit
for short words, 2 for long ones) is replaced by that
neighbour. A tie, or no neighbour, leaves the word alone;
a line needing more than two replacements passes through
untouched. It is the seat Gollemer's chat grew, ported
(`brain/normalize.go`), on by default in `cmd/ask`
(`-typos=false` turns it off), and purely an inference
switch: the same brain, measured both ways.

| set | seat off | seat on |
| --- | --- | --- |
| 10 typo'd lines, greedy | 0/10 | **10/10** |
| 10 typo'd lines, checked sampling | 0/10 | **10/10** |
| 30 clean lines, greedy | 11/30 | 10/30 |
| 30 clean lines, checked sampling | 16/30 | 17/30 |

Every typo in the set had a unique defensible neighbour,
and once the word was repaired the un-augmented base brain
answered on-topic every time:

```
> do you like gardning
yes, i like gardening a lot. it is one of my favorite topics.
> how do i start fishnig
start with a short, fishing, or fishing gets first, time you second.
```

(the second reply is rough — the seat guarantees the topic
reaches the encoder, not that the sentence is pretty).

The clean set is the honest cost: a wash, one line each
way. The seat is context-free, so it sometimes "corrects"
a real word that simply isn't in the vocabulary —
`i think it will rain later` became `i think it well rain
later` (*will* → *well*, one edit, unique neighbour), and
its greedy reply lost the topic (checked sampling still
carried it). `gardening sounds relaxing` became
`gardening sounds reading` — wrong, though that line
failed both ways. The rule is defensible word by word;
it is not always right.

### Step B — synonym augmentation (trainer-side)

A small built-in table of near-synonym pairs drawn from the
example corpus's own vocabulary (like/love/enjoy, good/great,
tired/sleepy, start/begin, ...). With `-synonyms p`, each
training pair's input may have one word swapped for a table
synonym with probability `p` per epoch, re-rolled every epoch
from the same seeded stream as mutation augmentation. The
swap only lands when the synonym is itself in the training
vocabulary, and the target reply is never touched: the brain
learns that "I love gardening" and "I enjoy gardening" are
the same conversation. It composes with `-augment` (a line
can get both a synonym and a typo).

Measured on the same 36-line held-out set, same settings
(60 epochs, copy layer, mutation 0.3 in both arms, typo
normalizer off):

| training | greedy | checked sampling |
|---|---|---|
| mutation only | 13/36 | 19/36 |
| mutation + synonyms 0.3 | 14/36 | 23/36 |

Where it helped — "i am so tired today" (tired/sleepy is a
table pair):

- mutation only: "am with to hear you fine. what you been
  today up to?"
- with synonyms, checked: "tired am to rest walk, a book,
  or a quiet or little pages of forever." (on-topic, if
  scrambled)

Where it did not — "i think it will rain later" flipped the
other way:

- mutation only: "that is great. rain makes many people
  happy too."
- with synonyms: "start with a walk, book, or book or, and
  air every pass to time."

Seven held-out lines were gained and three lost on checked
sampling. The synonym arm's training loss also runs higher
(1.241 vs 1.066) — expected, since its inputs genuinely vary
more; the losses are not comparable across arms.

### Step C — concept data (contrast + category pairs)

Steps A and B change the machinery; this step changes the
data. `examples/pairs.jsonl` gained 247 hand-written pairs
in two families over the topics the file already used:

- **contrast** — either/or questions and "not X but Y"
  lines ("do you like gardening or cooking", "i like tea,
  not coffee"), whose replies name *both* topics;
- **category** — grouping lines ("is gardening a hobby or
  a job", "what kind of thing is coffee") and boundary
  lines that say no with the right category ("is coffee a
  hobby" → "No, coffee is a drink, not a hobby.").

Measured against a 16-line held-out concept set (shapes
kept out of training), same settings in both arms
(60 epochs, copy layer, mutation 0.3 + synonyms 0.3):

| training data | concept greedy | concept checked | main set greedy | main set checked |
|---|---|---|---|---|
| original 204 pairs | 6/16 | 12/16 | 14/36 | 23/36 |
| extended 451 pairs | 12/16 | 14/16 | 18/36 | 20/36 |

The topic rule only asks for the topic word, so a sharper
count for the category half: how often the reply names the
*category* (hobby, drink, pet, vehicle...). Old data:
greedy 1/8, checked 2/8. Extended: 4/8 both. The extended
brain learned to say the grouping out loud:

- "is yoga a sport" → "no, yoga is exercise. bread is a
  food." (the category lands; the tail is a borrowed
  template)
- "is fishing a hobby" → "no, fishing is a hobby." (right
  category, wrong yes/no — the boundary family's "no"
  leaked onto a yes question)

And the honest costs. The plural category shape was never
trained, and it shows: "are trains vehicles" passed with
the old data and fails with the new ("that is sweet of are
to say. kindness is easy to share."). On the main set,
greedy rose 14 → 18 but checked sampling *fell* 23 → 20 —
five familiar lines (morning coffee, tired today, dogs as
friends among them) lost their checked rescue to greeting
deflections like "hello! how are you today?". Concept data
teaches concepts; it also dilutes the chat lines, and at
this corpus size both effects are real.

### Step D — BPE subword tokenizer (`-tokenizer bpe`)

The word tokenizer has a hard wall: a word that never
appeared in training has no token, so the model cannot see
it or say it. The `bpe` package learns byte-pair merges
from the training text itself (700 merges here, `-bpe-merges`
configures it): frequent words stay whole or nearly so, and
everything else splits into shared pieces — "knitting"
becomes pieces of "painting" and "fishing". `-tokenizer
word|bpe` on cmd/train chooses; the choice and the merges
are recorded in the brain file, and LoadBrain restores the
matching tokenizer or refuses the file loudly. Word brains
keep the original bare-gob format, unchanged.

Measured on the extended examples (60 epochs, copy layer,
mutation + synonyms 0.3 in both arms). The probe set is 12
lines whose one content word appears *nowhere* in training
(knitting, swimming, chess, sailing, ...):

| tokenizer | vocab-wall probe | main held-out greedy | main held-out checked |
|---|---|---|---|
| word (vocab 584) | 0/12 | 18/36 | 20/36 |
| bpe (vocab 1470) | 0/12 | 19/36 | 24/36 |

Read the probe result carefully — the zeros are not the
same zero. The word brain answers "how do i start chess"
with "start with a tiny program that prints your name.":
the topic is simply invisible to it. The BPE brain's
replies are full of the probe words' *pieces* — "do you
like singing" → "yes, i likeing a lot. it is one of my
favorite topics.", "tell me about swimming" → "wi clears
the mind..." — it routes the line by the pieces it shares
with known words and starts to say the word, but at this
scale it never assembles one whole, so the topic rule
scores it 0/12 too. The wall is gone; what is on the other
side is fragments, not knowledge. BPE composes pieces of
known words — it does not teach new topics, and a brain
that never saw a line about sailing has nothing true to
say about it, in any tokenizer.

The surprise ran the other way on known words: the main
held-out set *improved* under BPE (checked 20/36 → 24/36),
because related words share pieces — gardening/garden,
painting/paint — and each piece gets more training than
each whole word did. The costs are real too: piece-level
slips ("that great. coding makes many people happy too.",
the dropped "is"), occasional fragments glued onto
punctuation, a vocabulary 2.5× larger, and epochs about
1.5× slower.

Taken together, the four steps say: repair the input when
you can (A), vary the training inputs (B), teach the
groupings explicitly (C), and remove the vocabulary wall
(D) — each moves unseen lines a measured amount, none of
them turns a tiny model into a knowledgeable one. The
intent layer above is what makes the remaining gaps safe
to ship: when the brain does not know, the meaning
classifier can still abstain.

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
intent/     the meaning classifier: bag-of-words softmax
            over "intent" labels, with confidence abstention
brain/      standalone glue: vocab building, the trainer loop
            with checkpoint-resume, greedy + checked sampling
cmd/train   train a brain: -data -out -epochs -layers -resume
cmd/ask     a demo REPL for a trained brain (not a chat product)
examples/   pairs.jsonl — 204 toy pairs to start from;
            commands.jsonl — 60 command rows with intents;
            filecommands.jsonl — 36 file-command rows, and
            filecmd/ — the runnable filesystem demo for them
```

## Provenance

Carved from [github.com/golangast/gollemer](https://github.com/golangast/gollemer)
(the `internal/ai/neural/...` packages and the layers trainer
in its chat package) in November 2026, and relicensed under
GPL-3.0 for this standalone form. It is a copy, not a live
mirror: improvements in Gollemer do not flow here automatically.
Three carved files were gofmt'd during the extraction (two were
already unformatted upstream).
