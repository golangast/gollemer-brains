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

Train the bundled example dataset (3,283 pairs of tiny small
talk):

```sh
go run ./cmd/train -data examples/pairs.jsonl -out brain.gob -epochs 60 -layers copy -augment 0.3 -synonyms 0.3
```

Real output from that run (abridged — one line per epoch in the
actual log):

```
[train] dataset: 2888 pairs -> 2310 train, 578 held out; vocab size=1834
[train] 2310 encoded pairs, 73 batches/epoch, lr=0.001000, layers=[copy], epochs<=60
[train] epoch=1 avg_loss=5.757236 best=5.757236 lr=0.001000
...
[train] epoch=60 avg_loss=0.084978 best=0.078508 lr=0.000500
[train] saved best model (loss=0.078508) to brain.gob
[train] held-out Q: please play some jazz
[train] held-out A: some ask — that's asking me to time — i follow you. (reference: You want to hear some jazz. I understand — that's a play-music request.)
```

A few dozen epochs on the toy set already shows learning (the
loss falls and replies become readable); the default of 200
epochs is the full effect. On a busy shared VM the 60-epoch run
above took about two hours; on a quiet desktop it is much
faster. Bigger datasets take proportionally longer per epoch.

Then talk to it:

```sh
go run ./cmd/ask -model brain.gob
```

```
brain loaded. type a line, or 'quit' to exit.
> hello
hello! how are you today?
> do you like gardening
yes, i like gardening a lot. it is one of my favorite topics.
> hey, how is your day going
i am doing well, thank you for asking. how are you doing?
```

That last line used to garble ("i day going not going brain,
trained on a small file example pairs.") when the example set
was 204 pairs. The corpus has since grown — 204 → 451 → 2,643
→ 2,888 pairs — and unseen paraphrases mostly land now; the measured
before/after is in Step F below. Mostly is not always: the
tiny brain still splices replies on some new phrasings, and
Step F shows one of those too, verbatim.

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
  `-ckpt-every N` spaces the `.latest` writes to every N
  epochs when you would rather replay a few epochs than pay
  the checkpoint I/O every time.
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

Rows may also carry a `"kind"` — the broad family a
meaning belongs to — and the classifier then predicts a
kind alongside the label: a neighbourhood answer for
lines whose exact meaning it cannot name. That, and the
subword piece features tried alongside it, are measured
in "Classifier: pieces and kinds" below.

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

The data is `examples/filecommands.jsonl`: 1,634 rows.
The first 1,456 are the file rows over three meanings
(`create_file`, `delete_file`, `list_files`). The
filename vocabulary is 240 names —
common file-worthy nouns in `.txt` and `.md`, a few in
`.json`, `.csv`, and `.log` — and every name appears in
exactly six rows: three create phrasings and three
delete phrasings, rotated through 18 sentence shapes per
meaning so no name and no shape dominates. The
create/delete rows also carry an `"output"` — the bare
filename in the sentence — which is what trains the
extractor; the 16 list rows are label-only. Sixteen
further names (summit, harbor, meadow, copper, …) are
held out of the file entirely, for the novel-name
measurement below; a test in the demo pins them out, so
a future data edit cannot silently contaminate that
measurement.

The remaining 178 rows are a fourth class, `chat`:
label-only small talk — greetings, thanks, farewells,
feelings, weather questions, joke requests, questions
about the brain itself — written fresh for this file
(cousins of the chat corpus's lines, not copies of
them). They exist because of what the assistant
chapter below measured: a classifier that has only
ever seen file sentences reads "how are you" as a
delete request, because a greeting has nowhere else to
land. The chat rows give small talk a landing place of
its own. Being label-only, they are skipped by the
extractor's training entirely — the seq2seq sees
exactly the rows it saw before — and only the
classifier learns from them. A hygiene test pins their
presence and their label-only shape, like the held-out
names above.

Train with the BPE tokenizer — why is the measured
story below:

```sh
go run ./cmd/train -data examples/filecommands.jsonl -out filebrain.gob -epochs 300 -layers char,copy -augment 0.5 -tokenizer bpe -ckpt-every 10
go run ./examples/filecmd -model filebrain.gob
```

```
> make me a brand new file named budget.txt
intent: create_file (1.00), file: budget.txt
created demo-files/budget.txt
> please remove budget.txt
intent: delete_file (1.00), file: budget.txt
delete budget.txt? [y/n] y
deleted demo-files/budget.txt
```

(Both exchanges are from the measured run below.)

The safety rules are the host's, on purpose: the
extractor's answer is accepted only as a single plain
filename (no separators, no `..`, checked again against
the sandbox folder); `delete_file` needs a confidence of
at least 0.80 *and* a typed yes at a prompt that names
the file; a line with no confident meaning does nothing
and says so.

**Measured, not promised.** Three measurements, all
against brains trained on the file above. The intent
classifier and the extractor are scored separately,
because they are not equally strong — and which
tokenizer the extractor trains under turns out to
matter more than everything else combined.

*Set P — new phrasings, trained names.* 20 lines whose
sentence shapes are absent from training, naming files
that are in it:

| tokenizer | intent right | extractor right | end-to-end right |
| --- | --- | --- | --- |
| word | 20/20 | 2/20 | 2 (15 wrong, 3 gate-abstained) |
| bpe | 20/20 | **20/20** | 17 (0 wrong, 3 gate-abstained) |

The three BPE abstentions are deletes where the
extractor had the right name ("i want movies.txt gone"
→ `movies.txt`) but the classifier's confidence, 0.68
to 0.72, was under the demo's 0.80 delete bar — the
gate doing its job, counted honestly against the
score. Under the word tokenizer the same lines come
back with a plausible wrong name: "would you make a
file called budget.txt for me" → `house.md`; "please
save this as report.csv" → `family.md`.

*Set N — trained phrasings, brand-new names.* 20 lines
naming the 16 held-out names, in sentence shapes from
training:

| tokenizer | intent right | extractor right |
| --- | --- | --- |
| word | 20/20 | 0/20 |
| bpe | 20/20 | 0/20 |

Both tokenizers fail this completely, differently. The
word brain cannot see the name at all — it has no token
for it — and guesses a favourite. The BPE brain can see
the name's pieces, spells trained names perfectly, and
still substitutes a trained name of the right shape:
"i need a file called harbor.md" → `hotel.md`;
"please make copper.json" → `project.json`;
"save this as juniper.csv" → `seeds.csv`. The extension
is right in 19 of 20; the stem is a stand-in every
time. BPE removed the vocabulary wall for *reading*
the name; transcribing a stem it never trained on is a
skill neither tokenizer's brain has at this scale.

*The try-these list, end to end.* The demo's original
15 unseen lines, re-run against the expanded training
(the names in them are trained names now — that is the
point of the expansion):

| run | right | wrong | abstained |
| --- | --- | --- | --- |
| original 36-row data, word | 6 | 8 | 1 |
| expanded data, word | 4 | 11 | 0 |
| expanded data, bpe | 13 | 2 | 0 |
| expanded data + chat class, **bpe (shipped)** | **15** | **0** | **0** |

With BPE every create and every delete lands on the
right file — "create shopping.md for me" creates
shopping.md; "i don't need todo.md anymore, delete it"
deletes todo.md at the prompt. The two wrong lines in
the three-class runs were the classifier's known weak
spot: "what is the weather like today" false-fired
`list_files (0.63)` and listed, and "tell me a joke"
false-fired `create_file (0.99)` and created a file
(agenda.csv, in the sandbox). The chat class is the
fix, and in the shipped run both lines are read as
`chat` (0.98 and 0.99) and do nothing — the list runs
clean at 15 of 15. The class generalises past its own
examples: 22 fresh small-talk phrasings, none of them
in the training file ("hey, long time no see", "will
tomorrow be sunny", "tell me the funniest joke you
know", "sorry, i forgot your name", …), land on `chat`
22 of 22, at confidences 0.97–1.00 — none on a file
label, none abstained. And the file side did not pay
for the new class: Set P against the shipped brain is
unchanged at intent 20/20, extraction 20/20,
end-to-end 17 right / 0 wrong / 3 gate-abstained. The expanded word run is
worse than the original — with six names the extractor
could lean on memorized favourites and occasionally be
right; with 240 it has to copy, and under the word
tokenizer it never learns how: on exact training lines
it extracts 4–5 of 20. A recipe search over the word
tokenizer (augment 0.5 vs 0, copy-only vs char+copy)
found every configuration on the same plateau — the
search is part of the record, not a footnote: no knob
in the trainer teaches the word-level slot at 240
names. Under BPE the same task is a spelling task, and
the brain learns it to a training loss of 0.02 (a
different unit than the word runs' 1.63–1.68 — piece tokens
are not word tokens; the comparable numbers are the
extraction scores above).

So "it knows most of them" is now true, with the
boundary in plain sight. Most *names*: the 240 trained
filenames extract perfectly in phrasings never trained,
and the meanings land 20/20 on every set — including
sentences naming files the classifier never saw, since
a bag of words does not care which name is in the
sentence. Most *unseen names*: none. A filename nobody
trained is a name the demo will get wrong while
sounding sure of itself, which is exactly why the
delete gate and the named-file prompt are load-bearing
rather than decorative. A brand-new *meaning* is the
other boundary: it needs its first example rows plus
one binding function in the host — meanings are
declared, not discovered.

## Runnable example: one assistant, both brains

`examples/assistant` loads two brains in one program and
routes every line by meaning. The file brain's intent
classifier reads the line first: when it returns
`create_file`, `delete_file`, or `list_files`, the line
is handled exactly as in `examples/filecmd` — same
extractor, same sanitizer, same sandbox, same delete
gate and confirmation prompt. When the classifier
returns nothing — or returns its fourth class, `chat`,
the label-only small-talk class the file data now
teaches — the line falls through to the chat brain
(trained on `examples/pairs.jsonl`) and is simply
answered. One prompt, two skills:

```sh
go run ./cmd/train -data examples/pairs.jsonl -out brain.gob -epochs 60 -layers copy -augment 0.3 -synonyms 0.3
go run ./cmd/train -data examples/filecommands.jsonl -out filebrain.gob -epochs 300 -layers char,copy -augment 0.5 -tokenizer bpe -ckpt-every 10
go run ./examples/assistant -chat brain.gob -files filebrain.gob
```

The routing rule, in plain words: **a confident file
meaning acts; anything else talks — and a delete the
classifier is not confident enough about is refused as
a command, never quietly re-routed into small talk.**
A confident `chat` reading is the positive form of
"anything else": the classifier recognises the line as
conversation and the assistant lets it fall through,
with no intent line printed and no refusal — the reply
is the whole output, exactly as for a line the
classifier abstains on.

**Measured, not promised.** One scripted session against
brains trained with exactly the two commands above,
mixing chat lines and file lines on purpose. This is
the same eight lines as the session this section used
to report — re-run after the file data learned its
fourth class:

```
assistant: talk to me — or ask me to create, delete, or list files in demo-files/. Type 'quit' to exit.
> how are you
doing great, thanks! how about you?
> what do you like to do
i like simple things: reading, gardening, music, and a warm cup of coffee.
> create a file named jim
intent: create_file (1.00), file: garage.md
created demo-files/garage.md
> create a file called notes.txt
intent: create_file (1.00), file: notes.txt
created demo-files/notes.txt
> what files do i have
intent: list_files (0.83)
demo-files/: garage.md, notes.txt
> get rid of notes.txt
intent: delete_file (1.00), file: notes.txt
delete notes.txt? [y/n] y
deleted demo-files/notes.txt
> tell me a joke
why did the scarecrow win an award? because he was outstanding in his field.
> thank you
you are very welcome. i am happy to help.
```

Score: **8 of the 8 lines routed as intended** — up
from **5 of 8** for the same script before the chat
class existed. The previous run's three misroutes are
the comparison that matters:

- *"how are you" and "thank you"* were read as
  `delete_file` (0.64 / 0.69) and refused at the gate —
  no file touched, but the reply was stolen. The file
  classifier had only ever seen file sentences, so a
  greeting had nowhere to land but a file meaning.
  Now both lines are read as `chat` (1.00 and 0.99,
  verified against the classifier directly) and reach
  the chat brain, which answers them.
- *"tell me a joke"* false-fired `create_file (0.99)`
  and created archive.md in the sandbox — a sentence
  meant as conversation did something to disk. Now it
  is read as `chat (0.99)`, falls through, and the
  chat brain answers with an actual joke. Nothing is
  created.

The file arc is unchanged: create jim's line, create
notes.txt, list, confirmed delete — all land, and the
sandbox ended holding exactly garage.md, with
notes.txt created and deleted on request. And the
standing boundary is still standing inside that arc:
"create a file named jim" *acts* — `create_file
(1.00)`, and a file appeared — but jim is not one of
the 240 trained filenames, so the extractor did its
Set N substitution and created garage.md. The meaning
was understood completely; the name was beyond it,
exactly as the previous chapter measured. The chat
class fixes where small talk lands; it does not teach
the extractor new names.

Two things changed between the sessions, and the
score belongs to both: the file data grew its `chat`
class (the routing fix — the greetings and the joke
are its doing), and the chat brain is now the
2,888-pair corpus rather than the 451-pair one (the
reply quality — "i like simple things…" instead of
the old splice is its doing). The routing does not
care which chat brain is loaded; the halves still
improve independently.

What did not change: the caveat from the filecmd
chapter stays load-bearing in principle. A bag-of-words
classifier can still, on some line nobody tested,
false-fire — and in a combined program a false-fire
either steals a reply or touches the disk. This
session contains no such line, and the fresh
small-talk set in the filecmd chapter (22 of 22 on
`chat`) says the common cases are covered; neither is
a proof about every line. The gates remain what bound
the damage when a reading is wrong — no file was
wrongly deleted in either session, and nothing has
ever left the sandbox.

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

### Step E — growing the chat data (density for readability)

Steps A–D were scored by topic-carrying, and a reply can
carry its topic and still read like two canned sentences
spliced together — because at 451 pairs, splicing is what
the data taught. This step attacks readability with data
density: `examples/pairs.jsonl` grew 451 → 2,643 pairs,
the originals kept verbatim, the new pairs built by
families (greetings, feelings, daily routine, work and
school, family, food, weather, 27 new topics alongside the
old ones) with several distinct replies per input family
instead of one canned line. Audit: no new input reproduces
a held-out evaluation line, no input carries more than
three distinct replies.

Recipe search on the new corpus (copy layer, word
tokenizer): 60 epochs with mutation + synonyms 0.3 beat
60 epochs without augmentation (concept set 15/16 vs
12/16 greedy) and a 180-epoch run that early-stopped at
120 (14/16, and no better anywhere else). The winner is
the quickstart command above. The control arm is the old
451 pairs under that same winning recipe, so the table
below isolates the data:

| measure (held-out, typos off) | 451 pairs | 2,643 pairs |
|---|---|---|
| paraphrase set, greedy / checked | 22/36, 25/36 | 21/36, 25/36 |
| concept set, greedy / checked | 11/16, 14/16 | 15/16, 16/16 |
| fluency probe, coherent / partial / garbled | 4 / 8 / 12 | 16 / 3 / 5 |

The fluency probe is new: 24 everyday lines, replies
scored by hand as coherent (grammatical, on-topic),
partial (readable but broken or deflected), or garbled.
That is the number that moved. Verbatim, checked replies:

- "i am feeling a bit tired today" — 451: "a strong
  opinion! am, catching light., though a quiet of its
  own fans." → 2,643: "get some rest when you can. a
  tired body needs a quiet evening."
- "what do you like to do" — still bad under the shipped
  recipe: "i do like with a tiny about small things. one
  that is of my kind, thank you." The 180-epoch arm
  answered this one cleanly ("i do like simple things:
  reading, gardening, music, and a warm cup of coffee.")
  and lost on the aggregate table; individual lines move
  between runs, and the aggregate is what ships.

The honest costs: on the paraphrase set's greedy pass the
old corpus edges the new (22 vs 21 — its topics are
exactly that set's topics; the wider corpus dilutes them,
and checked sampling ties at 25). And 5 of 24 probe lines
still garble. Density is a lever, not a cure.

## Intent understanding in the chat brain

The chat brain's job is *knowing*, not doing: acting on a
request lives in the runnable examples (filecmd, assistant).
This step teaches the chat brain itself to recognize what a
line means — a request's intent, an ordinary line's topic —
by concepts, so unseen phrasings resolve too, and to let
that recognition show in its reply.

Two data moves, both inside `examples/pairs.jsonl`:

- **Intent labels woven into the chat data.** Ten request
  families (create_file, delete_file, create_folder,
  build_server, set_reminder, set_timer, play_music,
  send_message, search_web, book_travel) × 15 phrasings,
  every row carrying *both* an `"intent"` and a reply that
  reflects the recognized meaning — it names what was
  asked and its object ("A file called notes — that's
  asking me to create a file. I follow you."). Alongside
  them, eight everyday topics (greeting, farewell,
  feelings, weather_talk, hobby_talk, food_talk, thanks,
  about_brain) × 12 labeled rows, so the classifier learns
  what *talk* looks like too: "what is the user talking
  about" now has an answer for ordinary lines, not just
  commands.
- **The deflection rows rewritten.** The corpus growth had
  taught 17 request lines to deflect ("I cannot create
  files — I am only a chat brain."). Those outputs are now
  recognition-style, same inputs, so the two styles do not
  fight; 10 of the 17 also gained the intent label they
  belong to.

The corpus is now 2,888 pairs, 255 of them labeled, 18
labels. Training is the quickstart recipe — the trainer
builds the classifier from the labeled rows and writes
`brain.gob.intents.json` as before — and `cmd/ask` shows
both halves:

```sh
go run ./cmd/ask -model brain.gob -intents
```

```
> create a file named jim
a file named — that's asking me to create a file. i follow you.
intent: create_file (0.99)
> create a webserver in main.go
in a webserver — i hear you: a server to build. i follow you.
intent: build_server (0.99)
```

**Measured — set U1: unseen phrasings of trained families
and topics (30 lines, none in training).** The classifier's
top guess was the right label on **30/30**; at the shipped
0.5 threshold that becomes **28 correct, 2 abstentions,
0 wrong** (the two abstentions had the right label on top,
at 0.43 and 0.48). Replies, scored REFLECTS (shows the
right intent/topic) / VAGUE (generic, not wrong) / MISSES
(wrong intent asserted, deflection, or garble):
**19 / 3 / 8**.

| line | classifier | reply | verdict |
| --- | --- | --- | --- |
| create a file named jim | create_file (0.99) | a file named — that's asking me to create a file. i follow you. | REFLECTS (the name itself is out of vocabulary and drops) |
| i need a new file for my recipes | create_file (0.98) | a new recipes with a tomorrow, fresh is a fine again. | MISSES |
| get rid of the file named draft | delete_file (0.99) | you want a new file, lee. i understand — that's a create-a-file request. | MISSES (reply asserts the wrong intent) |
| remove my homework file | delete_file (0.94) | got it — delete a file called lee. i can't do that part here, but i know what you're asking. | REFLECTS (object substituted: lee) |
| delete the letter file please | delete_file (1.00) | letter — i hear you: a file to delete. i follow you. | REFLECTS |
| create a folder named jim | create_folder (0.99) | a folder, lee. i understand — that's a create-a-folder request. | REFLECTS (object substituted) |
| make a new folder for my photos | create_folder (0.99) | a hello! a new folder, lee. i understand — that is a fine way. | VAGUE (spliced) |
| create a webserver in main.go | build_server (0.99) | in a webserver — i hear you: a server to build. i follow you. | REFLECTS |
| can you build me a small web server in go | build_server (1.00) | got it — search go to the game score. i can't do that part here, but i know what you're asking. | MISSES (a search reply spliced in) |
| remind me to call the doctor tomorrow | set_reminder (1.00) | call the tomorrow — you're asking me to time something. i hear you. | MISSES (the timer frame, not reminder) |
| set a reminder for my meeting at noon | set_reminder (0.94) | you want a reminder: my essay. i understand — that's a set-a-reminder request. | REFLECTS (object substituted) |
| set a timer for twenty minutes | set_timer (1.00) | twenty minutes — i hear you: a timer to start. i follow you. | REFLECTS |
| could you set a timer for the pasta | set_timer (0.97) | got it — a timer for the weekend. i can't do that part here, but i know what you're asking. | REFLECTS (object substituted) |
| put on some jazz for me | play_music (0.98) | got it — a quiet tune. i can't do that part here, but i know what you're asking. | REFLECTS |
| play my favorite song | play_music (0.99) | a good song in every slice — that is the heart of music. | VAGUE (music talk, no intent shown) |
| send a text to kim saying i am late | send_message (0.89) | text late — i hear you: something to find online. i follow you. | MISSES (reply asserts searching) |
| write an email to my boss | send_message (1.00) | that's a request to send a message: alex. i understand. | REFLECTS (object substituted) |
| search the web for train times | search_web (0.91) | kind train — that's asking me to play music. i follow you. | MISSES (reply asserts play-music) |
| look up how to fix a leaky faucet | search_web (0.91) | you want me to look something up: how to fix a bike. i understand — that's a web-search request. | REFLECTS (object near-miss: bike) |
| book me a flight to denver | book_travel (0.99) | you want to | MISSES (fragment) |
| find me a hotel in paris for friday | abstained (best: book_travel 0.43) | you want to a hotel in your keep for something small, and i will be kind of strong, teaches patience. | MISSES |
| i want to book a train ticket to the city | book_travel (0.99) | you want me to book a train ticket to boston. i understand — that's a book-travel request. | REFLECTS (object substituted: boston) |
| good morning, how is everything | greeting (0.85) | all good here. how are things with you? | REFLECTS |
| bye for now, talk soon | farewell (0.98) | see you later! have a happy day. | REFLECTS |
| i am feeling really happy today | feelings (0.99) | that is wonderful. a happy day is a good day. | REFLECTS |
| it looks like rain out there | weather_talk (0.56) | understood. rain, water — that is the happy thing skill. | VAGUE |
| do you like playing tennis | hobby_talk (0.88) | very much. talking about tennis always makes my day better. | REFLECTS |
| i had pizza for lunch | food_talk (0.84) | pizza is a party on a plate. a fine dinner. | REFLECTS |
| thanks a lot for your help | thanks (1.00) | you are welcome! talking with you is a pleasure. | REFLECTS |
| who built you | abstained (best: about_brain 0.48) | whoever wrote my example pairs and ran the training. i owe them my words. | REFLECTS |

Read the two halves separately, as in the filecmd chapter:
the **classifier** is the reliable half (it never once
picked a wrong label at threshold). The **reply** brain
shows the right intent in 19 of 30, but its object slot
behaves exactly like the extractor's did — of the request
replies that reflected the intent, the object was named
right only a few times; trained names get substituted
(lee, boston, my essay) and an out-of-vocabulary name
(jim) drops out entirely. The meaning is recognized far
more reliably than the reply can spell it.

**Measured — set U2: three request families that appear
nowhere in training** (order_food, translate_text,
find_directions; 5 lines each). There is no correct label;
the classifier outcome is scored NEIGHBOURHOOD (it abstains,
or lands on a request-family label) vs WRONG (a confident
chat-topic or unrelated label), and replies with the same
rubric as U1.

| line (family) | classifier | reply | verdicts |
| --- | --- | --- | --- |
| order a pizza for dinner (order_food) | abstained (best: set_timer 0.13) | a pizza is a party on a plate. everyone takes a slice and nobody is sad. | NEIGHBOURHOOD / VAGUE |
| can you get me a burger from the place nearby (order_food) | abstained (best: set_reminder 0.46) | got it — a get the heart, but i understand — that's a create-a-folder request. | NEIGHBOURHOOD / MISSES |
| i want some sushi delivered (order_food) | play_music (0.65) | a quiet day is a gift. tea, a book, and nowhere to hurry is its day. | NEIGHBOURHOOD (a request label, the wrong family) / MISSES |
| order two coffees to go (order_food) | abstained (best: farewell 0.37) | two coffee is smells like morning. a warm meal you earned it. | NEIGHBOURHOOD / MISSES |
| get a sandwich delivered to my office (order_food) | send_message (0.50) | neither wins for long. yes, i like sandwich a lot. it is one of my favorite topics. | NEIGHBOURHOOD / VAGUE |
| translate hello into french (translate_text) | greeting (0.63) | hello! how are you today? | WRONG / MISSES |
| how do you say thank you in spanish (translate_text) | about_brain (0.77) | you are very welcome. talking with you is a pleasure. | WRONG / MISSES |
| can you translate this sentence into german (translate_text) | about_brain (0.53) | got it — a sentence a day i can't do that part here, but i know what you're asking. | WRONG / REFLECTS (the recognition frame fired) |
| what is the french word for cat (translate_text) | weather_talk (0.91) | a quiet battle of plans. that is the good part, in short. | WRONG / MISSES |
| translate good night into italian (translate_text) | abstained (best: greeting 0.23) | nice night is big open eyes. rest is a fine word for the days. | NEIGHBOURHOOD / MISSES |
| how do i get to the airport (find_directions) | abstained (best: search_web 0.36) | good for you. airport grows from small steps, taken often. | NEIGHBOURHOOD / MISSES |
| give me directions to the library (find_directions) | abstained (best: search_web 0.45) | you have done the right word some days. tell me more. | NEIGHBOURHOOD / MISSES |
| what is the way to the nearest station (find_directions) | abstained (best: search_web 0.50) | the nearest | NEIGHBOURHOOD / MISSES |
| directions to the park please (find_directions) | abstained (best: search_web 0.43) | to please you are welcome! talking with you start with a song you know by heart. | NEIGHBOURHOOD / MISSES |
| how far is it to the city center (find_directions) | greeting (0.53) | then it is. far is a friends, in the water evening. | WRONG / MISSES |

Classifier: **10/15 NEIGHBOURHOOD, 5/15 WRONG**. The
neighbourhood is real where a trained cousin exists:
directions lines put search_web on top in 4 of 5 (and
abstained rather than commit), food lines abstained or
landed on request labels. It fails where the new family's
words live inside chat topics — translation lines are
built from *hello*, *thank you*, *good night*, and the
classifier confidently reads them as greeting, thanks,
and small talk. Replies: **1 REFLECTS / 2 VAGUE / 12
MISSES** — the reply brain does not transfer to unseen
families at all; it answers from topic echoes, and only
once did a recognition frame fire.

**Regressions, checked against the corpus-growth
numbers.** The fluency probe improved on this run —
coherent / partial / garbled **16/3/5 → 19/2/3** — with
the usual per-line churn in both directions ("what do
you like to do" answers cleanly this run; "can you help
me learn chess" garbles). The concept set held: greedy
**15/16** (tied), checked **15/16** (was 16/16 — the one
dip is "are cats animals", answered without the word
*cats*). The paraphrase set dipped under the published
typos-off protocol: greedy / checked **21/36, 25/36 →
19/36, 22/36**, with all six typo'd lines failing — under
that protocol the typo fixer is off; with it on (the
`cmd/ask` default) the same brain scores 25/36 and 28/36
on the set. Per-line outcomes move between training runs;
both instruments are reported, not picked between.

**The boundary, plainly:** a new kind of intent with no
trained relative gets a neighbourhood guess, not
knowledge. The classifier's neighbourhood is worth
something — it knows a directions question lives near
search, and it abstains when it has no cousin to point
at. The reply's neighbourhood is worth much less, and a
family whose vocabulary overlaps ordinary talk
(translation) fools both halves. Recognition of *trained*
meanings from *unseen* phrasings is the thing that works
here, measured above; recognition of *new kinds* of
meaning is the thing that does not, also measured above.

## Classifier: pieces and kinds

The chapter above ends at a precise boundary: the
classifier knows trained meanings in unseen phrasings,
and guesses neighbourhoods for new ones. Two mechanisms
were built to push on exactly that — **kinds**, a
hierarchy the classifier can fall back on, and
**pieces**, subword features so unseen words register at
all. Both were measured against the classifier as it
was, on suites built for the purpose. One shipped as the
default. The other did not. This section is the record
of both, because the numbers are the point.

### Kinds (shipped)

A row may carry a `"kind"` next to its `"intent"` — the
broad family its meaning belongs to:

```json
{"input": "create a file named jim", "intent": "create_file", "kind": "make"}
```

The taxonomy applied to both shipped data files:

| kind | labels in pairs.jsonl | labels in filecommands.jsonl |
| --- | --- | --- |
| make | create_file, create_folder, build_server, grocery_list | create_file |
| remove | delete_file | delete_file |
| show | — | list_files |
| schedule | set_reminder, set_timer | — |
| play | play_music | — |
| communicate | send_message | — |
| find | search_web, convert_units, define_word, spell_word, summarize_text, find_recipe, find_place | — |
| travel | book_travel, reserve_table | — |
| talk | greeting, farewell, feelings, weather_talk, hobby_talk, food_talk, thanks, about_brain | chat |

The trainer derives the label→kind map from the data
and validates it: one label claiming two kinds is a
training error that names the label and both kinds.
Alongside the label softmax it trains a second softmax
over the kinds, on the same features. Kind training is
**class-balanced** — every kind contributes the same
total gradient weight, however many rows it has —
because the kind model's whole job is the weak-evidence
case, and there the biggest kind's share of the rows
(the prior) is exactly the wrong answer. The label
classifier keeps natural frequencies; telling meanings
apart is its job, and there the prior is evidence too.

`Predict` is unchanged. `PredictKind` is the new answer,
`KindOf` maps a label to its kind, and `Describe` — what
`cmd/ask -intents` prints — appends the kind when the
brain has one, *including when the label abstains*: the
neighbourhood is what an unseen meaning can still offer.
Brains trained on kindless data behave and display
exactly as before. Live, on a brain trained with the
shipped recipe:

```
> create a file named jim
that's a request to create a file: work. i understand.
intent: create_file (0.99) · kind: make (0.99)
> create a webserver in main.go
in a webserver — i hear you: a server to build. i follow you.
intent: build_server (0.99) · kind: make (1.00)
```

### Pieces (built, measured, not the default)

The second mechanism extends the features: alongside
each word, its subword pieces — character n-grams
(length 3–5) in one scheme, BPE pieces (the repo's `bpe`
package, merges trained on the intent rows' words) in
the other. An unseen word then registers through the
pieces it shares with trained words: "creat" is mostly
"create". Every sidecar now records its feature scheme,
and a sidecar written before schemes existed is
**rejected with a retrain-to-upgrade error** rather
than scored under a featurizer its weights never knew.

### The suites

Three suites were declared before any arm ran, and the
shipped fences were re-run for every arm. **HF**: six
request families that appear nowhere in training
(order_food, translate_text, find_directions, fix_code,
play_game, tell_story) × 6 lines, scored by the U2
neighbourhood rubric for labels, and for kinds against
acceptable-kind sets declared in the suite file before
scoring (order_food {communicate, make}, translate_text
{communicate, find}, find_directions {find}, fix_code
{make}, play_game {play}, tell_story {talk}). **WC**:
12 word-collision traps ("say hello to mom for me" must
be send_message, "hello mom" greeting; "make me a
sandwich" must not become create_file). **TY**: 10
typo'd intent lines ("creat a file called notes.txt",
"remid me to water the plants", ...). Arms: the word
classifier as it was, words+n-grams, words+BPE — same
rows, same training, only the features differ.

Labels, at the shipped 0.5 threshold unless marked raw
(the argmax, right or wrong):

| measure | words | + n-grams | + BPE |
| --- | --- | --- | --- |
| TY typo lines | 9/10 (raw 9/10) | 9/10 (raw **10/10**) | 8/10 (raw 9/10) |
| WC collision traps | **9/12** | 8/12 | 7/12 |
| HF neighbourhood (labels) | 30/36 | 27/36 | **31/36** |
| fence: U1 raw / at threshold | **30/30, 28/2/0** | 29/30, 28/2/0 | 29/30, 27/3/0 |
| fence: file Set P intent | **20/20** | 17/20 | 19/20 |
| fence: file small-talk → chat | **22/22** | 21/22 | 22/22 |

Kinds (the shipped word arm; the piece arms score the
same totals within a point): on HF, **right 9/36, wrong
22, abstained 5** — play_game 5/6 (a game is *play*, the
cousin of play_music), fix_code 3/6, tell_story 1/6, and
0 for order_food, translate_text, and find_directions.
On the WC traps' declared kinds, **7/9** — including
"make me a sandwich", whose *label* abstains while its
kind lands on make (0.77), and "make a file for my
sandwich recipes" at make (1.00); the two misses are
kinds that abstain ("say hello to mom for me",
"delete my account"), not kinds that err.

The wrong kinds are mostly one word: **talk**. Balancing
took the prior's thumb off the scale — talk's winning
confidences fell from the 0.74–1.00 range to 0.54–0.99 —
but an unseen line's vocabulary *is* everyday vocabulary,
and everyday vocabulary is what talk is made of. The
features lean where the words live.

### The verdict on pieces

Each piece scheme won exactly one measure — n-grams the
typo lines' raw top label, BPE the held-out
neighbourhood by a single line — and lost the traps and
the fences to do it. The mechanism is visible in one
line: "say hello to mom for me". The word classifier
abstains; its best guess, greeting, sits at 0.29. Stack
n-gram pieces on and the same lean comes back greeting
(0.91) — said with confidence. Pieces multiply the
evidence, and multiplied evidence multiplies whatever
lean was already there, right or wrong: "delete my
account" goes from an abstention (best guess delete_file
at 0.45) to delete_file at 0.78 under n-grams and 0.53
under BPE. On the file demo's own lines the same
inflation costs real commands ("i no longer need
letter.md" → chat at 0.62 under n-grams; an abstention
at 0.50 under BPE, where the word classifier says
delete_file).

So `Train` keeps whole words, and both piece schemes
remain available through `TrainWith` (`Options{Scheme:
...}`) for data whose own measurement disagrees. One
part of the pieces work shipped unconditionally: the
trainer itself. Piece feature spaces are thousands wide,
and the old dense SGD loop scaled with the whole space;
it now updates sparsely, with the L2 decay applied
lazily per label — the same training, up to float
rounding, at a cost proportional to what a line
contains. The word classifier's fences reproduce
history exactly under it (U1 30/30 raw, 28/2/0; Set P
20/20; small-talk 22/22), and intent training that took
5.8s on the chat rows and 10.4s on the file rows now
takes well under a second and about a second.

### The updated boundary

What can an unseen line expect now? An unseen *phrasing*
of a trained meaning: the label, usually at high
confidence — unchanged, still the strong half. An unseen
*meaning* near trained ones: a kind, usually right when
the family has a close trained cousin (a game is play;
fixing code is make about half the time), and sometimes
a neighbouring label. An unseen meaning far from home:
the kind defaults to talk or abstains — a guess about
the neighbourhood, offered as a guess, and the label
stays home. Certainty, never: the kind is reported with
its confidence, and below the threshold it is not
reported at all. That is more than this classifier
could say before — previously a new meaning got an
abstention or a wrong label, and nothing in between —
and it is less than understanding a new kind of request,
which no measurement here shows.

## Collision-zone data

The kinds chapter ended with a diagnosis and a named
cure. The diagnosis: an unseen family's words *are*
everyday words, and everyday words belong to `talk`, so
stranger meanings collapsed into talk — kinds went
9 right / 22 wrong / 5 abstained on the held-out
families, with order_food, translate_text, and
find_directions at zero. The cure: training mass in the
collision zones, not another mechanism. This section is
that build. No mechanism changed — the classifier is
the shipped word classifier, the schemes and thresholds
untouched. Only the data grew, deliberately, in the
zones *around* the exam families. The exam stayed
frozen: none of the six held-out families was trained,
in any phrasing.

### What was added

156 rows, all intent-labeled and kinded
(`examples/pairs.jsonl`: 2,888 → 3,044 pairs; labeled
rows 255 → 411; labels 18 → 26):

- **Language-transform cluster** — translate_text's
  cousins, all kind `find`, 12 phrasings each:
  `convert_units` ("convert five miles into
  kilometres"), `define_word` ("what does serendipity
  mean"), `spell_word` ("how do you spell necessary"),
  `summarize_text` ("summarize this paragraph for
  me"). Conversions, definitions, and spellings are
  lookups, and a lookup is what `find` already means;
  the cluster's mass lands in one of translate's two
  acceptable kinds on purpose.
- **Food-request cluster** — order_food's cousins:
  `find_recipe` ("find me a recipe for tomato soup",
  kind `find`), `reserve_table` ("reserve a table for
  four on friday", kind `travel` — a reservation is a
  booking, and the collision suite already declares
  table-booking as travel), `grocery_list` ("add milk
  to my shopping list", kind `make`).
- **Location cluster** — find_directions' cousin:
  `find_place` ("where is the nearest pharmacy", "is
  there a park near here", kind `find`). Location
  *questions*, not directions requests — the point is
  to teach that where/nearest shapes are `find`, not
  talk.
- **Statement-vs-request contrasts** — 60 past-tense
  reports and opinions on the *same topic words* as
  the request families, labeled as talk: "i ordered a
  pizza last night" (food_talk), "she translated the
  letter for me" (hobby_talk), "the airport was busy
  this morning" (hobby_talk), "we played chess
  yesterday" (hobby_talk). Same words, report shape →
  talk; the request shape is what makes a request.

Audit, as always: no duplicate inputs, none of the
new lines already in the corpus, and no exam line
(HF/WC/TY) verbatim or near it — the highest token
overlap between any new line and any exam line is
0.44 (Jaccard), and the closest pairs are the intended
shared frames with different content ("what does
serendipity mean" against the exam's "what does
gracias mean in english", 0.43).

### Held-out kinds, before and after

The same suites, the same scoring, the shipped word
classifier retrained on the enriched rows:

| HF family | kinds before | kinds after |
| --- | --- | --- |
| order_food | 0 right / 5 wrong / 1 abstain | **0** / 5 / 1 |
| translate_text | 0 / 6 / 0 | **5** / 1 / 0 |
| find_directions | 0 / 5 / 1 | **3** / 2 / 1 |
| fix_code | 3 / 3 / 0 | **1** / 3 / 2 |
| play_game | 5 / 0 / 1 | 5 / 0 / 1 |
| tell_story | 1 / 3 / 2 | 2 / 1 / 3 |
| **total** | **9 / 22 / 5** | **16 / 12 / 8** |

What moved, family by family:

- **translate_text: 0 → 5.** The language cluster
  worked almost exactly as aimed. "what does gracias
  mean in english" is now answered by the cousin
  family itself — `define_word (0.87) · kind: find
  (0.96)` — and "translate this word into spanish for
  me" lands on `summarize_text · find (0.99)`. The
  one holdout is "translate good morning into german",
  still `talk (0.98)`: *good morning* is greeting's
  home turf, and no cousin mass was aimed at it. The
  kinds chapter's villain line, "translate hello into
  french", no longer draws the confident greeting
  (0.63) it used to: live on the full brain below, the
  label abstains and the kind is still `talk`, but at
  0.84, down from 0.96 — *hello* still pulls it home,
  just less surely.
- **find_directions: 0 → 3.** "how far is the beach
  from here" → `find (0.97)`; "how do i get downtown
  from here" → `find (0.77)`. On "what is the fastest
  route to the airport" the two heads now disagree:
  the label still says weather_talk (0.76), but the
  kind says `find (0.84)` — the kind head sees the
  route words even when the label head doesn't.
  Still wrong: "which way is the train station"
  (`talk`), "show me the way to the market"
  (`schedule`, as before).
- **order_food: still 0, printed at 0.** Its wrongs
  moved — three still land on talk, but two now land
  on `find` (0.52, 0.76), pulled by the recipe mass —
  and `find` is not in order_food's acceptable set
  {communicate, make}. The food-request mass taught
  *finding* food, not *ordering* it, and ordering is
  the family that stayed unclaimed.
- **fix_code: 3 → 1 — the regression, stated
  plainly.** The find cluster is a strong attractor
  for request shapes, and fix_code got no mass of its
  own: "can you debug this code for me" now reads
  `summarize_text (0.85) · kind: find (0.95)`, "there
  is a bug in the server code" flipped from kind-right
  to `find`, and "my code has a bug in it" dropped to
  an abstention. Mass in one zone pulls on the
  neighbours that got none.
- **tell_story: 1 → 2**, mostly by wrongs becoming
  abstentions; **play_game** unchanged at 5. Across
  the suite, wrongs fell 22 → 12 while abstentions
  rose 5 → 8: part of the gain is the classifier
  saying less, more honestly.

The label-level scores need one footnote. HF label
neighbourhood went 30/36 → 29/36, but three of the
seven misses are landings on the *new* labels
(define_word, summarize_text, find_recipe) whose
kinds are right — the neighbourhood lists were
declared before those labels existed and cannot
credit them. WC labels went 9/12 → 7/12 for a related
pair of reasons: "book a table for two tonight" now
gets its true new meaning, `reserve_table (0.93)`,
which the frozen ok-list predates and scores as a
miss; "hello mom" now abstains (raw top still
greeting); and one genuine new error — "set the table
for dinner" reads as `reserve_table (0.64)`, a dinner
statement claimed by the booking family. Against
that, "look up at the stars tonight" improved from a
confident search_web to an abstention. WC *kinds*
held at 7/9 — the same two misses, both abstentions
("say hello to mom for me", "delete my account"). TY
held at 9/10, same single miss ("make a foldr for my
photos").

### Fences

The U1 fence is untouched: 30/30 raw, 28 correct /
2 abstained / 0 wrong at threshold, exactly the
baseline. The file fences are untouched (Set P intent
20/20, small-talk 22/22 — filecommands.jsonl did not
change). A full chat brain trained on the enriched
corpus with the shipped recipe (the run was killed
twice by environment restarts and finished through
checkpoint-resume, epoch 60, best loss 0.0875) reads
the four live lines as:

```
> translate hello into french
hello! it is good to talk with you.
intent: (none confident enough) · kind: talk (0.84)
> order me a pizza please
pizza is a party on a plate. everyone takes a slice and nobody is sad.
intent: (none confident enough) · kind: (none confident enough)
> how do i get to the airport
small: one pot, one plant, water and sun. airport rewards a patient beginner.
intent: (none confident enough) · kind: find (0.72)
> create a file named jim
a file — that's asking me to create a file. i follow you.
intent: create_file (0.98) · kind: make (0.99)
```

The airport line is the two halves in one frame: the
kind head claims it for `find` while the reply brain
garbles it into a gardening splice — understanding
and fluency remain separate scores. The fluency
probe, same protocol as the intent chapter, scores
**17 coherent / 3 partial / 4 garbled** against the
previous brain's 19/2/3: churn in both directions
(the rain line and the goodnight line came clean this
run; the smile line and the weekend line garbled),
not a collapse.

### The revised boundary

A stranger family can now expect more than it could —
unevenly. Requests that *transform or look up
language* land in `find` most of the time, and
location questions land there half the time, because
those zones now have mass. Ordering food still
defaults to talk or find and stays unclaimed: no
trained kind means *commissioning* something, and
mass about finding food does not teach it. And the
gravity well moved rather than vanished — `find` is
the new strong attractor, strong enough to pull two
fix_code lines out of `make`. Collision-zone mass is
a lever with a direction: it buys the zones you feed,
partly at the price of the neighbours you don't.

## A concept curriculum for code questions

The collision chapter fed the zones around *request*
families. This chapter feeds a different gap, found by
hand-testing: ask the chat brain "what is go" or "what
is a function" and it had nothing — the corpus held
everyday topics and request families, and not one Go
concept. The Go knowledge the author remembered lived
in the parked Gollemer project. So this build teaches
the chat brain the *concepts* of Go, organised the way
the learning resources organise them.

**The boundary, stated plainly:** this brain does not
fix code and does not write code. Nothing in the new
data teaches repairing or authoring programs — no
corrected code, no "here is the fixed version". What
it learns is to *recognise* a concept or a code
question and say what it means. `fix_code` stays an
untrained exam family precisely because acting is out
of scope; its held-out score below measures only
whether the guess lands in the code neighbourhood.

### Concepts, kinds, phrases, words

The data is built in a four-layer hierarchy, because
that is how the guessing works:

| layer | for the slice concept |
|---|---|
| **kind** | `find` — the category: information work |
| **concept / intent** | `explain_code:slices` — shipped as the label `explain_slice` |
| **phrases** | "how do slices work in go", "what does append do in go", "how is a slice different from an array", "what is a nil slice in go", "what is a slice header in go", … (8 in all) |
| **words** | slice, append, grow, array, length, capacity, header, backing, nil, make — the vocabulary spread across the phrases and replies, not just the concept's name |

Every one of the 49 new labels carries all four
layers; an audit found seven concepts at the
two-phrase minimum (exported names, JSON, recover,
sorting, templates, variadic functions, waitgroups)
and they were topped up to four phrases each before
anything was measured.

### Sources

The resource index is `advocates.json` — the author's
own Go learning links (54 topics, ~460 links), the
same file Gollemer's guide seat used as its link
index. One source inside it is primary:

- **Go by Example** (gobyexample.com, by Mark
  McGranaghan and Eli Bendersky, CC BY 3.0) — fetched
  and read, ~30 topic pages. Each topic became one
  concept meaning. Answers are short paraphrases of
  the page's facts in this brain's voice — never
  copied at length.
- Also fetched and read: **Effective Go** (names and
  exported names, formatting), **YourBasic Go**
  (conversions, data races, files), the **golangbot**
  tutorial series (compiled, statically typed,
  garbage-collected), Russ Cox's **Go Data Structures:
  Interfaces** (implicit satisfaction; an interface
  value is a type-and-data pair), the **Go Blog** on
  slices (the slice header), and **Go Web Examples**
  (HTTP servers, handlers, templates).
- Topic-only — named here honestly because they could
  not be read as text (a JavaScript tour app, video
  courses, paywalls, playground snippets): A Tour of
  Go and Learn Go with Tests seeded *which* concepts
  belong in a beginner curriculum; no content is
  claimed from them.
- Gollemer's own files were gap-fillers only: the
  `go_expansion_*` files supplied the mutex, context,
  JSON, and testing phrasings (10 rows adapted); the
  codeintent files (8,044 rows of repo-specific file
  Q&A) were skipped as a source, and the 36
  fix/debug-phrased rows found in them were excluded
  by rule, not used. Skipped in the index as not
  concept-bearing: job boards, ssh tools, surveys,
  frontend frameworks, Docker, databases.

239 rows in all (`examples/pairs.jsonl`: 3,044 →
3,283 pairs; labeled rows 411 → 650; labels 26 → 75):
129 grounded in Go by Example, 76 more from the other
advocates-indexed sources, 10 adapted from Gollemer's
files, and 24 hand-written code-question cousins —
`explain_error` ("what does declared and not used
mean") and `read_code` ("what does this code do"),
understanding only: their replies explain what an
error *means* and what code *does*, and never repair
anything. Replies throughout are recognition first,
concept-first: name the concept, then the one-line
gist ("A slice — a flexible view over an array that
grows with append.").

### The probe (headline)

Declared before scoring: 25 held-out lines — 15 unseen
phrasings of trained concepts, 6 code-question lines,
and 4 lines in shapes never trained for any concept
(yes/no questions, example requests, "in simple
terms"). Kind first: `find` was guessed at threshold
on **25 of 25**. The concept *label* is the harder
test — each concept carries only a few training rows,
so confidence is diluted across ~75 labels:

| line | label at threshold | raw top |
|---|---|---|
| what is a goroutine | — abstained (0.49) | explain_goroutine ✓ |
| what does a package do in go | explain_package (0.58) | ✓ |
| what is an interface in go | explain_interface (0.92) | ✓ |
| explain what a slice is | explain_slice (0.68) | ✓ |
| what is a pointer in go | — abstained (0.40) | explain_pointer ✓ |
| how are errors handled in go | — abstained (0.16) | explain_testing ✗ |
| what is a struct in go | explain_struct (0.66) | ✓ |
| what does the main function do | — abstained (0.24) | explain_main ✓ |
| what is a map in go | — abstained (0.22) | explain_map ✓ |
| explain channels in go | explain_channel (0.82) | ✓ |
| what is a method in go | — abstained (0.24) | explain_method ✓ |
| what is defer in go | — abstained (0.37) | explain_defer ✓ |
| what is a variable in go | — abstained (0.10) | explain_string_rune ✗ |
| how do you run a go program | — abstained (0.24) | about_brain ✗ |
| what is an array in go | — abstained (0.33) | explain_array ✓ |
| what does this compiler error mean | explain_error (0.99) | ✓ |
| can you explain what this function does | read_code (0.82) | ✓ |
| what is this code doing | read_code (0.93) | ✓ |
| i do not understand this error message | — abstained (0.39) | explain_error ✓ |
| walk me through what this code does | read_code (0.98) | ✓ |
| what does undefined mean in this error | explain_error (0.99) | ✓ |

Concept lines: 5/15 at threshold, 12/15 by raw top.
Code-question lines: 5/6 at threshold, 6/6 by raw
top — the two dense cousin families (12 rows each)
fire confidently, which is the density lesson again:
concepts with a handful of rows guess right but
quietly. The four never-trained shapes: 0/4 labels at
threshold, but by raw top two land exactly ("is a map
like a dictionary" → explain_map; "is go a compiled
language" → explain_code) and one lands on the
sibling concept (channel → explain_buffered_channel);
only the example request misses. Shape generalisation
is partial, and is reported as such.

Replies from the fully trained brain, judged strictly
(coherent = on-topic, grammatical, actually answers):

Judged strictly, the replies lag the classifier —
the same gap the intent chapter named, wider here
because concept explanations are longer than
small-talk lines. Of 25: **1 coherent, 3 partial,
21 garbled**. The one clean answer is a cousin line
("that compiler error — the compiler is reporting
something it could not accept, in its own terse
wording. its message names the line and the rule the
code ran into."). The three partials lead with the
taught gist before trailing off: interface ("…only
and pair the with…" — the type-and-data pair is in
there), channel ("a channel — a typed pipe connecting
goroutines."), and the compiled-language line ("go —
a compiled, statically typed, and…"). The rest name
the concept and then garble, or borrow a neighbour's
explanation (the pointer line answers with the
conversion text; the variable line with the boolean
text). The classifier knows these concepts; the
generator cannot yet say them reliably.

### Held-out families and fences

HF kinds (words arm): 15 right / 12 wrong / 9 abstain,
against 16 / 12 / 8 last round — a wash in total, with
label neighbourhood up to 34/36 (from 29/36). Per
family: translate_text held at 5, play_game held at
5, find_directions held at 3, tell_story 2, order_food
still 0 (wrongs split between talk and find, neither
acceptable for it). `fix_code`: **0 right / 4 wrong /
2 abstain** — printed as-is. An intermediate data
draft read 2/4/0; the final corpus reads 0/4/2, and
the family's kind score visibly moves ±2 when a
dozen find-kind rows are added or removed, which is
itself the finding: the guess now lives in the
code-question neighbourhood (all four wrongs are kind
`find` — "this is a question about code" — and no
line fires a wrong *meaning*), while the frozen exam
accepts only `make` for it. Fences: U1 identical
(raw 30/30; 28 right / 2 abstain / 0 wrong), file
Set P 20/20, small-talk 22/22, WC labels 7/12 and WC
kinds 7/9 (both unchanged), TY 7/10 at threshold
against 9/10 last round — raw top still 9/10; the
two lost lines ("serch the web for trains" 0.48,
"send a mesage to kim" 0.45) sit a hair under the
threshold, diluted by the new labels. Fluency probe:
20 coherent / 2 partial / 2
garbled of 24, against 17/3/4 last round — "what do
you like to do" now answers cleanly, and the two
garbles are the stars line and the sunshine line..

Live, from the trained brain (the author's own
hand-test lines):

```text
> what is go
  go is a value an is. a can skill.
  intent: explain_code (0.62) · kind: find (0.98)
> what is a function
  a function is a new animal, a fresh bread part, in time.
  intent: (none confident enough) · kind: find (0.98)
> how do i code
  i am sorry today is awful. bad days end, even the long ones.
  intent: explain_code (0.74) · kind: find (0.93)
> where is the code
  i is trained it when you are new line i like about a simple cake.
  intent: find_place (0.87) · kind: find (0.95)
> can you debug this code for me
  i — you're asking me to code to its build. i follow you.
  intent: (none confident enough) · kind: find (0.97)
> translate hello into french
  hello! how are you today?
  intent: (none confident enough) · kind: talk (0.86)
> create a file named jim
  that's a request to create a file: homework. i understand.
  intent: create_file (0.97) · kind: make (0.98)
```

Read plainly: the classifier side of the screenshot
complaint is fixed — code questions land in the code
neighbourhood (explain_code fires for "what is go"
and "how do i code"; the debug line abstains on label
with kind find, no wrong meaning fired; translate
abstains on label and does not fire greeting —
though its reply is still greeting-shaped small
talk, the generator greeting where the classifier
would not; create_file holds at
0.97, with the reply again substituting a trained
name, homework, for jim — the standing extraction
limit). The reply side is not fixed: of the seven,
only the file line answers coherently.

### The revised boundary

The chat brain now knows the beginner Go curriculum
as *meanings*: ~49 concepts it can recognise from
varied wordings and explain in a sentence or two,
plus what error messages mean and what code does. It
knows them at kind level essentially always, and at
label level whenever the concept's raw guess is
allowed to speak — at threshold, only the denser
concepts speak. It still does not write code, fix
code, or know anything about *your* code — there is
no repo knowledge here, only the concepts. And the
reply brain still lags the classifier: it can garble
a line whose meaning it guessed correctly, as the
probe reply table shows. More rows per concept is
the named next lever; it is data, not mechanism,
again.

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
            and an optional kind level (plus optional
            subword piece schemes)
brain/      standalone glue: vocab building, the trainer loop
            with checkpoint-resume, greedy + checked sampling
cmd/train   train a brain: -data -out -epochs -layers -resume
cmd/ask     a demo REPL for a trained brain (not a chat product)
examples/   pairs.jsonl — 3,283 toy pairs to start from
            (650 of them intent-labeled and kinded; see
            "Intent understanding");
            commands.jsonl — 60 command rows with intents;
            filecommands.jsonl — 1,634 rows: 1,456 file-command rows over
            240 filenames, plus 178 label-only "chat" rows,
            filecmd/ — the runnable filesystem demo for them, and
            assistant/ — filecmd and the chat brain in one routed program
```

## Provenance

Carved from [github.com/golangast/gollemer](https://github.com/golangast/gollemer)
(the `internal/ai/neural/...` packages and the layers trainer
in its chat package) in November 2026, and relicensed under
GPL-3.0 for this standalone form. It is a copy, not a live
mirror: improvements in Gollemer do not flow here automatically.
Three carved files were gofmt'd during the extraction (two were
already unformatted upstream).
