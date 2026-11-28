# Triage: someone else's content, interpreted

A help desk gets the same customer lines all day — a billing
mistake, a crash, a password question, an outage — and each line
has to land in the right queue with the right urgency. The desk
knows what its lines mean. It should only have to say so once.

This example is that desk, and it is deliberately **not** the repo
owner's world: nothing here is about his scripts or his commands.
It exists to prove the brains interpret a stranger's content from
the stranger's own declarations.

## Run it

```
go run ./examples/triage
```

Type a customer line; it answers with what the line means here —
the declared action it matches, or the learned meaning and its
kind, or an abstention. It runs nothing: binding an action name
to real code is the desk's business. Add `-inspect` to see why a
line read the way it did — which words, cousin links, and rules
produced the reading.

## What it knows, and where it learned it

- `commands.json` — the desk's actions, declared once each: a
  name, the desk tool's command, a plain description, its slots.
  Matching is runtime; adding an action needs no retraining.
- `pairs.jsonl` — 288 example customer lines teaching twelve
  meanings (`route_billing`, `answer_password`,
  `escalate_outage`, …) in this domain's own five kinds
  (`route`, `answer`, `escalate`, `resolve`, `track`).
- `brain.gob` — the brain trained on those pairs, committed, so
  the example runs from a pull. It was trained with the one
  command any domain uses:

  ```
  go run ./cmd/train -dir examples/triage
  ```

- `probes.jsonl` — held-out lines no training row contains:
  unseen phrasings of the declared meanings, action lines, and
  control lines from outside the domain entirely. The probe test
  (`go test ./examples/triage/`) reports them right / wrong /
  abstain; the controls must abstain — a confident guess at a
  line the desk never declared is the one failure this system
  refuses.

## The limit, plainly

This is a bounded domain. The interpreter reads customer lines
well exactly to the extent the desk declared the meanings in
them — the descriptions above and the example lines. It does
not read arbitrary text and understand it, and it will not
pretend to: a line outside the declared domain gets an
abstention, not a guess.
