# gollemer-brains

You have scripts that do useful things. To use one, you have to remember its name, where it lives, and what order its arguments go in — and if you say what you want in your own words instead, nothing understands you. Every tool works like this: you learn its words, exactly, or you get nothing. Here you teach the tool your words, once, in one sentence — and from then on you just say what you want.

```
> i have a command named copy and it will copy files from one folder to another and it is in file copy.sh
learned: copy — copy files from one folder to another
command: ./copy.sh (slots: from, to). it chooses copy from now on.
> copy files from jim to jake
understood: copy — arguments: from=jim, to=jake
preview: copy files from jim to jake
would run: ./copy.sh jim jake
```

Real jobs are worse than one script. Backing up, migrating, and cleaning up is three scripts in the right order, and the order lives in your head. And a program that guesses what you mean is not a program you want pressing enter — so this one never runs anything. Say the whole job in one line and it comes back as a plan: the right scripts, in the right order, each with the exact command it would run. Then it stops.

```
> compress the logs, back up my photos and migrate a database from folder jim to jake
plan: 3 steps (3 ready, 0 gaps)
[ ] 1. Prepare — concept: backup-photos
    Would run: ./scripts/backup-photos.sh
[ ] 2. Execute — concept: migrate-database
    Would run: ./scripts/migrate-db.sh jim jake
[ ] 3. Cleanup — concept: compress-logs
    Would run: ./scripts/compress-logs.sh
nothing has run yet.
```

Sometimes there is no script at all — just a head full of unsorted thoughts at the end of the day. Some are two-minute jobs, some need an hour of quiet, some are just noise you need said out loud. Dump them here and they come back sorted by what they really are. The guesses are not magic, and the limits are plain: a script can only be found from the words you described it with, a name it has never seen can come back misheard, and questions it was never taught can come back garbled. What it knows, it knows because you told it — in sentences you can read, in files you can edit.

```
> reply to the invoice email
  -> quick_win · kind: act · 0.90
> the parser rewrite needs a quiet afternoon
  -> deep_focus · kind: act · 0.97
> i am so tired of this flaky test
  -> venting · kind: express · 0.94
> sort

Quick wins — act now, minutes each (1):
  - reply to the invoice email

Deep focus — needs a real block, schedule it (1):
  - the parser rewrite needs a quiet afternoon

Venting — no action, it just needed saying (1):
  - i am so tired of this flaky test
```

## Run it

```
go run ./examples/customcmd     say what you want; get the script, the preview, the plan
go run ./examples/thoughtsort   dump your thoughts; get them back sorted
go run ./examples/assistant     ask questions and work with files in one place
go run ./examples/filecmd       work with files in a sandbox
go run ./examples/triage        read support tickets against meanings a help desk declared
```

## Your own content

Nothing above is special to the words in this repo. The meanings
live in two files — a declarations file saying what each thing
means, and a pairs file of example lines teaching those meanings —
and any program can read lines against them through the
`interpret` package: hand it the two files, get back what each
line means. The triage example is exactly that, in a domain that
is not this repo's: a help desk's actions and customer lines,
declared in its folder, interpreted from a pull. The file formats
are the contract, written down in [FORMATS.md](FORMATS.md).

The limit is plain here too: it interprets a domain exactly as
far as that domain's meanings have been declared, and no further.
It does not read arbitrary text and understand it — a line
outside the declared meanings gets an abstention, not a guess.

License: GPL-3.0.
