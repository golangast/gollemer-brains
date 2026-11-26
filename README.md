# gollemer-brains

gollemer-brains is a small language brain written in pure Go, with no dependencies. You teach it meanings by giving it example sentences, once. When you later say something it has never seen, it guesses the meaning from the words the sentence carries. That is the whole idea: declare what things mean, and let unseen phrasings find their way to the meaning, instead of hardcoding every way a person might ask.

```
                        one prompt
                            |
        +-------------------+-------------------+
        |                   |                   |
  your scripts        the chat brain      the file brain
  (described by       (questions and      (file actions,
   you, in words)      conversation)       understood)
        |                   |                   |
        +-------------------+-------------------+
                            |
              preview, plan, would-run line
```

The main program is one prompt, and every brain sits behind it; you never choose a brain. You describe your own scripts in plain words, and from then on the prompt knows which script you mean when you ask for it, pulls the arguments out of your sentence, and shows you a preview of what would run. Say several wants in one line and it lays them out as a plan, in the order they should run. Ask a question and the chat brain answers; talk about files and the file brain reads it. If no brain can place your line, it asks to be taught, and what you teach it is kept. It runs nothing. It shows you the exact command it would run, and stops there — executing is a separate step, and it is not in this program.

```
> i have a command named copy and it will copy files from one folder to another and it is in file copy.sh
learned: copy — copy files from one folder to another
command: ./copy.sh (slots: from, to). it chooses copy from now on.
> copy files from jim to jake
understood: copy — arguments: from=jim, to=jake
preview: copy files from jim to jake
would run: ./copy.sh jim jake
```

The smaller examples each do one thing. One sorts a dump of what's on your mind into quick wins, deep focus, venting, and parked. One works with files inside a sandbox. One combines chat and files. The limits are plain: the prompt guesses from your words, so a script is only as findable as the description you gave it. Names it has never seen can be misheard. The chat brain understands more than it can say, and its replies are sometimes garbled. Everything it knows is written in files you can read and edit — nothing is hidden, and nothing needs retraining when you teach it something new.

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

## Run it

```
go run ./examples/customcmd     the one prompt: your scripts, chat, files, plans
go run ./examples/thoughtsort   sort a head full of thoughts
go run ./examples/assistant     chat and files together
go run ./examples/filecmd       files in a sandbox
```

The brains ship trained, so there is nothing to build first.

License: GPL-3.0.
