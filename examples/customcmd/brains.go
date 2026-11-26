package main

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/intent"
)

// All the brains: the matcher is the first call on a line,
// but it is no longer the only mind in the room. When the
// matcher abstains — the line names no script — the other
// brains read it, each doing the job it was trained for:
//
//   - the INTENT classifier (trained in memory at startup
//     from examples/pairs.jsonl) says what the line MEANS:
//     a label, its kind, and a confidence. Its read steers
//     everything below and is shown to the user — routing
//     is visible, never a hidden choice the user must make.
//   - the CHAT brain (brain.gob, the seq2seq trained on
//     pairs.jsonl) ANSWERS conversation: greetings, small
//     talk, feelings, and the questions it knows (the
//     explain_* meanings). Its replies can still garble —
//     the speaking wall stands — but the route is its job.
//   - the FILE brain (filebrain.gob, trained on
//     filecommands.jsonl) READS file wants: create, delete,
//     list. It prints what it understood — and that is all.
//     customcmd never acts on files; acting is the filecmd
//     example's job, in its sandbox.
//   - the generic EXTRACTOR (custombrain.gob) stays what
//     it was: the slot reader for matched scripts.
//
// The user never chooses a brain. They type a line — a
// want, a question, a greeting, a plan — and the routing
// below picks the reader behind the scenes. Brain names
// appear only in the banner's status list and on the
// labels of the answers themselves.
//
// Every brain is optional at runtime: a missing or
// unwired brain simply removes its route (the banner
// names it "not trained"), and with none wired the REPL
// behaves exactly as it did before they arrived — which
// is also how the older tests, wiring none, keep passing
// byte for byte.

// fileActionLabels are the meanings the file brain owns:
// the file route fires for these and no others.
var fileActionLabels = map[string]bool{
	"create_file": true,
	"delete_file": true,
	"list_files":  true,
}

// wantKinds are the kinds whose confident label means
// "a want no script here can do" — the teach offer's
// territory. A confident label of one of these kinds is
// itself want evidence, beside looksLikeWant's signals.
var wantKinds = map[string]bool{
	"make":        true,
	"remove":      true,
	"schedule":    true,
	"play":        true,
	"communicate": true,
	"travel":      true,
}

// intentRead is one line's read by the intent classifier:
// the label it chose ("" when it abstained) with its
// confidence and kind, plus the kind classifier's own
// guess for lines the label read left behind.
type intentRead struct {
	label     string
	conf      float64
	kind      string // KindOf(label); "" when the label abstained
	kindGuess string
	kindConf  float64
}

// readIntent asks the in-memory classifier about a line.
// ok is false when no classifier is wired.
func (d *demo) readIntent(line string) (intentRead, bool) {
	if d.clf == nil {
		return intentRead{}, false
	}
	r := intentRead{}
	r.label, r.conf = d.clf.Predict(line)
	if r.label != "" {
		r.kind = d.clf.KindOf(r.label)
	}
	r.kindGuess, r.kindConf = d.clf.PredictKind(line)
	return r, true
}

// describe renders the read the way answer blocks show
// it: the meaning, its kind, and the confidence — or the
// honest abstention, with the kind neighbourhood when
// even that much is known.
func (r intentRead) describe() string {
	if r.label != "" {
		return fmt.Sprintf("intent: %s · kind: %s · %.2f", r.label, r.kind, r.conf)
	}
	if r.kindGuess != "" {
		return fmt.Sprintf("intent: no confident label · kind looks like %s · %.2f", r.kindGuess, r.kindConf)
	}
	return "intent: no confident read"
}

// routeAbstain is consulted when the matcher abstains: it
// gives the line to the brain whose job it is, and
// reports whether one took it. The precedence (declared
// in the build notes before it was measured):
//
//  1. a file-action label goes to the file brain;
//  2. a confident conversational label (kind talk or
//     find — small talk, feelings, questions) goes to the
//     chat brain;
//  3. a confident want-kind label is NOT routed here —
//     it is a want, and the caller offers teaching;
//  4. with no label, the file brain's own classifier
//     reads the line (its labels are exactly the file
//     actions plus a chat class): a confident file
//     action goes to the file route;
//  5. with no label, a line that LOOKS like a want
//     (looksLikeWant) is the caller's teach territory,
//     not the chat brain's — the kind classifier guesses
//     "talk" for most unseen lines, talk being its
//     diffuse neighbourhood, and routing by that guess
//     alone would send every unknown want to chat and
//     starve the teach loop;
//  6. a remaining line whose kind guess is talk or find
//     goes to the chat brain.
//
// A route whose brain is not wired does not fire: the
// line falls through to the caller's ordinary path.
func (d *demo) routeAbstain(line string, ranked []cmdmatch.Scored, read intentRead, haveRead bool) bool {
	if !haveRead {
		return false
	}
	if fileActionLabels[read.label] {
		if d.filePredict == nil {
			return false
		}
		d.fileRoute(line, read)
		return true
	}
	if read.label != "" {
		if read.kind == "talk" || read.kind == "find" {
			if d.chatAnswer == nil {
				return false
			}
			d.chatRoute(line, read)
			return true
		}
		return false // a want-kind label: the caller's teach path
	}
	if d.filePredict != nil {
		if lp, _ := d.filePredict(line); fileActionLabels[lp] {
			d.fileRoute(line, read)
			return true
		}
	}
	if d.looksLikeWant(line, ranked) {
		return false
	}
	if read.kindGuess == "talk" || read.kindGuess == "find" {
		if d.chatAnswer == nil {
			return false
		}
		d.chatRoute(line, read)
		return true
	}
	return false
}

// chatRoute hands a conversational line to the chat
// brain and prints its answer, labelled, with the intent
// read that sent it there.
func (d *demo) chatRoute(line string, read intentRead) {
	ans, err := d.chatAnswer(line)
	if err != nil || strings.TrimSpace(ans) == "" {
		fmt.Fprintln(d.out, "chat brain: (it could not answer that)")
	} else {
		fmt.Fprintf(d.out, "chat brain: %s\n", ans)
	}
	fmt.Fprintf(d.out, "  (%s)\n", read.describe())
}

// fileRoute hands a file want to the file brain and
// prints what it understood: its label for the line and,
// for create/delete, the filename its extractor heard.
// UNDERSTANDING ONLY — the block says so on its face:
// nothing is created, deleted, or listed here.
func (d *demo) fileRoute(line string, read intentRead) {
	label, conf := read.label, read.conf
	if lp, lc := d.filePredict(line); lp != "" {
		label, conf = lp, lc // the file brain's own read leads its block
	}
	what := label
	if label == "create_file" || label == "delete_file" {
		if raw, err := d.fileExtract(line); err == nil {
			if name, ok := cleanFileName(raw); ok {
				what = fmt.Sprintf("%s — file %s", label, name)
			} else {
				what = fmt.Sprintf("%s — no clean filename heard", label)
			}
		}
	}
	fmt.Fprintf(d.out, "file brain: reads this as %s\n", what)
	fmt.Fprintf(d.out, "  (%s · the file brain's own confidence %.2f — understanding only: nothing was created, deleted, or listed; acting on files is the filecmd example's job.)\n", read.describe(), conf)
}

// cleanFileName judges the file brain's raw answer the
// way filecmd's host does: exactly one token, a plain
// filename with an extension — no path separators, no
// "..". The model proposes; the host disposes. (The same
// guard, kept beside its second user; filecmd's copy is
// the original.)
func cleanFileName(raw string) (string, bool) {
	fields := strings.Fields(strings.TrimSpace(raw))
	if len(fields) != 1 {
		return "", false
	}
	name := fields[0]
	if strings.ContainsAny(name, `/\`) || strings.Contains(name, "..") {
		return "", false
	}
	if !fileNameRe.MatchString(name) {
		return "", false
	}
	return name, true
}

var fileNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]*\.[A-Za-z0-9]+$`)

// teachOffer decides, on the ordinary abstain path,
// whether the line earns the teach offer, and prints the
// intent read before the offer when the classifier chose
// a label (the brief's shape: "this reads as: make ·
// create_folder"). The offer fires on today's signals
// (looksLikeWant) or on a confident want-kind label —
// and never for a line a brain route already took, which
// is how small talk stays out of it.
func (d *demo) teachOffer(line string, ranked []cmdmatch.Scored, read intentRead, haveRead bool) bool {
	if !d.looksLikeWant(line, ranked) && !(haveRead && read.label != "" && wantKinds[read.kind]) {
		return false
	}
	if haveRead && read.label != "" {
		fmt.Fprintf(d.out, "this reads as: %s · %s (%.2f)\n", read.kind, read.label, read.conf)
	}
	d.teachFlow(line)
	return true
}

// brainsStatus renders the banner's status list: every
// brain the program can use, with the ones that did not
// load named as not trained. It is a status line, not a
// menu — nothing in it is chosen by the user.
func brainsStatus(clf *intent.Classifier, chatOK, fileOK bool) string {
	part := func(name string, ok bool) string {
		if ok {
			return name
		}
		return name + " (not trained)"
	}
	return strings.Join([]string{
		"scripts",
		part("intent", clf != nil),
		part("chat", chatOK),
		part("file", fileOK),
		"extractor",
	}, " · ")
}
