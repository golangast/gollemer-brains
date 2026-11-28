package main

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/data"
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
// appear only on the labels of the answers themselves.
//
// Every brain is optional at runtime: a missing or
// unwired brain simply removes its route, and with none
// wired the REPL behaves exactly as it did before they
// arrived — which is also how the older tests, wiring
// none, keep passing.

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
	// fallback marks a label composed by the factorized
	// fallback (factor.go) after the primary read
	// abstained. It routes and gists exactly like a
	// primary read; the mark is internal (no annotation
	// is ever printed) and exists for tests and tuning.
	fallback bool
	// withheld records a confident primary read the
	// disjointness check set aside (disjoint.go): the
	// label it was, and the disjoint-kind rival it was
	// torn against. Both empty when nothing was
	// withheld. The mark is internal, like fallback's.
	withheld       string
	withheldRival  string
	withheldRivalK string // the rival's kind (for the inspect trace)
}

// readIntent asks the in-memory classifier about a line.
// ok is false when no classifier is wired. When the label
// read abstains, the factorized fallback (factor.go) gets
// its one chance to compose a read from the kind guess
// and the line's object words; the read it returns is
// consumed at exactly the point this one is, so the
// precedence around it — matcher first, looksLikeWant
// before any kind-guess routing in the no-label branch —
// is untouched.
func (d *demo) readIntent(line string) (intentRead, bool) {
	if d.clf == nil {
		return intentRead{}, false
	}
	r := intentRead{}
	r.label, r.conf = d.clf.Predict(line)
	if r.label != "" {
		r.kind = d.clf.KindOf(r.label)
		// The disjointness validation (disjoint.go): a
		// confident read torn against a disjoint-kind
		// rival within the near-tie gap is withheld —
		// the line continues as an abstention below,
		// fallback and kind routing included.
		ranked := d.clf.RankedLabels(line)
		if len(ranked) > 0 && ranked[0].Label == r.label {
			if rival, ok := disjointWithhold(d.clf.KindOf, ranked); ok {
				r.withheld = r.label
				r.withheldRival = rival
				r.withheldRivalK = d.clf.KindOf(rival)
				r.label = ""
				r.kind = ""
			}
		}
	}
	r.kindGuess, r.kindConf = d.clf.PredictKind(line)
	if r.label == "" {
		if label, ok := d.fallbackRead(line, r.kindGuess); ok {
			r.label = label
			r.kind = d.clf.KindOf(label)
			r.fallback = true
		}
	}
	return r, true
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
//  6. a remaining line whose kind guess is FIND is
//     claimed SILENTLY. It used to go to the chat brain,
//     but a find-shaped line the label read abstained
//     on, the factorized fallback could not compose, and
//     no gist answers has nothing behind it but free
//     generation — and generation on a missed concept
//     read is where the garble lived, up to a wrong
//     concept's text stated confidently. Silence is the
//     honest answer (the same rule John's no-dead-end
//     ruling gives unclaimed lines). A remaining line
//     whose kind guess is TALK still goes to the chat
//     brain: conversation is generation's home turf, and
//     the measured record has talk abstentions producing
//     partial, on-topic replies ("explain variables to
//     me" opens with the variable gist's own first
//     sentences) that silence would only lose.
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
	if read.kindGuess == "find" {
		// Silent abstention (precedence 6 above): the
		// line is claimed — no teach offer, no dead-end
		// line — and nothing is printed. Whether the chat
		// brain is wired no longer matters here: there is
		// no read for it to answer from.
		return true
	}
	if read.kindGuess == "talk" {
		if d.chatAnswer == nil {
			return false
		}
		d.chatRoute(line, read)
		return true
	}
	return false
}

// modalGists computes, per intent label, the concept's
// taught gist: the most frequent output among the
// label's pairs (ties keep the earliest in file order).
// The speaking data was built convergent -- a handful of
// canonical gists reused across many phrasings -- so the
// mode is the concept's canonical answer, not an average.
func modalGists(pairs []data.Pair) map[string]string {
	counts := map[string]map[string]int{}
	for _, p := range pairs {
		if p.Intent == "" || strings.TrimSpace(p.Output) == "" {
			continue
		}
		if counts[p.Intent] == nil {
			counts[p.Intent] = map[string]int{}
		}
		counts[p.Intent][p.Output]++
	}
	gists := make(map[string]string, len(counts))
	for _, p := range pairs {
		if p.Intent == "" {
			continue
		}
		if _, counted := counts[p.Intent]; !counted {
			continue
		}
		if _, done := gists[p.Intent]; done {
			continue
		}
		best, bestN := "", 0
		seen := map[string]bool{}
		for _, q := range pairs {
			if q.Intent != p.Intent || seen[q.Output] {
				continue
			}
			seen[q.Output] = true
			if counts[p.Intent][q.Output] > bestN {
				best, bestN = q.Output, counts[p.Intent][q.Output]
			}
		}
		gists[p.Intent] = best
	}
	return gists
}

// questionOpeners are the line-initial forms that mark a
// line as a question even without a "?": the probe lines
// and most typed questions arrive unpunctuated.
var questionOpeners = []string{
	"what", "why", "how", "when", "which", "who",
	"can you", "could you", "tell me", "explain",
	"is", "are", "does", "do",
}

// looksLikeQuestion reports whether a line has a
// question's shape: a trailing "?" or a question opener.
func looksLikeQuestion(line string) bool {
	s := strings.ToLower(strings.TrimSpace(line))
	if strings.HasSuffix(s, "?") {
		return true
	}
	for _, o := range questionOpeners {
		if s == o || strings.HasPrefix(s, o+" ") {
			return true
		}
	}
	return false
}

// answerChat produces the chat brain's answer to a line
// the read routed here. The classifier reads concept
// lines far better than the chat brain answers them, so
// a confident label (a non-empty label IS at/above the
// classifier's standing threshold) on a question-shaped
// line is answered with the concept's taught gist --
// the modal output of the label's own training rows --
// instead of free generation. (Prepending the label as
// a marker to the decoder input was measured first and
// lifted nothing; see the build notes.) Every other
// line -- no label, no gist, not a question -- takes the
// plain generation path, unchanged.
func (d *demo) answerChat(line string, read intentRead) (string, error) {
	if read.label != "" && looksLikeQuestion(line) {
		if gist, ok := d.chatGists[read.label]; ok && strings.TrimSpace(gist) != "" {
			return gist, nil
		}
	}
	return d.chatAnswer(line)
}

// chatRoute hands a conversational line to the chat
// brain and prints its answer, labelled. The intent read
// that sent it there steers the routing but is not
// printed (John's rule, 2026-11-26: no annotations —
// just the answer).
func (d *demo) chatRoute(line string, read intentRead) {
	ans, err := d.answerChat(line, read)
	if err != nil || strings.TrimSpace(ans) == "" {
		fmt.Fprintf(d.out, "%s (it could not answer that)\n", d.paint(ansiLabel, "chat brain:"))
	} else {
		fmt.Fprintf(d.out, "%s %s\n", d.paint(ansiLabel, "chat brain:"), ans)
	}
}

// fileRoute hands a file want to the file brain and
// prints what it understood: its label for the line and,
// for create/delete, the filename its extractor heard.
// UNDERSTANDING ONLY — the block says so on its face:
// nothing is created, deleted, or listed here.
func (d *demo) fileRoute(line string, read intentRead) {
	label := read.label
	if lp, _ := d.filePredict(line); lp != "" {
		label = lp // the file brain's own read leads its block
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
	fmt.Fprintf(d.out, "%s reads this as %s\n", d.paint(ansiLabel, "file brain:"), what)
	fmt.Fprintln(d.out, "  understanding only — nothing was created, deleted, or listed.")
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
// whether the line earns the teach offer. The offer
// fires on today's signals (looksLikeWant) or on a
// confident want-kind label — and never for a line a
// brain route already took, which is how small talk
// stays out of it. The offer stands alone: no read is
// printed before it (John's rule, 2026-11-26).
func (d *demo) teachOffer(line string, ranked []cmdmatch.Scored, read intentRead, haveRead bool) bool {
	if !d.looksLikeWant(line, ranked) && !(haveRead && read.label != "" && wantKinds[read.kind]) {
		return false
	}
	d.teachFlow(line)
	return true
}
