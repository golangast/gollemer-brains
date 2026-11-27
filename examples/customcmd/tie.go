package main

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Near-tie disambiguation: the ONE question the abstain
// path is allowed to ask. The matcher already abstains
// when two scripts explain a line almost equally well
// (its margin bar) — a coin flip dressed as an answer
// would be worse than either choice, but silence throws
// away a line the user almost certainly meant as one of
// the two. So when the abstention is a genuine near tie,
// the REPL asks which one, once:
//
//	did you mean <name1> or <name2>? (1/2, or enter for neither)
//
// The tie is defined on the matcher's ranked scores,
// exactly: at least two candidates with real matched
// mass, the leader at or above tieBar, the gap to the
// runner-up at or under tieGap — and each candidate
// resting on at least two matched words. That last
// clause is what "abstains by margin" means: the
// matcher's other bars (threshold, mass floor, two
// words) must all be cleared, leaving closeness as the
// only reason for the abstention. Without it, a line
// like "create a folder" — where half the registry ties
// on the single word "folder" and the matcher abstains
// on its two-words bar, not its margin — would be
// greeted with a question it never earned, and the
// teach offer it did earn would never arrive. (The
// leader being above the match threshold is what makes
// it a tie and not a shrug: both scripts genuinely
// explain the line.)
// Answering "1"/"2" — or the script's name — presents
// that script, with its slots extracted from the
// ORIGINAL line. Anything else, including enter, falls
// through to the ordinary abstain flow (brains, teach
// offer, silence) as if the question had never happened.
//
// The question is never asked in a plan (plan steps are
// rendered, not interviewed), never when one candidate
// is clearly ahead, and never twice for one input line:
// it lives at exactly one point in handle, on the
// single-line abstain path.
const (
	tieBar = 0.50 // the leader's score must reach this
	tieGap = 0.10 // and lead the runner-up by no more than this
)

// disambiguate offers the near-tie question for a line
// the matcher abstained on, and reports whether a script
// was chosen and presented. ranked is the matcher's
// ranking of the line, best first.
func (d *demo) disambiguate(line string, ranked []cmdmatch.Scored) bool {
	if d.nextLine == nil || len(ranked) < 2 {
		return false
	}
	top, second := ranked[0], ranked[1]
	if top.Mass <= 0 || second.Mass <= 0 {
		return false
	}
	if top.Matched < 2 || second.Matched < 2 {
		return false // an abstention on the two-words bar, not the margin
	}
	if top.Score < tieBar || top.Score-second.Score > tieGap {
		return false
	}
	fmt.Fprintf(d.out, "did you mean %s or %s? (1/2, or enter for neither)\n", top.Script.Name, second.Script.Name)
	ans, ok := d.nextLine("")
	if !ok {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(ans)) {
	case "1", strings.ToLower(top.Script.Name):
		d.presentMatch(top.Script, top.Score, line)
		return true
	case "2", strings.ToLower(second.Script.Name):
		d.presentMatch(second.Script, second.Score, line)
		return true
	}
	return false
}
