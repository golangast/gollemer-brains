package main

import "github.com/golangast/gollemer-brains/intent"

// Disjointness: a validation check on the intent read
// (round 4). Some kinds are action POLARITIES — one
// brings a thing into being, the other takes it out;
// one starts a thing, the other stops it. A single line
// cannot mean both, so when the classifier's top two
// reads sit in disjoint kinds at near-equal confidence,
// the "winner" is a coin flip dressed as a read: a hair
// of probability, not evidence, picked it. The check
// withholds such a read — the label is set aside (and
// remembered on the read, for tests and the inspect
// trace) — and the line continues down the ordinary
// abstention path: the factorized fallback may still
// compose a read (its whole-object-set evidence is a
// different standard than the torn softmax, and if it
// composes exactly one label, that label stands), the
// kind guess still routes, and silence or the teach
// offer still close the line. Nothing is blended and
// nothing is picked by the hair.
//
// The pairs, derived from the kind semantics:
//
//   - make × remove: bringing into being vs taking out
//     of being (create_file against delete_file is the
//     torn pair the seam lines actually produce —
//     "do not copy the file" reads delete_file over
//     copy_file on negation-blindness alone).
//   - schedule × remove: putting a future thing on the
//     books vs cancelling/removing — the canonical
//     set-vs-cancel opposition.
//   - play × remove: starting a thing vs stopping it.
//
// Deliberately NOT paired: find against anything (a
// question can ride along with any neighbourhood, and
// find's near-ties with talk are the speaking corpus's
// ordinary traffic — withholding them would silence
// good reads); talk against anything (talk is the
// diffuse neighbourhood — greetings and feelings are
// not the opposite of a reminder, even though this
// corpus files cancel_stop under talk, so the
// schedule-vs-cancel opposition cannot be stated at
// kind level without also claiming schedule opposes
// greeting); and same-kind rivals never trigger the
// check at all (struct vs field is a sibling miss, a
// different wart, untouched here).
//
// Near-equal is disjointGap on the two probabilities —
// the same 0.10 the near-tie question uses on matcher
// scores (tie.go's tieGap), the repo's standing "too
// close to call" distance. Measured before building:
// across the router exam, the speaking probes, the
// factor probes and the round-3 seam lines, NO line's
// confident read has a disjoint-kind runner-up within
// the gap — the check fires zero times on the fences
// and exists for the genuinely torn read, the shape
// new data (or another user's corpus, round 5) brings.
const disjointGap = 0.10

// disjointKindPairs is the pair list above, as a set of
// unordered kind pairs.
var disjointKindPairs = map[[2]string]bool{
	{"make", "remove"}:     true,
	{"play", "remove"}:     true,
	{"remove", "make"}:     true,
	{"remove", "play"}:     true,
	{"remove", "schedule"}: true,
	{"schedule", "remove"}: true,
}

// kindsDisjoint reports whether two kinds are declared
// opposites. A kind is never disjoint with itself, and
// an unknown kind is disjoint with nothing.
func kindsDisjoint(a, b string) bool {
	if a == "" || b == "" || a == b {
		return false
	}
	return disjointKindPairs[[2]string{a, b}]
}

// disjointWithhold applies the check to a confident
// primary read: given the classifier's full ranking of
// the line, it reports the rival label when the read
// must be withheld — the winner and the runner-up in
// disjoint kinds, their probabilities within
// disjointGap. Any other shape (no runner-up, a
// same-kind or non-disjoint rival, a clear margin)
// reports no rival and the read stands.
func disjointWithhold(kindOf func(string) string, ranked []intent.LabelScore) (rival string, ok bool) {
	if len(ranked) < 2 {
		return "", false
	}
	top, second := ranked[0], ranked[1]
	if top.Prob-second.Prob > disjointGap {
		return "", false
	}
	k1, k2 := kindOf(top.Label), kindOf(second.Label)
	if !kindsDisjoint(k1, k2) {
		return "", false
	}
	return second.Label, true
}
