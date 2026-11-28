package main

import (
	"github.com/golangast/gollemer-brains/intent"
	"github.com/golangast/gollemer-brains/interpret"
)

// The factorized fallback read — the composition of a
// kind and a line's object words into a label, the kind
// tree it walks, and the parent walk — lives in the
// interpret package (interpret/factor.go), under the
// programs that use it. This file is the adapter: the
// names customcmd has always used, bound to that
// machinery, plus the demo's two touchpoints (lazy
// derivation from its classifier, and the fallback read
// with its inspect traces). Behaviour is unchanged —
// the move is mechanical, and the factor tests pin the
// same cases they always did.

type labelFactor = interpret.LabelFactor
type factorTable = interpret.FactorTable
type kindTree = interpret.KindTree

func deriveFactors(clf *intent.Classifier) *factorTable {
	return interpret.DeriveFactors(clf)
}

func deriveKindTree(kinds map[string]string) *kindTree {
	return interpret.DeriveKindTree(kinds)
}

// factors returns the demo's factor table, derived from
// its classifier on first use (tests wire the classifier
// after constructing the demo, so derivation is lazy).
func (d *demo) factors() *factorTable {
	if d.factorTbl == nil && d.clf != nil {
		d.factorTbl = deriveFactors(d.clf)
	}
	return d.factorTbl
}

// fallbackRead composes a read for a line the label read
// left behind (see interpret.FactorTable.FallbackRead
// for the composition and the walk). The outcome's
// provenance line goes to the inspect trace, exactly as
// it did when the composition lived here.
func (d *demo) fallbackRead(line, kindGuess string) (string, bool) {
	t := d.factors()
	if t == nil {
		return "", false
	}
	label, ok, trace := t.FallbackRead(d.clf, line, kindGuess)
	if trace != "" {
		d.tracef("%s", trace)
	}
	return label, ok
}
