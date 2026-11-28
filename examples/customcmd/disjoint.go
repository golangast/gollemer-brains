package main

import (
	"github.com/golangast/gollemer-brains/intent"
	"github.com/golangast/gollemer-brains/interpret"
)

// Disjointness — the validation check that withholds a
// confident read torn against an opposite-kind rival —
// lives in the interpret package (interpret/disjoint.go),
// under the programs that use it. This file is the
// adapter: the names customcmd has always used, bound to
// that check. Behaviour is unchanged.

const disjointGap = interpret.DisjointGap

func kindsDisjoint(a, b string) bool {
	return interpret.KindsDisjoint(a, b)
}

func disjointWithhold(kindOf func(string) string, ranked []intent.LabelScore) (string, bool) {
	return interpret.DisjointWithhold(kindOf, ranked)
}
