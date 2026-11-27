package cmdmatch

import (
	"fmt"
	"testing"
)

// Determinism: a Matcher built from the same registry
// must be the same matcher in every process. The cousin
// claim rule (see expand) is where construction order
// used to leak into scores — two profile words claiming
// the same cousin settled on whichever the map ranging
// visited first, and the cousin borrows its winner's
// idf, so verdicts near the bars flipped per process.
// These tests pin the rule and the stability it buys.

// conflictRegistry puts two DEFINING sources in one
// profile claiming the same cousin ("save", stemmed
// "sav", is a cousin of both "back" and "backup"), with
// different document frequencies: "back" appears in two
// profiles, "backup" in one, so idf(backup) > idf(back)
// and the stronger-evidence source must win the claim.
func conflictRegistry() *Registry {
	return &Registry{Scripts: []Declaration{
		{
			Name:        "vault-photos",
			Command:     "./vault.sh",
			Description: "back backup the photo vault",
		},
		{
			Name:        "truck-log",
			Command:     "./truck.sh",
			Description: "back the truck",
		},
	}}
}

func expandedClaim(t *testing.T, m *Matcher, script, cousin string) expandedWord {
	t.Helper()
	for _, p := range m.profiles {
		if p.decl.Name == script {
			if ex, ok := p.expanded[cousin]; ok {
				return ex
			}
			t.Fatalf("script %s has no expanded claim on %q", script, cousin)
		}
	}
	t.Fatalf("no script %s", script)
	return expandedWord{}
}

func TestCousinClaimStrongerIDFWins(t *testing.T) {
	for i := 0; i < 100; i++ {
		m := NewMatcher(conflictRegistry())
		ex := expandedClaim(t, m, "vault-photos", "sav")
		if ex.source != "backup" || !ex.defining {
			t.Fatalf("build %d: claim on sav = %+v; want source backup, defining", i, ex)
		}
	}
}

func TestCousinClaimDefiningBeatsExampleDespiteIDF(t *testing.T) {
	// "back" is defining in vault-photos and appears in a
	// second profile too (so its idf is the LOWER one);
	// "backup" is example-only in vault-photos with the
	// higher idf. Class beats idf: the defining source
	// keeps the claim.
	reg := &Registry{Scripts: []Declaration{
		{
			Name:        "vault-photos",
			Command:     "./vault.sh",
			Description: "back the photo vault",
			Examples:    []string{"backup the photo vault"},
		},
		{
			Name:        "truck-log",
			Command:     "./truck.sh",
			Description: "back the truck",
		},
	}}
	for i := 0; i < 100; i++ {
		m := NewMatcher(reg)
		ex := expandedClaim(t, m, "vault-photos", "sav")
		if ex.source != "back" || !ex.defining {
			t.Fatalf("build %d: claim on sav = %+v; want source back, defining", i, ex)
		}
	}
}

func TestCousinClaimLexicographicTieBreak(t *testing.T) {
	// "back" and "copy" are both defining in one profile,
	// each in exactly one profile registry-wide, so their
	// idfs tie; both claim "sav". The lexicographically
	// smaller source ("back") wins.
	reg := &Registry{Scripts: []Declaration{
		{
			Name:        "vault-photos",
			Command:     "./vault.sh",
			Description: "back copy the photo vault",
		},
		{
			Name:        "truck-log",
			Command:     "./truck.sh",
			Description: "park the truck",
		},
	}}
	for i := 0; i < 100; i++ {
		m := NewMatcher(reg)
		ex := expandedClaim(t, m, "vault-photos", "sav")
		if ex.source != "back" || !ex.defining {
			t.Fatalf("build %d: claim on sav = %+v; want source back, defining", i, ex)
		}
	}
}

func TestClaimWinsRule(t *testing.T) {
	idf := func(w string) float64 {
		return map[string]float64{"back": 1.0, "backup": 2.0, "copy": 1.0}[w]
	}
	// Class first: a defining claim beats an example one
	// even at lower idf, and loses to one at any idf.
	if !claimWins("back", true, expandedWord{source: "backup", defining: false}, idf) {
		t.Error("defining claim should beat an example claim")
	}
	if claimWins("backup", false, expandedWord{source: "back", defining: true}, idf) {
		t.Error("example claim should not beat a defining claim")
	}
	// Then idf: the stronger source displaces, the weaker
	// does not.
	if !claimWins("backup", true, expandedWord{source: "back", defining: true}, idf) {
		t.Error("higher-idf claim should displace a lower-idf one")
	}
	if claimWins("back", true, expandedWord{source: "backup", defining: true}, idf) {
		t.Error("lower-idf claim should not displace a higher-idf one")
	}
	// Then name: equal class and idf, the smaller source
	// wins.
	if !claimWins("back", true, expandedWord{source: "copy", defining: true}, idf) {
		t.Error("lexicographically smaller source should win an idf tie")
	}
	if claimWins("copy", true, expandedWord{source: "back", defining: true}, idf) {
		t.Error("lexicographically larger source should lose an idf tie")
	}
}

func TestRankDeterministicAcrossBuilds(t *testing.T) {
	lines := []string{
		"save the photo vault",
		"store the photos in the vault",
		"back up the truck",
	}
	want := ""
	for i := 0; i < 200; i++ {
		m := NewMatcher(conflictRegistry())
		got := ""
		for _, line := range lines {
			for _, s := range m.Rank(line) {
				got += fmt.Sprintf("%s %.6f/%.6f/%d;", s.Script.Name, s.Score, s.Mass, s.Matched)
			}
			got += "\n"
		}
		if i == 0 {
			want = got
			continue
		}
		if got != want {
			t.Fatalf("build %d ranked differently:\n%s\nwant:\n%s", i, got, want)
		}
	}
}
