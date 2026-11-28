package cmdmatch

import "testing"

// Explain: the provenance of a match — which line words
// reached the profile, and by which link. The registry
// is the determinism round's known case: backup-photos'
// description carries "backs", whose cousin is "save".

func explainRegistry() *Registry {
	return &Registry{Scripts: []Declaration{
		{
			Name:        "backup-photos",
			Command:     "./backup-photos.sh",
			Description: "backs up my photo library to the external drive",
			Examples:    []string{"archive the pictures to the spare drive every sunday"},
		},
		{
			Name:        "start-dev-server",
			Command:     "./dev.sh",
			Description: "starts the local development server for my website",
		},
	}}
}

func hitFor(hits []WordHit, word string) (WordHit, bool) {
	for _, h := range hits {
		if h.Word == word {
			return h, true
		}
	}
	return WordHit{}, false
}

func TestExplainDefiningAndCousin(t *testing.T) {
	m := NewMatcher(explainRegistry())
	hits := m.Explain("save the photos", "backup-photos")
	// "save" (stemmed "sav") is not in the profile: it
	// arrives as a cousin of the defining word "back".
	sav, ok := hitFor(hits, "sav")
	if !ok {
		t.Fatalf("no hit for sav in %v", hits)
	}
	if sav.Via != "cousin" || sav.Source != "back" || !sav.Defining {
		t.Errorf("sav hit = %+v, want cousin of defining word back", sav)
	}
	// "photos" (stemmed "photo") is a defining word of
	// the description itself.
	photo, ok := hitFor(hits, "photo")
	if !ok || photo.Via != "defining" {
		t.Errorf("photo hit = %+v, want defining", photo)
	}
}

func TestExplainExampleAndMiss(t *testing.T) {
	m := NewMatcher(explainRegistry())
	hits := m.Explain("spare drive pictures", "backup-photos")
	// "spare" (stemmed "spar") appears only in the
	// example sentence.
	if h, _ := hitFor(hits, "spar"); h.Via != "example" {
		t.Errorf("spar hit = %+v, want example", h)
	}
	// A word the profile does not know explains as a
	// miss, not as silence: the hit is present, Via "".
	hits = m.Explain("quantum bananas", "backup-photos")
	if h, ok := hitFor(hits, "quantum"); !ok || h.Via != "" {
		t.Errorf("quantum hit = %+v (present %v), want a miss", h, ok)
	}
	// An unknown script explains nothing.
	if got := m.Explain("save the photos", "no-such-script"); got != nil {
		t.Errorf("Explain for unknown script = %v, want nil", got)
	}
}

func TestExplainTypoRepair(t *testing.T) {
	m := NewMatcher(explainRegistry())
	hits := m.Explain("photoo library", "backup-photos")
	h, ok := hitFor(hits, "photoo")
	if !ok {
		t.Fatalf("no hit for photoo in %v", hits)
	}
	if h.ScoredAs != "photo" || h.Via != "defining" {
		t.Errorf("photoo hit = %+v, want scored as photo, defining", h)
	}
}
