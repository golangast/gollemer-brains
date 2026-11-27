package main

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// wordDiff counts the token positions at which two
// sentences differ; -1 when their token counts differ at
// all (then "one substitution" is not even the shape).
func wordDiff(a, b string) int {
	af, bf := strings.Fields(a), strings.Fields(b)
	if len(af) != len(bf) {
		return -1
	}
	n := 0
	for i := range af {
		if af[i] != bf[i] {
			n++
		}
	}
	return n
}

func TestSynthesizeCapAndDedup(t *testing.T) {
	// A description dense with family words offers far
	// more than six substitutions; the cap holds, the
	// variants are unique, and none repeats the source.
	decl := cmdmatch.Declaration{
		Name:        "everything",
		Description: "copy the photos and save the files and store the images and rename the documents",
		Examples:    []string{"copy the photos and save the files and store the images and rename the documents"},
	}
	out := synthesizeExamples(decl)
	if len(out) == 0 {
		t.Fatal("synthesizeExamples returned nothing for a family-dense description")
	}
	if len(out) > maxSynthesizedExamples {
		t.Errorf("got %d variants, cap is %d", len(out), maxSynthesizedExamples)
	}
	seen := map[string]bool{decl.Description: true}
	for _, v := range out {
		if seen[v] {
			t.Errorf("variant %q duplicates the source or another variant", v)
		}
		seen[v] = true
	}
}

func TestSynthesizeOneSubstitution(t *testing.T) {
	// Every variant of a simple entry differs from its
	// source in exactly one token: one cousin swapped in,
	// or one value swapped out — never a rewrite.
	decl := cmdmatch.Declaration{
		Name:        "mover",
		Description: "move the files from jim to jake",
		Slots:       []string{"from", "to"},
		Examples:    []string{"move the files from jim to jake"},
	}
	out := synthesizeExamples(decl)
	if len(out) == 0 {
		t.Fatal("no variants for a substitutable entry")
	}
	for _, v := range out {
		if d := wordDiff(decl.Description, v); d != 1 {
			t.Errorf("variant %q differs from the source in %d tokens, want exactly 1", v, d)
		}
	}
}

func TestSynthesizeValueSwap(t *testing.T) {
	// The from-seat's jim is swapped for the pool's first
	// unbound value, holidays — and no variant seats the
	// same value at both ends.
	decl := cmdmatch.Declaration{
		Name:        "copier",
		Description: "copy files from jim to jake",
		Slots:       []string{"from", "to"},
		Examples:    []string{"copy files from jim to jake"},
	}
	out := synthesizeExamples(decl)
	found := false
	for _, v := range out {
		if v == "copy files from holidays to jake" {
			found = true
		}
		if strings.Contains(v, "from jake to jake") || strings.Contains(v, "from jim to jim") {
			t.Errorf("variant %q seats one value at both ends", v)
		}
	}
	if !found {
		t.Errorf("variants %q lack the expected value swap to holidays", out)
	}
}

func TestSynthesizeAnchorProtection(t *testing.T) {
	// The file seat's anchor word "file" is in a synonym
	// family (file/document) — and is never substituted:
	// it is the seat the value hangs on, not a topic.
	decl := cmdmatch.Declaration{
		Name:        "pagewipe",
		Description: "delete the first line of a file",
		Slots:       []string{"file"},
		Examples:    []string{"delete the first line of a file"},
	}
	out := synthesizeExamples(decl)
	if len(out) == 0 {
		t.Fatal("no variants for the delete entry (its verb should substitute)")
	}
	for _, v := range out {
		if strings.Contains(v, "document") {
			t.Errorf("variant %q substituted the anchor word \"file\"", v)
		}
		if !strings.Contains(v, "a file") {
			t.Errorf("variant %q lost the anchor phrase: %q", v, out)
		}
	}
}

func TestSynthesizeValueProtection(t *testing.T) {
	// A slot VALUE that happens to be a family word —
	// the folder named "photos" — is never
	// cousin-substituted in its seat: variants may swap
	// the value wholesale (the value move) or substitute
	// other words, but "from photos to" never becomes
	// "from pictures to".
	decl := cmdmatch.Declaration{
		Name:        "snaps",
		Description: "copy the files from photos to jake",
		Slots:       []string{"from", "to"},
		Examples:    []string{"copy the files from photos to jake"},
	}
	out := synthesizeExamples(decl)
	if len(out) == 0 {
		t.Fatal("no variants for the value-bearing entry")
	}
	for _, v := range out {
		for _, bad := range []string{"from pictures to", "from images to", "from pics to"} {
			if strings.Contains(v, bad) {
				t.Errorf("variant %q cousin-substituted the seated value \"photos\"", v)
			}
		}
	}
}
