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
	// Every variant of a simple entry is one declared
	// move away from its source, never a free rewrite:
	// a substitution differs in exactly one token (one
	// cousin swapped in, or one value swapped out); the
	// shape moves are the question form (the source with
	// its "can you" dress), a rearrangement (the same
	// tokens, reordered), or the short form (a strict
	// prefix of the source).
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
	sameTokens := func(a, b string) bool {
		count := map[string]int{}
		for _, f := range strings.Fields(a) {
			count[f]++
		}
		for _, f := range strings.Fields(b) {
			count[f]--
		}
		for _, n := range count {
			if n != 0 {
				return false
			}
		}
		return true
	}
	for _, v := range out {
		switch {
		case wordDiff(decl.Description, v) == 1:
		case v == "can you "+decl.Description:
		case sameTokens(decl.Description, v):
		case strings.HasPrefix(decl.Description, v+" "):
		default:
			t.Errorf("variant %q is no declared move away from the source", v)
		}
	}
}

func TestSynthesizeSeatVariants(t *testing.T) {
	// A generic description seats no values, so the
	// value move cannot fire — the seat move rewrites
	// the generic seat phrase with pool pairs instead,
	// in pool order, skipping pool values the sentence
	// already uses as words ("photos" is this entry's
	// topic, so it is never seated as a value).
	decl := cmdmatch.Declaration{
		Name:        "photodump",
		Description: "copy the photos from one folder to another",
		Slots:       []string{"from", "to"},
		Examples:    []string{"copy the photos from one folder to another"},
	}
	out := synthesizeExamples(decl)
	for _, want := range []string{
		"copy the photos from jim to jake",
		"copy the photos from holidays to backup",
	} {
		found := false
		for _, v := range out {
			if v == want {
				found = true
			}
		}
		if !found {
			t.Errorf("variants %q lack the seated variant %q", out, want)
		}
	}
	for _, v := range out {
		if strings.Contains(v, "to photos") || strings.Contains(v, "from photos to") {
			t.Errorf("variant %q seated the topic word \"photos\" as a value", v)
		}
	}
}

func TestSynthesizeShapeVariants(t *testing.T) {
	// The shape moves on a plain from/to entry: the
	// question form, the destination-first rearrangement,
	// and the short form with the slot tail dropped.
	decl := cmdmatch.Declaration{
		Name:        "photodump",
		Description: "copy the photos from one folder to another",
		Slots:       []string{"from", "to"},
		Examples:    []string{"copy the photos from one folder to another"},
	}
	out := synthesizeExamples(decl)
	for _, want := range []string{
		"can you copy the photos from one folder to another",
		"copy the photos to another from one folder",
		"copy the photos",
	} {
		found := false
		for _, v := range out {
			if v == want {
				found = true
			}
		}
		if !found {
			t.Errorf("variants %q lack the shape variant %q", out, want)
		}
	}
}

func TestSynthesizeImperativeForm(t *testing.T) {
	// A teach-on-ask want line keeps its framing as the
	// description; the imperative under the frame is
	// among its variants.
	decl := cmdmatch.Declaration{
		Name:        "greeter",
		Description: "i want to greet the visitors at the door",
		Examples:    []string{"i want to greet the visitors at the door"},
	}
	out := synthesizeExamples(decl)
	found := false
	for _, v := range out {
		if v == "greet the visitors at the door" {
			found = true
		}
	}
	if !found {
		t.Errorf("variants %q lack the stripped imperative", out)
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
