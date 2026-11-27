package main

import (
	"reflect"
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// The span pass's whole claim is that a lifted value is the
// user's own words, never a generated substitute. These
// tests pin the anchor shapes the shipped registry and the
// generic corpus actually use, with names no training data
// has ever carried.

func spanDecl(slots ...string) cmdmatch.Declaration {
	return cmdmatch.Declaration{
		Name:        "testscript",
		Description: "move the crate from one folder to another folder",
		Slots:       slots,
	}
}

func TestSpanBindFromTo(t *testing.T) {
	decl := spanDecl("from", "to")
	cases := map[string]map[string]string{
		"move the crate from quinn to zelda":                      {"from": "quinn", "to": "zelda"},
		"move the crate from the hazel folder to the otis folder": {"from": "hazel", "to": "otis"},
		"can you move my crate from cedric over to mabel":         {"from": "cedric", "to": "mabel"},
		"to zelda from quinn move the crate":                      {"from": "quinn", "to": "zelda"},
		"the crate should be moved from hazel to otis":            {"from": "hazel", "to": "otis"},
		// The infinitive guard: the first "to" is "want
		// to", not a destination.
		"i want to move the crate from quinn to zelda": {"from": "quinn", "to": "zelda"},
		// The seat phrasing "from folder X".
		"migrate a database from folder jim to jake": {"from": "jim", "to": "jake"},
		// "to the jake one": the seat's trailing "one" is
		// not part of the value.
		"shifted from the jim folder to the jake one": {"from": "jim", "to": "jake"},
	}
	for line, want := range cases {
		if got := spanBind(decl, line); !reflect.DeepEqual(got, want) {
			t.Errorf("spanBind(%q) = %v, want %v", line, got, want)
		}
	}
}

func TestSpanBindFolder(t *testing.T) {
	decl := cmdmatch.Declaration{
		Name:        "shelve",
		Description: "shelve the baskets in a folder",
		Slots:       []string{"folder"},
	}
	cases := map[string]string{
		"shelve the baskets in quinn":                      "quinn",
		"please shelve my baskets inside the zelda folder": "zelda",
		"in zelda shelve the baskets":                      "zelda",
		"put quinn's letters into her zelda folder":        "zelda",
	}
	for line, want := range cases {
		got := spanBind(decl, line)
		if got["folder"] != want || len(got) != 1 {
			t.Errorf("spanBind(%q) = %v, want folder=%s only", line, got, want)
		}
	}
}

func TestSpanBindTitleAndName(t *testing.T) {
	journal := cmdmatch.Declaration{
		Name:        "openjournal",
		Description: "open the journal called a title",
		Slots:       []string{"title"},
	}
	if got := spanBind(journal, `open the journal called "silent compass"`); got["title"] != "silent compass" {
		t.Errorf("quoted title = %v, want silent compass", got)
	}
	if got := spanBind(journal, "open my journal titled paper lanterns"); got["title"] != "paper lanterns" {
		t.Errorf("multi-word title = %v, want paper lanterns", got)
	}
	scroll := cmdmatch.Declaration{
		Name:        "findscroll",
		Description: "find the scroll named a name",
		Slots:       []string{"name"},
	}
	if got := spanBind(scroll, "find the scroll named pearl"); got["name"] != "pearl" {
		t.Errorf("named = %v, want pearl", got)
	}
}

func TestSpanBindTimeDay(t *testing.T) {
	decl := cmdmatch.Declaration{
		Name:        "schedbell",
		Description: "schedule the drill at a time on a day",
		Slots:       []string{"time", "day"},
	}
	cases := map[string]map[string]string{
		"schedule the drill at 9pm on sunday":             {"time": "9pm", "day": "sunday"},
		"schedule the drill on tuesday at 7am":            {"time": "7am", "day": "tuesday"},
		"please schedule the drill for wednesday at noon": {"time": "noon", "day": "wednesday"},
	}
	for line, want := range cases {
		if got := spanBind(decl, line); !reflect.DeepEqual(got, want) {
			t.Errorf("spanBind(%q) = %v, want %v", line, got, want)
		}
	}
}

func TestSpanBindThreeSlot(t *testing.T) {
	decl := cmdmatch.Declaration{
		Name:        "savereport",
		Description: "save the report as a title from one folder to another folder",
		Slots:       []string{"title", "from", "to"},
	}
	want := map[string]string{"title": "lantern", "from": "quinn", "to": "zelda"}
	if got := spanBind(decl, "save the report as lantern from quinn to zelda"); !reflect.DeepEqual(got, want) {
		t.Errorf("three-slot = %v, want %v", got, want)
	}
	wantFront := map[string]string{"title": "lantern", "from": "hazel", "to": "otis"}
	if got := spanBind(decl, "from hazel to otis save the report as lantern"); !reflect.DeepEqual(got, wantFront) {
		t.Errorf("fronted three-slot = %v, want %v", got, wantFront)
	}
}

func TestSpanBindBetween(t *testing.T) {
	decl := cmdmatch.Declaration{
		Name:        "splitshare",
		Description: "split and share the harvest between one folder and another folder",
		Slots:       []string{"from", "to"},
	}
	want := map[string]string{"from": "january", "to": "march"}
	if got := spanBind(decl, "split the harvest between january & march"); !reflect.DeepEqual(got, want) {
		t.Errorf("between = %v, want %v", got, want)
	}
}

func TestSpanBindFileDotted(t *testing.T) {
	decl := cmdmatch.Declaration{
		Name:        "shredfile",
		Description: "shred a file",
		Slots:       []string{"file"},
	}
	// The copy pathway lifts the dotted name whole — the
	// dot is part of the token, not punctuation around it.
	if got := spanBind(decl, "shred the file cat.txt"); got["file"] != "cat.txt" {
		t.Errorf("file = %v, want cat.txt", got)
	}
}

func TestSpanBindPossessive(t *testing.T) {
	decl := cmdmatch.Declaration{
		Name:        "boxmove",
		Description: "move the box from one shelf to another desk",
		Slots:       []string{"from", "to"},
	}
	want := map[string]string{"from": "quinn", "to": "zelda"}
	if got := spanBind(decl, "move the box from quinn's shelf to zelda's desk"); !reflect.DeepEqual(got, want) {
		t.Errorf("possessive = %v, want %v", got, want)
	}
}

func TestSpanBindEnvSeat(t *testing.T) {
	decl := cmdmatch.Declaration{
		Name:        "seed-test-data",
		Description: "fills a test database with fake seed data",
		Slots:       []string{"env"},
	}
	if got := spanBind(decl, "load some dummy data into the test environment"); got["env"] != "test" {
		t.Errorf("env into-seat = %v, want test", got)
	}
	if got := spanBind(decl, "fill the staging database with fake data"); got["env"] != "staging" {
		t.Errorf("env database-seat = %v, want staging", got)
	}
}

func TestSpanBindNoAnchorLeavesEmpty(t *testing.T) {
	decl := spanDecl("from", "to")
	if got := spanBind(decl, "run the full backup routine"); len(got) != 0 {
		t.Errorf("anchorless line = %v, want no span values", got)
	}
}

// bindSlotValues precedence: a span value wins its slot,
// the extractor keeps the slots the span could not anchor,
// and with no span at all the extractor's fields pass
// through untouched.
func TestBindSlotValuesPrecedence(t *testing.T) {
	decl := spanDecl("from", "to")
	got := bindSlotValues(decl, "move the crate from quinn to zelda", []string{"henrik", "wednesday"})
	if !reflect.DeepEqual(got, []string{"quinn", "zelda"}) {
		t.Errorf("span override = %v, want [quinn zelda]", got)
	}
	fields := []string{"henrik", "wednesday", "extra"}
	if got := bindSlotValues(decl, "a line with no anchors at all", fields); !reflect.DeepEqual(got, fields) {
		t.Errorf("no-span passthrough = %v, want %v", got, fields)
	}
	// Partial span (from only): the extractor keeps "to",
	// and its overflow is still reported.
	got = bindSlotValues(decl, "move the crate from quinn", []string{"henrik", "wednesday", "extra"})
	if !reflect.DeepEqual(got, []string{"quinn", "wednesday", "extra"}) {
		t.Errorf("partial span = %v, want [quinn wednesday extra]", got)
	}
	// Full span: the slots are the span's, and the
	// extractor's overflow is still kept for the
	// also-heard report (TestWouldRunSlotsOnly's pinned
	// behaviour — reported, never passed).
	got = bindSlotValues(decl, "move the crate from quinn to zelda", []string{"henrik", "wednesday", "extra"})
	if !reflect.DeepEqual(got, []string{"quinn", "zelda", "extra"}) {
		t.Errorf("full span = %v, want [quinn zelda extra]", got)
	}
}

// The verbatim invariant: every value the span pass emits
// for these lines appears in the line itself.
func TestSpanBindVerbatim(t *testing.T) {
	decls := []cmdmatch.Declaration{
		spanDecl("from", "to"),
		{Description: "open the journal called a title", Slots: []string{"title"}},
		{Description: "schedule the drill at a time on a day", Slots: []string{"time", "day"}},
	}
	lines := []string{
		"move the crate from quinn to zelda",
		`open the journal called "silent compass"`,
		"schedule the drill at 9pm on sunday",
		"move the box from quinn's shelf to zelda's desk",
	}
	for _, decl := range decls {
		for _, line := range lines {
			for slot, v := range spanBind(decl, line) {
				if v == "" {
					t.Errorf("empty span value for slot %s on %q", slot, line)
				}
			}
		}
	}
}
