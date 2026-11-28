package main

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/intent"
)

// The factorized fallback (factor.go): when the label
// read abstains, a read may be composed from the kind
// guess and the line's object words — adopted only when
// exactly one known meaning matches both. These tests
// build classifiers by hand (the Classifier's fields are
// its persisted form) so the wiring is deterministic:
// no training, no corpus, no timing.

// tableClf carries the label/kind inventory the
// derivation and the fallback read from. Its weights are
// never consulted by those paths.
func tableClf() *intent.Classifier {
	return &intent.Classifier{
		Labels: []string{
			"create_file", "explain_buffered_channel", "explain_channel",
			"explain_gadget", "explain_widget", "greeting",
		},
		Kinds: map[string]string{
			"create_file":              "make",
			"explain_buffered_channel": "find",
			"explain_channel":          "find",
			"explain_gadget":           "find",
			"explain_widget":           "find",
			"greeting":                 "talk",
		},
	}
}

func TestDeriveFactors(t *testing.T) {
	tab := deriveFactors(tableClf())
	// greeting has no object part: excluded.
	if _, ok := tab.byLabel["greeting"]; ok {
		t.Errorf("single-token label greeting was factorized: %v", tab.byLabel["greeting"])
	}
	w := tab.byLabel["explain_widget"]
	if w.kind != "find" || len(w.objects) != 1 || w.objects[0] != "widget" {
		t.Errorf("explain_widget factor = %+v, want kind find objects [widget]", w)
	}
	bc := tab.byLabel["explain_buffered_channel"]
	if bc.kind != "find" || len(bc.objects) != 2 || bc.objects[0] != "buffered" || bc.objects[1] != "channel" {
		t.Errorf("explain_buffered_channel factor = %+v, want kind find objects [buffered channel]", bc)
	}
	cf := tab.byLabel["create_file"]
	if cf.kind != "make" || len(cf.objects) != 1 || cf.objects[0] != "file" {
		t.Errorf("create_file factor = %+v, want kind make objects [file]", cf)
	}
	// A label whose rows carried no kind is excluded.
	c2 := tableClf()
	c2.Labels = append(c2.Labels, "mystery_label")
	tab2 := deriveFactors(c2)
	if _, ok := tab2.byLabel["mystery_label"]; ok {
		t.Errorf("kindless label mystery_label was factorized")
	}
}

func TestFallbackReadExactlyOne(t *testing.T) {
	d := &demo{clf: tableClf()}
	cases := []struct {
		line, kind, want string
	}{
		// The plain composition.
		{"what is a widget", "find", "explain_widget"},
		// The plural surface form counts as presence.
		{"what are widgets", "find", "explain_widget"},
		{"what are channels for", "find", "explain_channel"},
		// The compound object needs ALL its words: no
		// "buffered" in the line, so explain_channel
		// stands alone (the case above); with it, both
		// channel labels match and the read abstains.
		{"what is a buffered channel", "find", ""},
		// Two objects of the predicted kind: abstain.
		{"what is a widget and a gadget", "find", ""},
		// The kind factor disciplines the object: a file
		// is not a find, however the line is phrased.
		{"create a file", "find", ""},
		{"create a file", "make", "create_file"},
		// No kind guess, no composition.
		{"what is a widget", "", ""},
		// An object of another kind entirely: abstain.
		{"what is a widget", "talk", ""},
		// A cousin of the object counts as the object,
		// under the same kind discipline: "document" is
		// file's cousin, in its verbatim and plural forms.
		{"create a document", "make", "create_file"},
		{"create documents", "make", "create_file"},
		{"create a document", "find", ""},
		// The families' deliberate separation is the
		// control: a folder is NOT a cousin of file, so
		// the folder line composes nothing.
		{"create a folder", "make", ""},
	}
	for _, c := range cases {
		got, ok := d.fallbackRead(c.line, c.kind)
		want := c.want
		if want == "" {
			if ok {
				t.Errorf("fallbackRead(%q, %q) = %q, want abstention", c.line, c.kind, got)
			}
			continue
		}
		if !ok || got != want {
			t.Errorf("fallbackRead(%q, %q) = %q, %v; want %q", c.line, c.kind, got, ok, want)
		}
	}
}

// wireClf is a classifier whose weights make one line
// abstain at the label level (three labels tie) while
// its kind read is confident — the exact shape the
// fallback exists for — and another line read
// confidently, which the fallback must never touch.
func wireClf() *intent.Classifier {
	// Features: music=0, what=1, widget=2.
	return &intent.Classifier{
		Labels:    []string{"explain_widget", "play_music", "tell_story"},
		Words:     []string{"music", "what", "widget"},
		Weights:   [][]float64{{0, 1, 1}, {4, 1, 1}, {0, 1, 1}},
		Bias:      []float64{0, 0, 0},
		Threshold: 0.5,
		Kinds: map[string]string{
			"explain_widget": "find",
			"play_music":     "play",
			"tell_story":     "talk",
		},
		KindLabels:  []string{"find", "play", "talk"},
		KindWeights: [][]float64{{0, 2, 2}, {4, 0, 0}, {0, 0, 0}},
		KindBias:    []float64{0, 0, 0},
	}
}

func TestReadIntentFallbackAdopts(t *testing.T) {
	d := &demo{clf: wireClf()}
	r, ok := d.readIntent("what is a widget")
	if !ok {
		t.Fatal("readIntent not ok with a wired classifier")
	}
	if r.label != "explain_widget" || !r.fallback || r.kind != "find" {
		t.Errorf("read = %+v, want composed explain_widget (fallback, kind find)", r)
	}
}

func TestReadIntentConfidentUntouched(t *testing.T) {
	d := &demo{clf: wireClf()}
	r, _ := d.readIntent("music")
	if r.label != "play_music" || r.fallback {
		t.Errorf("read = %+v, want primary play_music, no fallback", r)
	}
}

// TestFallbackGistRoute: a composed read routes and
// gists exactly like a primary one — the chat route
// answers a question-shaped composed line with the
// concept's taught gist.
func TestFallbackGistRoute(t *testing.T) {
	d, out := brainsDemo(t, nil)
	d.clf = wireClf()
	d.factorTbl = nil
	d.chatGists = map[string]string{"explain_widget": "a widget is a small useful thing."}
	d.handle("what is a widget")
	got := out.String()
	if !strings.Contains(got, "chat brain: a widget is a small useful thing.") {
		t.Errorf("composed read did not gist-route:\n%s", got)
	}
}

// walkClf ranks create_file top on a line carrying file,
// gadget and widget: its parent kind (make) composes
// cleanly there, which is exactly what the walk must NOT
// reach when the first composition step was ambiguous.
func walkClf() *intent.Classifier {
	// Features: file=0, gadget=1, widget=2.
	return &intent.Classifier{
		Labels:    []string{"create_file", "explain_gadget", "explain_widget"},
		Words:     []string{"file", "gadget", "widget"},
		Weights:   [][]float64{{3, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		Bias:      []float64{0, 0, 0},
		Threshold: 0.5,
		Kinds: map[string]string{
			"create_file":    "make",
			"explain_gadget": "find",
			"explain_widget": "find",
		},
		KindLabels:  []string{"find", "make"},
		KindWeights: [][]float64{{0, 0, 0}, {0, 0, 0}},
		KindBias:    []float64{0, 0},
	}
}

func TestDeriveKindTree(t *testing.T) {
	tree := deriveKindTree(tableClf().LabelKinds())
	if got := tree.parentOf("explain_widget"); got != "find" {
		t.Errorf("parentOf(explain_widget) = %q, want find", got)
	}
	if got := tree.parentOf("greeting"); got != "talk" {
		t.Errorf("parentOf(greeting) = %q, want talk", got)
	}
	if got := tree.parentOf("no_such_label"); got != "" {
		t.Errorf("parentOf(unknown) = %q, want \"\"", got)
	}
	kids := tree.children["find"]
	want := []string{"explain_buffered_channel", "explain_channel", "explain_gadget", "explain_widget"}
	if len(kids) != len(want) {
		t.Fatalf("children[find] = %v, want %v", kids, want)
	}
	for i := range want {
		if kids[i] != want[i] {
			t.Fatalf("children[find] = %v, want sorted %v", kids, want)
		}
	}
}

func TestParentWalkAdopts(t *testing.T) {
	d := &demo{clf: wireClf()}
	// No kind guess at all: the original fallback
	// abstained here outright. The walk starts from the
	// top label (explain_widget), goes up to its parent
	// kind (find), and composes there.
	got, ok := d.fallbackRead("what is a widget", "")
	if !ok || got != "explain_widget" {
		t.Errorf("fallbackRead via parent walk = %q, %v; want explain_widget", got, ok)
	}
}

func TestParentWalkRespectsSpokenKindGuess(t *testing.T) {
	d := &demo{clf: wireClf()}
	// The kind read SPOKE (talk) and its neighbourhood
	// composed nothing. The walk does not cross into a
	// different kind to fix that — it fills a missing
	// kind read, it does not override a present one.
	// (The measured case: "what does a timer do", kind
	// guess find, top label set_timer — the unscoped
	// walk adopted set_timer for a question line.)
	got, ok := d.fallbackRead("what is a widget", "talk")
	if ok {
		t.Errorf("fallbackRead across a spoken kind guess = %q, want abstention", got)
	}
}

func TestParentWalkAbstains(t *testing.T) {
	d := &demo{clf: wireClf()}
	// The parent kind composes nothing for this line:
	// the walk changes nothing.
	if got, ok := d.fallbackRead("what", ""); ok {
		t.Errorf("fallbackRead(%q) = %q, want abstention", "what", got)
	}
	// A classifier with no weights ranks nothing: no
	// parent is named, the walk cannot start.
	d2 := &demo{clf: tableClf()}
	if got, ok := d2.fallbackRead("what is a widget", ""); ok {
		t.Errorf("fallbackRead with weightless classifier = %q, want abstention", got)
	}
}

func TestParentWalkAmbiguityIsFinal(t *testing.T) {
	d := &demo{clf: walkClf()}
	// Inside find, the line composes TWO candidates
	// (gadget and widget): ambiguity. The top label is
	// create_file, whose parent kind would compose
	// cleanly — the walk must not rescue an ambiguous
	// step by changing neighbourhoods.
	got, ok := d.fallbackRead("create a file what is a widget and a gadget", "find")
	if ok {
		t.Errorf("fallbackRead = %q, want abstention (ambiguity is final)", got)
	}
	// The same ambiguity AT the walk level: no kind
	// guess, the top label's parent kind (find) holds
	// both composed candidates — the walk abstains too.
	d2 := &demo{clf: walkClf()}
	got, ok = d2.fallbackRead("what is a widget and a gadget", "")
	if ok {
		t.Errorf("fallbackRead via walk = %q, want abstention (ambiguous in the parent kind)", got)
	}
}
