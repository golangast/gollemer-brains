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
