package interpret_test

import (
	"os"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
	"github.com/golangast/gollemer-brains/data"
	"github.com/golangast/gollemer-brains/intent"
	"github.com/golangast/gollemer-brains/interpret"
)

// The library surface, tested the way an adopter uses
// it: files in, readings out. The registry and corpus
// here are small and invented for the test — a photo
// script with from/to slots, and three meanings in two
// kinds.

func testRegistry(t *testing.T) *cmdmatch.Registry {
	t.Helper()
	reg := &cmdmatch.Registry{Scripts: []cmdmatch.Declaration{
		{
			Name:        "copy-photos",
			Command:     "./copy-photos.sh",
			Description: "copies photos from one folder to another",
			Slots:       []string{"from", "to"},
			Examples:    []string{"copy the photos", "back up my pictures"},
		},
		{
			Name:        "wipe-card",
			Command:     "./wipe-card.sh",
			Description: "erases the memory card completely",
			Examples:    []string{"wipe the card", "clear the memory card"},
		},
	}}
	if err := reg.Validate(); err != nil {
		t.Fatal(err)
	}
	return reg
}

func testPairs() []data.Pair {
	var pairs []data.Pair
	add := func(label, kind string, lines ...string) {
		for _, line := range lines {
			pairs = append(pairs, data.Pair{Input: line, Intent: label, Kind: kind})
		}
	}
	add("explain_widget", "find",
		"what is a widget", "tell me about widgets", "explain the widget to me",
		"what does widget mean", "describe a widget", "widgets explained please",
		"what is a widget exactly", "can you explain widgets", "meaning of widget",
		"what are widgets for", "widget definition", "how does a widget work")
	add("play_music", "play",
		"play some music", "put on a song", "play music please",
		"start the music", "i want to hear music", "music please",
		"play a tune", "turn on some songs", "let us hear music",
		"queue up a song", "play that song again", "start playing music")
	add("cancel_music", "remove",
		"stop the music", "cancel the song", "no more music",
		"turn the music off", "stop playing songs", "kill the music",
		"end the song", "halt the music", "music off please",
		"stop that tune", "cancel playback", "silence the songs")
	return pairs
}

func testInterpreter(t *testing.T) *interpret.Interpreter {
	t.Helper()
	clf, err := intent.Train(testPairs(), intent.DefaultThreshold)
	if err != nil {
		t.Fatal(err)
	}
	return interpret.New(testRegistry(t), clf)
}

func TestInterpretCommandMatchWithSlots(t *testing.T) {
	in := testInterpreter(t)
	r := in.Interpret("copy the photos from jim to jake")
	if r.Command != "copy-photos" {
		t.Fatalf("Command = %q, want copy-photos (reading %+v)", r.Command, r)
	}
	if r.Slots["from"] != "jim" || r.Slots["to"] != "jake" {
		t.Errorf("Slots = %v, want from=jim to=jake — verbatim from the line", r.Slots)
	}
	if r.CommandScore <= 0 {
		t.Errorf("CommandScore = %v, want a positive score", r.CommandScore)
	}
	if len(r.Trace) == 0 || !strings.HasPrefix(r.Trace[0], "matcher: copy-photos") {
		t.Errorf("Trace = %v, want the matcher provenance first", r.Trace)
	}
}

func TestInterpretIntentRead(t *testing.T) {
	in := testInterpreter(t)
	r := in.Interpret("what is a widget")
	if r.Label != "explain_widget" || r.Kind != "find" {
		t.Errorf("read = %q/%q, want explain_widget/find", r.Label, r.Kind)
	}
	if r.KindGuess != "find" {
		t.Errorf("KindGuess = %q, want find", r.KindGuess)
	}
}

func TestInterpretAbstainsOnForeignLine(t *testing.T) {
	in := testInterpreter(t)
	r := in.Interpret("quantum bananas orbit saturn quietly")
	if r.Command != "" || r.Label != "" {
		t.Errorf("foreign line read as command %q label %q, want a clean abstention", r.Command, r.Label)
	}
	if len(r.Trace) == 0 {
		t.Errorf("an abstention still owes its provenance")
	}
}

func TestInterpretDeterministic(t *testing.T) {
	in := testInterpreter(t)
	a := in.Interpret("copy the photos from jim to jake")
	for i := 0; i < 20; i++ {
		b := in.Interpret("copy the photos from jim to jake")
		if b.Command != a.Command || b.Label != a.Label || b.CommandScore != a.CommandScore ||
			b.Slots["from"] != a.Slots["from"] || strings.Join(b.Trace, "\n") != strings.Join(a.Trace, "\n") {
			t.Fatalf("reading %d differs:\n%+v\nvs\n%+v", i, b, a)
		}
	}
}

func TestOpenFromFiles(t *testing.T) {
	dir := t.TempDir()
	reg := testRegistry(t)
	if err := reg.Save(dir + "/commands.json"); err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	for _, p := range testPairs() {
		sb.WriteString(`{"input": "` + p.Input + `", "intent": "` + p.Intent + `", "kind": "` + p.Kind + `"}` + "\n")
	}
	if err := os.WriteFile(dir+"/pairs.jsonl", []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	in, err := interpret.Open(dir+"/commands.json", dir+"/pairs.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if got := len(in.Commands()); got != 2 {
		t.Errorf("Commands = %d, want 2", got)
	}
	if got := len(in.Labels()); got != 3 {
		t.Errorf("Labels = %d, want 3", got)
	}
	if k := in.KindOf("play_music"); k != "play" {
		t.Errorf("KindOf(play_music) = %q, want play", k)
	}
	r := in.Interpret("stop the music")
	if r.Label != "cancel_music" {
		t.Errorf("read = %q, want cancel_music", r.Label)
	}
}

func TestOpenDeclarationsOnly(t *testing.T) {
	dir := t.TempDir()
	if err := testRegistry(t).Save(dir + "/commands.json"); err != nil {
		t.Fatal(err)
	}
	in, err := interpret.Open(dir+"/commands.json", "")
	if err != nil {
		t.Fatal(err)
	}
	r := in.Interpret("copy the photos from jim to jake")
	if r.Command != "copy-photos" || r.Label != "" {
		t.Errorf("declarations-only reading = %+v, want the command and no label", r)
	}
	if _, err := interpret.Open("", ""); err == nil {
		t.Errorf("Open with no sources = nil error, want an error")
	}
}

func TestExplainProvenance(t *testing.T) {
	in := testInterpreter(t)
	hits := in.Explain("copy the photos from jim to jake", "copy-photos")
	var vias []string
	for _, h := range hits {
		if h.Via != "" {
			vias = append(vias, h.Word+":"+h.Via)
		}
	}
	if len(vias) == 0 {
		t.Fatalf("Explain found no provenance in %v", hits)
	}
	if got := in.Explain("anything", "no-such-command"); got != nil {
		t.Errorf("Explain of an unknown command = %v, want nil", got)
	}
}

// TestFallbackThroughSurface: a line the primary read
// leaves behind can still be composed by the factorized
// fallback, and the Reading says so. The classifier is
// the hand-built one the fallback's own tests use: three
// labels tie on the widget line (label abstention) while
// the kind read is confident.
func TestFallbackThroughSurface(t *testing.T) {
	clf := &intent.Classifier{
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
	in := interpret.New(nil, clf)
	r := in.Interpret("what is a widget")
	if r.Label != "explain_widget" || !r.Fallback {
		t.Errorf("read = %+v, want the fallback-composed explain_widget", r)
	}
}
