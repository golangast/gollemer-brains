package data

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeFile(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "pairs.jsonl")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadPairsGoodFile(t *testing.T) {
	path := writeFile(t, "{\"input\":\"hello\",\"output\":\"Hi there!\"}\n\n{\"input\":\"bye\",\"output\":\"Goodbye!\"}\n")
	pairs, err := LoadPairs(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 2 || pairs[0].Input != "hello" || pairs[1].Output != "Goodbye!" {
		t.Fatalf("pairs = %+v", pairs)
	}
}

func TestLoadPairsExtraFieldsIgnored(t *testing.T) {
	// Gollemer's own chat_pairs.jsonl carries a "domain" field;
	// it must drop in unchanged.
	path := writeFile(t, "{\"input\":\"hi\",\"output\":\"Hello!\",\"domain\":\"social\"}\n")
	pairs, err := LoadPairs(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 1 || pairs[0].Input != "hi" {
		t.Fatalf("pairs = %+v", pairs)
	}
}

func TestLoadPairsMalformedLineNamesTheLine(t *testing.T) {
	path := writeFile(t, "{\"input\":\"ok\",\"output\":\"fine\"}\n{\"input\": broken\n")
	_, err := LoadPairs(path)
	if err == nil || !strings.Contains(err.Error(), "line 2") {
		t.Fatalf("err = %v, want a line-2 error", err)
	}
}

func TestLoadPairsEmptyInputRejected(t *testing.T) {
	path := writeFile(t, "{\"input\":\"\",\"output\":\"fine\"}\n")
	_, err := LoadPairs(path)
	if err == nil || !strings.Contains(err.Error(), "line 1") {
		t.Fatalf("err = %v, want a line-1 error", err)
	}
}

func TestLoadPairsEmptyOutputRejected(t *testing.T) {
	path := writeFile(t, "{\"input\":\"hi\",\"output\":\"  \"}\n")
	_, err := LoadPairs(path)
	if err == nil || !strings.Contains(err.Error(), "line 1") {
		t.Fatalf("err = %v, want a line-1 error", err)
	}
}

func TestLoadPairsIntentField(t *testing.T) {
	path := writeFile(t, "{\"input\":\"turn off the lights\",\"output\":\"Lights off.\",\"intent\":\"lights_off\"}\n")
	pairs, err := LoadPairs(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 1 || pairs[0].Intent != "lights_off" || pairs[0].Output != "Lights off." {
		t.Fatalf("pairs = %+v", pairs)
	}
}

func TestLoadPairsLabelOnlyIntentRowAccepted(t *testing.T) {
	// A pure command example: input + intent, no output at
	// all (and, second row, an explicitly empty one).
	path := writeFile(t, "{\"input\":\"lights off\",\"intent\":\"lights_off\"}\n{\"input\":\"lights on\",\"output\":\"\",\"intent\":\"lights_on\"}\n")
	pairs, err := LoadPairs(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 2 || pairs[0].Intent != "lights_off" || pairs[0].Output != "" || pairs[1].Intent != "lights_on" {
		t.Fatalf("pairs = %+v", pairs)
	}
}

func TestLoadPairsOutputlessWithoutIntentRejected(t *testing.T) {
	// No output and no intent teaches nothing: malformed.
	path := writeFile(t, "{\"input\":\"hi\"}\n")
	_, err := LoadPairs(path)
	if err == nil || !strings.Contains(err.Error(), "line 1") {
		t.Fatalf("err = %v, want a line-1 error", err)
	}
}

func TestLoadPairsMalformedIntentRejected(t *testing.T) {
	path := writeFile(t, "{\"input\":\"hi\",\"output\":\"Hello!\",\"intent\":5}\n")
	_, err := LoadPairs(path)
	if err == nil || !strings.Contains(err.Error(), "line 1") {
		t.Fatalf("err = %v, want a line-1 error", err)
	}
}

func TestSplitIsDeterministic80_20(t *testing.T) {
	pairs := make([]Pair, 100)
	for i := range pairs {
		pairs[i] = Pair{Input: "q", Output: "a"}
	}
	tr1, h1 := Split(pairs)
	tr2, h2 := Split(pairs)
	if len(tr1) != 80 || len(h1) != 20 {
		t.Fatalf("split = %d/%d, want 80/20", len(tr1), len(h1))
	}
	if len(tr2) != 80 || len(h2) != 20 {
		t.Fatalf("second split = %d/%d", len(tr2), len(h2))
	}
}
