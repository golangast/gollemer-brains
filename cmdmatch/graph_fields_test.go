package cmdmatch

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestNeedsAndStepsRoundTrip: the concept-graph fields
// (Needs, Steps) survive a JSON round trip like the
// other plan metadata, and stay absent — not empty —
// when unset, so existing registries marshal exactly as
// before.
func TestNeedsAndStepsRoundTrip(t *testing.T) {
	r := &Registry{Scripts: []Declaration{
		{Name: "migrate-database", Command: "./m.sh", Description: "migrates a database", Needs: []string{"backup-photos"}},
		{Name: "full-backup", Description: "the whole backup routine", Steps: []string{"back up my photos", "migrate the database"}},
		{Name: "plain", Command: "./p.sh", Description: "a plain script"},
	}}
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var back Registry
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if got := back.Scripts[0].Needs; len(got) != 1 || got[0] != "backup-photos" {
		t.Errorf("needs did not round-trip: %v", got)
	}
	if got := back.Scripts[1].Steps; len(got) != 2 || got[0] != "back up my photos" {
		t.Errorf("steps did not round-trip: %v", got)
	}
	if strings.Contains(string(raw), `"plain","command":"./p.sh","description":"a plain script","needs"`) ||
		strings.Contains(string(raw), `"steps":[]`) {
		t.Errorf("unset graph fields leaked into the JSON: %s", raw)
	}
}

// TestRoutineValidation: an entry with Steps is a
// routine — no command required, but a command alongside
// steps is a contradiction, and an empty step is a
// defect. A script without steps still needs its
// command, exactly as before.
func TestRoutineValidation(t *testing.T) {
	routine := Declaration{Name: "r", Description: "a routine", Steps: []string{"do one thing", "do another"}}
	if err := (&Registry{Scripts: []Declaration{routine}}).Validate(); err != nil {
		t.Errorf("plain routine should validate: %v", err)
	}
	both := routine
	both.Command = "./r.sh"
	if err := (&Registry{Scripts: []Declaration{both}}).Validate(); err == nil {
		t.Error("routine with both steps and a command should not validate")
	}
	emptyStep := routine
	emptyStep.Steps = []string{"do one thing", " "}
	if err := (&Registry{Scripts: []Declaration{emptyStep}}).Validate(); err == nil {
		t.Error("routine with an empty step should not validate")
	}
	noCommand := Declaration{Name: "s", Description: "a script"}
	if err := (&Registry{Scripts: []Declaration{noCommand}}).Validate(); err == nil {
		t.Error("script with no command and no steps should not validate")
	}
}
