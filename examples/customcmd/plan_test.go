package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// planDemo builds a demo over a registry file in a temp
// dir whose extractor answers "jim jake" for lines
// naming jim and nothing otherwise, and whose nextLine
// serves the given answers in order.
func planDemo(t *testing.T, path string, reg *cmdmatch.Registry, answers []string) (*demo, *bytes.Buffer) {
	t.Helper()
	var out bytes.Buffer
	d := &demo{
		registryPath: path,
		reg:          reg,
		matcher:      cmdmatch.NewMatcher(reg),
		extract: func(line string) (string, error) {
			if strings.Contains(line, "jim") {
				return "jim jake", nil
			}
			return "", nil
		},
		nextLine: func(prompt string) (string, bool) {
			if len(answers) == 0 {
				return "", false
			}
			a := answers[0]
			answers = answers[1:]
			return a, true
		},
		out: &out,
	}
	return d, &out
}

func TestSplitSegments(t *testing.T) {
	for line, want := range map[string][]string{
		"migrate a database from folder jim to jake": {"migrate a database from folder jim to jake"},
		"back up my photos, migrate the database, and clean the downloads": {
			"back up my photos", "migrate the database", "clean the downloads"},
		"back up the config and then run the checks": {"back up the config", "run the checks"},
		"back up the config then run the checks":     {"back up the config", "run the checks"},
		"back up my photos and sync my notes":        {"back up my photos", "sync my notes"},
		"then back up my photos,":                    {"back up my photos"},
		"how are you":                                {"how are you"},
	} {
		got := splitSegments(line)
		if strings.Join(got, "|") != strings.Join(want, "|") {
			t.Errorf("splitSegments(%q) = %v, want %v", line, got, want)
		}
	}
}

func TestStageRank(t *testing.T) {
	if !(stageRank("inspect") < stageRank("prepare") &&
		stageRank("prepare") < stageRank("execute") &&
		stageRank("execute") < stageRank("verify") &&
		stageRank("verify") < stageRank("cleanup")) {
		t.Error("stage ranks out of order")
	}
	if stageRank("") != stageRank("execute") || stageRank("bogus") != stageRank("execute") {
		t.Error("unstaged and unknown stages must rank as execute")
	}
}

// TestOrderStepsAfterBeatsTypedOrder: two steps of the
// same stage, the second typed first but declared After
// the first — the prerequisite wins.
func TestOrderStepsAfterBeatsTypedOrder(t *testing.T) {
	steps := []planStep{
		{segment: "publish the report", matched: true, decl: cmdmatch.Declaration{Name: "publish-report", Stage: "execute", After: []string{"gather-stats"}}},
		{segment: "gather the stats", matched: true, decl: cmdmatch.Declaration{Name: "gather-stats", Stage: "execute"}},
	}
	ordered, cycle := orderSteps(steps)
	if cycle {
		t.Fatal("no cycle here")
	}
	if ordered[0].decl.Name != "gather-stats" || ordered[1].decl.Name != "publish-report" {
		t.Errorf("after edge ignored: %v", []string{ordered[0].decl.Name, ordered[1].decl.Name})
	}
}

// TestOrderStepsCycle: two steps declared after each
// other — the ordering must terminate, keep both steps,
// and report the cycle rather than loop.
func TestOrderStepsCycle(t *testing.T) {
	steps := []planStep{
		{segment: "a", matched: true, decl: cmdmatch.Declaration{Name: "a", After: []string{"b"}}},
		{segment: "b", matched: true, decl: cmdmatch.Declaration{Name: "b", After: []string{"a"}}},
	}
	ordered, cycle := orderSteps(steps)
	if !cycle {
		t.Error("cycle not reported")
	}
	if len(ordered) != 2 {
		t.Errorf("cycle lost steps: %d", len(ordered))
	}
	if notes := planNotes(ordered, cycle); !strings.Contains(strings.Join(notes, "\n"), "form a cycle") {
		t.Errorf("cycle note missing: %v", notes)
	}
}

// TestPlanOutOfOrderSession: a plan typed cleanup-first
// comes out in stage order, with jim and jake bound in
// their step.
func TestPlanOutOfOrderSession(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := planDemo(t, path, reg, nil)
	d.handle("compress the logs, back up my photos and migrate a database from folder jim to jake")
	got := out.String()
	for _, want := range []string{
		"plan: 3 steps (3 ready, 0 gaps)",
		"[ ] 1. Prepare — concept: backup-photos",
		"    Target: files",
		"    preview: backs up my photo library to the external drive",
		"[ ] 2. Execute — concept: migrate-database",
		"    Target: database",
		"    preview: migrates a database from jim to jake",
		"    Would run: ./scripts/migrate-db.sh jim jake",
		"[ ] 3. Cleanup — concept: compress-logs",
		"    Target: logs",
		"nothing has run yet.",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("plan output missing %q:\n%s", want, got)
		}
	}
	if strings.Index(got, "[ ] 1. Prepare") > strings.Index(got, "[ ] 3. Cleanup") {
		t.Errorf("steps not in stage order:\n%s", got)
	}
}

// TestPlanGapTeachRerender: a plan with one unknown
// segment shows the gap, offers to teach it once, and —
// taught — prints the plan again with the gap filled.
func TestPlanGapTeachRerender(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := planDemo(t, path, reg, []string{"./scripts/water-plants.sh"})
	d.handle("back up my photos and water the greenhouse plants")
	got := out.String()
	for _, want := range []string{
		"plan: 2 steps (1 ready, 1 gap)",
		"? — no command knows this step yet: 'water the greenhouse plants'",
		"learned: when you say \"water the greenhouse plants\"",
		"plan: 2 steps (2 ready, 0 gaps)",
		"water-plants",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("gap/teach output missing %q:\n%s", want, got)
		}
	}
	if strings.Count(got, "plan: 2 steps") != 2 {
		t.Errorf("plan should print twice (before and after teaching):\n%s", got)
	}
}

// TestPlanTeachSkipStands: pressing enter at the gap's
// teach offer leaves the plan as printed — no second
// render, nothing saved.
func TestPlanTeachSkipStands(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := planDemo(t, path, reg, []string{""})
	before := len(reg.Scripts)
	d.handle("back up my photos and water the greenhouse plants")
	if strings.Count(out.String(), "plan: 2 steps") != 1 {
		t.Errorf("skipped teach should not re-render:\n%s", out.String())
	}
	back, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(back.Scripts) != before {
		t.Error("skipped teach wrote to the registry")
	}
}

// TestPlanVerifyWithoutExecute: a verify step with no
// execute step in the plan earns the missing-link note.
func TestPlanVerifyWithoutExecute(t *testing.T) {
	reg := &cmdmatch.Registry{Scripts: []cmdmatch.Declaration{
		{Name: "check-tests", Command: "./check.sh", Description: "checks the test results and reports failures", Stage: "verify"},
		{Name: "stash-photos", Command: "./stash.sh", Description: "stashes the photos on the spare drive", Stage: "prepare"},
	}}
	dir := t.TempDir()
	path := filepath.Join(dir, "commands.json")
	if err := reg.Save(path); err != nil {
		t.Fatal(err)
	}
	reg, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	d, out := planDemo(t, path, reg, nil)
	d.handle("check the tests and stash the photos")
	got := out.String()
	for _, want := range []string{
		"plan: 2 steps (2 ready, 0 gaps)",
		"[ ] 1. Prepare — concept: stash-photos",
		"[ ] 2. Verify — concept: check-tests",
		"verify with no execute step — nothing to check the result of.",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("verify-note output missing %q:\n%s", want, got)
		}
	}
}

// TestPlanAfterMissingNote: an After prerequisite that
// no step in the plan provides is called out by name.
func TestPlanAfterMissingNote(t *testing.T) {
	reg := &cmdmatch.Registry{Scripts: []cmdmatch.Declaration{
		{Name: "publish-report", Command: "./publish.sh", Description: "publishes the stats report to the team page", Stage: "execute", After: []string{"gather-stats"}},
		{Name: "stash-photos", Command: "./stash.sh", Description: "stashes the photos on the spare drive", Stage: "prepare"},
	}}
	dir := t.TempDir()
	path := filepath.Join(dir, "commands.json")
	if err := reg.Save(path); err != nil {
		t.Fatal(err)
	}
	reg, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	d, out := planDemo(t, path, reg, nil)
	d.handle("publish the report and stash the photos")
	got := out.String()
	if !strings.Contains(got, "note: publish-report is declared after gather-stats, but no step in this plan is gather-stats.") {
		t.Errorf("missing-after note absent:\n%s", got)
	}
}

// stubDemo builds a demo over the given scripts (saved
// to a temp registry) whose extractor answers by
// substring: a segment naming cat.txt hears "cat.txt",
// one naming trips hears "trips", one speaking of a
// first line hears "nina", and a rename segment hears
// "school" — the last two are favourites the segment
// never named, the live extractor's documented habit,
// which the context tests need in the room. Anything
// else hears nothing. That is the control the context
// tests need: exactly which segment stated a value is
// known.
func stubDemo(t *testing.T, scripts []cmdmatch.Declaration, answers []string) (*demo, *bytes.Buffer) {
	t.Helper()
	reg := &cmdmatch.Registry{Scripts: scripts}
	dir := t.TempDir()
	path := filepath.Join(dir, "commands.json")
	if err := reg.Save(path); err != nil {
		t.Fatal(err)
	}
	reg, err := cmdmatch.LoadRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	d := &demo{
		registryPath: path,
		reg:          reg,
		matcher:      cmdmatch.NewMatcher(reg),
		extract: func(line string) (string, error) {
			switch {
			case strings.Contains(line, "cat.txt"):
				return "cat.txt", nil
			case strings.Contains(line, "trips"):
				return "trips", nil
			case strings.Contains(line, "first line"):
				return "nina", nil
			case strings.Contains(line, "rename"):
				return "school", nil
			}
			return "", nil
		},
		nextLine: func(prompt string) (string, bool) {
			if len(answers) == 0 {
				return "", false
			}
			a := answers[0]
			answers = answers[1:]
			return a, true
		},
		out: &out,
	}
	return d, &out
}

var sortScript = cmdmatch.Declaration{
	Name: "sort-by-size", Command: "ls -lhS {folder}",
	Description: "sorts the files in a folder by size, largest first",
	Slots:       []string{"folder"}, Stage: "inspect", ActsOn: "files",
}

var delScript = cmdmatch.Declaration{
	Name: "del", Command: "./del.sh",
	Description: "deletes the first line of a file",
	Slots:       []string{"file"}, Stage: "execute",
}

// TestPlanInheritsSameSlot: a step whose segment never
// stated its folder inherits the value an earlier step
// bound to the same slot name, marked "(from step 1)"
// in its Target, its preview and (as if spoken) its
// Would run. The stub extractor offers the rename
// segment a favourite ("school") it never named: the
// grounded value from step 1 outranks the guess.
func TestPlanInheritsSameSlot(t *testing.T) {
	reg := []cmdmatch.Declaration{sortScript, {
		Name: "rename-photos", Command: "./scripts/rename-photos.sh",
		Description: "renames the photos in a folder from camera names to dates",
		Slots:       []string{"folder"}, Stage: "execute", ActsOn: "files",
	}}
	d, out := stubDemo(t, reg, nil)
	d.handle("sort the files in trips by size and rename the photos in the folder")
	got := out.String()
	for _, want := range []string{
		"[ ] 1. Inspect — concept: sort-by-size",
		"    Target: files in trips",
		"[ ] 2. Execute — concept: rename-photos",
		"    Target: files in trips (from step 1)",
		"    preview: renames the photos in the folder trips (from step 1) from camera names to dates",
		"    Would run: ./scripts/rename-photos.sh trips",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("inheritance output missing %q:\n%s", want, got)
		}
	}
}

// TestPlanEachFileStep: a file-slot step whose segment
// speaks of "the files", in a plan whose context
// supplies a folder, applies to each file of that
// folder — Target, Would run and (the concept deletes)
// the caution all say so. The stub extractor offers
// "nina" for the delete segment, a favourite it never
// named: that is not a named target, so it neither
// defeats the each-file reading nor rides into the
// rendered step.
func TestPlanEachFileStep(t *testing.T) {
	d, out := stubDemo(t, []cmdmatch.Declaration{sortScript, delScript}, nil)
	d.handle("sort the files in trips by size and then delete the first line in the files")
	got := out.String()
	for _, want := range []string{
		"[ ] 1. Inspect — concept: sort-by-size",
		"[ ] 2. Execute — concept: del",
		"    Target: each file in trips (from step 1)",
		"    Would run: ./del.sh — on each file in trips",
		"    careful: this command removes or changes things.",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("each-file output missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "nina") {
		t.Errorf("extractor favourite rode into the each-file step:\n%s", got)
	}
}

// TestPlanEachFileDefeatedByNamedFile: a segment that
// names one concrete file is a single target — no
// each-file reading, and the file binds as itself.
func TestPlanEachFileDefeatedByNamedFile(t *testing.T) {
	d, out := stubDemo(t, []cmdmatch.Declaration{sortScript, delScript}, nil)
	d.handle("sort the files in trips by size and delete the first line of the file cat.txt")
	got := out.String()
	for _, want := range []string{
		"    Target: file cat.txt",
		"    Would run: ./del.sh cat.txt",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("named-file output missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "on each file") {
		t.Errorf("named file read as each-file:\n%s", got)
	}
}

func TestStageLabel(t *testing.T) {
	for stage, want := range map[string]string{
		"": "Execute", "execute": "Execute", "inspect": "Inspect",
		" verify ": "Verify", "CLEANUP": "Cleanup", "prepare": "Prepare",
	} {
		if got := stageLabel(stage); got != want {
			t.Errorf("stageLabel(%q) = %q, want %q", stage, got, want)
		}
	}
}

// TestPlanNotForSingleOrUnmatchedLines: plan mode must
// not swallow the ordinary paths — a single want keeps
// the single-match block, and a multi-segment line whose
// segments match nothing keeps the whole-line abstention.
func TestPlanNotForSingleOrUnmatchedLines(t *testing.T) {
	path, reg := tempShippedRegistry(t)
	d, out := planDemo(t, path, reg, nil)
	d.handle("migrate a database from folder jim to jake")
	if strings.Contains(out.String(), "plan:") {
		t.Errorf("single want became a plan:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "understood: migrate-database") {
		t.Errorf("single want lost its block:\n%s", out.String())
	}
	out.Reset()
	d.handle("how are you and how is the weather")
	if strings.Contains(out.String(), "plan:") {
		t.Errorf("talk became a plan:\n%s", out.String())
	}
	if out.String() != "" {
		t.Errorf("unmatched multi-segment line should get silence, not a plan or a match:\n%s", out.String())
	}
}
