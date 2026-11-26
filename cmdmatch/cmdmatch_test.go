package cmdmatch

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testRegistry() *Registry {
	return &Registry{Scripts: []Declaration{
		{
			Name:        "migrate-database",
			Command:     "./scripts/migrate-db.sh",
			Description: "migrates a database from one folder to another",
			Slots:       []string{"from", "to"},
			Examples:    []string{"migrate the database from jim to jake", "run the database migration"},
		},
		{
			Name:        "backup-photos",
			Command:     "./scripts/backup-photos.sh",
			Description: "backs up my photo library to the external drive",
			Examples:    []string{"back up my photos", "make a backup of my photos"},
		},
		{
			Name:        "water-plants",
			Command:     "./scripts/water-plants.sh",
			Description: "logs watering the greenhouse plants and notes the time",
		},
	}}
}

func TestMatchByDescriptionWords(t *testing.T) {
	m := NewMatcher(testRegistry())
	// Words from the description, in a phrasing nobody wrote:
	// the mechanism the whole design rests on.
	d, score, ok := m.Match("move my database to a different folder")
	if !ok || d.Name != "migrate-database" {
		t.Errorf("Match = %q, %.2f, %v; want migrate-database", d.Name, score, ok)
	}
	// A script declared with a description only (no
	// examples) matches a description-style line: the
	// zero-retrain property in miniature.
	d, _, ok = m.Match("water the greenhouse plants and log it")
	if !ok || d.Name != "water-plants" {
		t.Errorf("Match = %q, %v; want water-plants", d.Name, ok)
	}
}

func TestMatchAbstainsOnTalk(t *testing.T) {
	m := NewMatcher(testRegistry())
	for _, line := range []string{
		"how are you doing today",
		"thank you so much",
		"good night, see you tomorrow",
		"i took some photos at the park yesterday",
	} {
		if d, score, ok := m.Match(line); ok {
			t.Errorf("Match(%q) = %q (%.2f); want an abstention", line, d.Name, score)
		}
	}
}

func TestTypoRepair(t *testing.T) {
	m := NewMatcher(testRegistry())
	// "databse" is in no profile; its unique distance-1
	// neighbour in the registry vocabulary is "databas"
	// (stemmed "database"), so the line still matches.
	d, _, ok := m.Match("migrate the databse")
	if !ok || d.Name != "migrate-database" {
		t.Errorf("Match with typo = %q, %v; want migrate-database", d.Name, ok)
	}
	if !editDistance1("databse", "databas") && !editDistance1("databse", "database") {
		t.Error("editDistance1 should see the transposition")
	}
	if editDistance1("database", "database") {
		t.Error("editDistance1 must be exactly one edit, not zero")
	}
	if editDistance1("abc", "xyz") {
		t.Error("editDistance1 accepted three substitutions")
	}
}

func TestStemConsistency(t *testing.T) {
	pairs := map[string]string{
		"migrates": "migrate",
		"copies":   "copy",
		"backed":   "back",
		"logs":     "log",
		"updates":  "update",
		"notes":    "note",
	}
	for a, b := range pairs {
		if stem(a) != stem(b) {
			t.Errorf("stem(%q) = %q, stem(%q) = %q; want the same form", a, stem(a), b, stem(b))
		}
	}
}

// expansionRegistry mirrors the shipped registry entries
// the near-miss lines were measured against, plus their
// nearest neighbours.
func expansionRegistry() *Registry {
	return &Registry{Scripts: []Declaration{
		{
			Name:        "rename-photos",
			Command:     "./scripts/rename-photos.sh",
			Description: "renames the photos in a folder from camera names to dates",
			Slots:       []string{"folder"},
			Examples:    []string{"rename the photos in holidays", "rename the pictures in the family folder", "rename my photos", "fix the photo names in trips"},
		},
		{
			Name:        "clean-downloads",
			Command:     "./scripts/clean-downloads.sh",
			Description: "cleans out the downloads folder and moves old files to the trash",
			Examples:    []string{"clean out my downloads", "tidy up my downloads"},
		},
		{
			Name:        "backup-photos",
			Command:     "./scripts/backup-photos.sh",
			Description: "backs up my photo library to the external drive",
			Examples:    []string{"back up my photos", "copy my photos to the external drive"},
		},
	}}
}

func TestSynonymExpansion(t *testing.T) {
	m := NewMatcher(expansionRegistry())
	// The two live near-miss lines: the description says
	// "renames the photos", the user said "changes file
	// names" — no shared content word but "names" until
	// change joined rename's family.
	for _, line := range []string{
		"the script that changes file names for me",
		"change all the file names in birthdays",
	} {
		d, score, ok := m.Match(line)
		if !ok || d.Name != "rename-photos" {
			t.Errorf("Match(%q) = %q, %.2f, %v; want rename-photos via expansion", line, d.Name, score, ok)
		}
	}
	// A cousin verb for a different script: erase is
	// clean's cousin, and junk is nobody's word.
	d, _, ok := m.Match("erase the junk cluttering my downloads")
	if !ok || d.Name != "clean-downloads" {
		t.Errorf("Match(erase line) = %q, %v; want clean-downloads", d.Name, ok)
	}
	// Cousins widen recall without inventing scripts:
	// there is no list script, and folder's cousins must
	// not conjure one — one matched word is not a match.
	if d, score, ok := m.Match("show me what's inside this directory"); ok {
		t.Errorf("Match(directory line) = %q (%.2f); want an abstention", d.Name, score)
	}
}

func TestRepairDoesNotStealCousins(t *testing.T) {
	// "pack" is one edit from the registry word "back"
	// and a cousin of "compress". Repair must not rewrite
	// it before expansion can claim it — with "back" in
	// the vocabulary, the theft is there to happen.
	reg := &Registry{Scripts: []Declaration{
		{
			Name:        "compress-logs",
			Command:     "./scripts/compress-logs.sh",
			Description: "compresses old log files into a single archive",
			Examples:    []string{"compress the logs", "zip up the logs"},
		},
		{
			Name:        "backup-photos",
			Command:     "./scripts/backup-photos.sh",
			Description: "backs up my photo library to the external drive",
			Examples:    []string{"back up my photos"},
		},
	}}
	m := NewMatcher(reg)
	d, _, ok := m.Match("pack the logs into one bundle")
	if !ok || d.Name != "compress-logs" {
		t.Errorf("Match(pack line) = %q, %v; want compress-logs via the pack cousin", d.Name, ok)
	}
}

func TestSynonymCountsLessThanExact(t *testing.T) {
	m := NewMatcher(expansionRegistry())
	scoreOf := func(line, name string) float64 {
		for _, s := range m.Rank(line) {
			if s.Script.Name == name {
				return s.Score
			}
		}
		return -1
	}
	// Same line, one word swapped for its cousin: the
	// cousin version must score lower for the script —
	// expansion borrows evidence, it does not mint it.
	exact := scoreOf("rename the photos", "rename-photos")
	cousin := scoreOf("retitle the photos", "rename-photos")
	if !(cousin > 0 && cousin < exact) {
		t.Errorf("cousin score %.2f should be positive and below exact %.2f", cousin, exact)
	}
}

func TestSynonymFamiliesDisjoint(t *testing.T) {
	owner := map[string]int{}
	for i, fam := range synonymFamilies {
		if len(fam) < 2 {
			t.Errorf("family %d has fewer than two words", i)
		}
		for _, w := range fam {
			s := stem(w)
			if stopwords[w] {
				t.Errorf("family %d contains stopword %q, which can never match", i, w)
			}
			if j, dup := owner[s]; dup && j != i {
				t.Errorf("stemmed word %q (from %q) appears in families %d and %d", s, w, j, i)
			}
			owner[s] = i
		}
	}
}

func TestRegistryRoundTripAndValidation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "commands.json")
	r := testRegistry()
	if err := r.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	back, err := LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	if len(back.Scripts) != 3 || back.Scripts[0].Name != "migrate-database" {
		t.Errorf("round trip lost scripts: %v", back.sortedNames())
	}
	// Add works, duplicates do not.
	if err := back.Add(Declaration{Name: "x", Command: "./x.sh", Description: "does x things"}); err != nil {
		t.Errorf("Add: %v", err)
	}
	if err := back.Add(Declaration{Name: "x", Command: "./x.sh", Description: "does x things"}); err == nil {
		t.Error("Add accepted a duplicate name")
	}
	if err := back.Add(Declaration{Name: "nodesc", Command: "./n.sh"}); err == nil {
		t.Error("Add accepted a script with no description")
	}
	// A registry file with a duplicate name is rejected.
	dup := `{"scripts": [
		{"name": "a", "command": "./a.sh", "description": "does a"},
		{"name": "a", "command": "./a.sh", "description": "does a again"}
	]}`
	bad := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(bad, []byte(dup), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadRegistry(bad); err == nil {
		t.Error("LoadRegistry accepted duplicate names")
	}
	if !strings.Contains(strings.Join(back.Names(), ","), "migrate-database") {
		t.Error("Names lost a script")
	}
}
