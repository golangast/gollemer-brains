package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Corrections that stick: the closed loop, with no
// retraining anywhere in it. When the REPL presents a
// match and the user's NEXT line says it was the wrong
// one, the correction is not a shrug and a retype — the
// user names the script they did mean, and three things
// happen (correctionFlow): the original want line joins
// the chosen script's examples in the registry (the same
// persist-and-rebuild discipline as teaching, so the
// line matches the right script from then on, in every
// later session), the correction is appended as one JSON
// line to corrections.jsonl beside the registry — the
// hard-negative record, data for future training rounds
// that nothing at runtime reads — and the chosen script
// is presented for the original line on the spot.
//
// The trigger set is small and exact, by design: a line
// that is exactly "no", or starts with "no,", "no i
// meant", "wrong", or "not that one" (all case-folded).
// Anything broader would eat ordinary wants; anything
// narrower would miss the words a correction is actually
// typed in. A correction with no presentation before it
// is not special at all — handle only consults this file
// when a match was just presented, so "no" said to an
// empty room falls through to the ordinary paths.

// isCorrection reports whether a line is one of the
// correction triggers above.
func isCorrection(line string) bool {
	l := strings.ToLower(strings.TrimSpace(line))
	switch {
	case l == "no":
		return true
	case strings.HasPrefix(l, "no,"), strings.HasPrefix(l, "no i meant"):
		return true
	case strings.HasPrefix(l, "wrong"), strings.HasPrefix(l, "not that one"):
		return true
	}
	return false
}

// correctionFlow runs a correction against the match
// just presented: it asks which script was meant, lists
// the registry numbered, and applies the answer. An
// answer that names nothing (enter, EOF, an unknown
// word, or the very script that was rejected) changes
// nothing and writes nothing.
func (d *demo) correctionFlow(prev *presentedMatch) {
	if d.nextLine == nil {
		return
	}
	names := d.reg.Names()
	fmt.Fprintln(d.out, "which script did you mean?")
	for i, n := range names {
		fmt.Fprintf(d.out, "%d. %s\n", i+1, n)
	}
	ans, ok := d.nextLine("")
	if !ok {
		return
	}
	chosen := resolveChoice(ans, names)
	if chosen == "" || chosen == prev.decl.Name {
		return
	}
	decl, found := d.reg.Find(chosen)
	if !found {
		return
	}
	if !d.addExample(chosen, prev.line) {
		return
	}
	d.appendCorrection(prev.line, chosen, prev.decl.Name)
	// Re-present the chosen script for the original
	// line. The example just added should make the line
	// match it outright; if the matcher still demurs,
	// present the chosen script with its ranked score
	// rather than nothing — the user said which one
	// they meant, and the answer should show it.
	if m, score, ok := d.matcher.Match(prev.line); ok && m.Name == chosen {
		d.presentMatch(m, score, prev.line)
		return
	}
	score := 0.0
	for _, s := range d.matcher.Rank(prev.line) {
		if s.Script.Name == chosen {
			score = s.Score
			break
		}
	}
	d.presentMatch(decl, score, prev.line)
}

// resolveChoice reads a correction answer: a 1-based
// number into names, or a script name (case-folded).
// Anything else resolves to "".
func resolveChoice(ans string, names []string) string {
	ans = strings.TrimSpace(ans)
	if n, err := strconv.Atoi(ans); err == nil {
		if n >= 1 && n <= len(names) {
			return names[n-1]
		}
		return ""
	}
	for _, name := range names {
		if strings.EqualFold(ans, name) {
			return name
		}
	}
	return ""
}

// addExample appends an example sentence to an EXISTING
// registry entry and saves the registry — the correction
// loop's write path, sibling to persist (which appends
// whole entries). The matcher is rebuilt over the saved
// registry, so the example matches from the next line
// on. An example the entry already carries is a no-op
// success: there is nothing to write.
//
// The landed line also sprouts its paraphrase variants
// (synthesize.go), generated from that line alone in the
// entry's context: a correction teaches one sentence,
// and the variants widen it the same way a fresh teaching
// is widened — the loop's lesson should not be narrower
// than the door's.
func (d *demo) addExample(name, example string) bool {
	for i := range d.reg.Scripts {
		if d.reg.Scripts[i].Name != name {
			continue
		}
		for _, ex := range d.reg.Scripts[i].Examples {
			if ex == example {
				return true
			}
		}
		origLen := len(d.reg.Scripts[i].Examples)
		d.reg.Scripts[i].Examples = append(d.reg.Scripts[i].Examples, example)
		existing := map[string]bool{d.reg.Scripts[i].Description: true}
		for _, ex := range d.reg.Scripts[i].Examples {
			existing[ex] = true
		}
		variants := synthesizeFrom(d.reg.Scripts[i], []string{example}, existing)
		d.reg.Scripts[i].Examples = append(d.reg.Scripts[i].Examples, variants...)
		if err := d.reg.Save(d.registryPath); err != nil {
			fmt.Fprintf(d.out, "not saved: could not write the registry: %v\n", err)
			d.reg.Scripts[i].Examples = d.reg.Scripts[i].Examples[:origLen]
			return false
		}
		d.matcher = cmdmatch.NewMatcher(d.reg)
		d.regStamp = stampOf(d.registryPath)
		return true
	}
	return false
}

// correctionRecord is one line of corrections.jsonl:
// the line the user typed, the script they chose, and
// the script that was wrongly presented — the rejected
// field is the hard negative, kept as data. Nothing at
// runtime reads this file; it is the raw material a
// future training round would mine.
type correctionRecord struct {
	Input    string `json:"input"`
	Chose    string `json:"chose"`
	Rejected string `json:"rejected"`
}

// appendCorrection writes one correction record to
// corrections.jsonl beside the registry file, creating
// the file when absent. A failure here is silent by
// design: the registry example — the part that changes
// what the program does — has already landed, and this
// file is exhaust, not machinery.
func (d *demo) appendCorrection(input, chose, rejected string) {
	raw, err := json.Marshal(correctionRecord{Input: input, Chose: chose, Rejected: rejected})
	if err != nil {
		return
	}
	path := filepath.Join(filepath.Dir(d.registryPath), "corrections.jsonl")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.Write(append(raw, '\n'))
}
