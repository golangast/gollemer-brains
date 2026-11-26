package main

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Declarations: the second door into the registry. A user
// can state a command in one natural sentence —
//
//	"i have a command named copy and it will copy files
//	 from one folder to another and it is in file copy.sh"
//
// — and the REPL parses it deterministically (no ML, no
// guessing model) into the same registry entry `add` and
// the teach flow write: name copy, command ./copy.sh,
// description "copy files from one folder to another",
// slots from,to. The sentence is recognised by its
// markers — a command/script NAMED or CALLED something,
// or "my command X" — never by its topic: an ordinary
// want ("copy files from jim to jake") carries no marker
// and falls through to matching untouched, and when the
// parse is unsure it falls through too.

// parsedDeclaration is what a declaration sentence
// yielded. file is the file token as written ("" when no
// file could be found); description is the doing-clause
// ("" when none could be isolated — the flow asks).
type parsedDeclaration struct {
	name        string
	file        string
	description string
}

var (
	// A command/script named or called X — covers "i have
	// a command named X", "there's a script called X",
	// "add a command named X", "a command named X that…".
	reNamedCmd = regexp.MustCompile(`(?i)\b(?:command|script)\s+(?:named|called)\s+([a-z0-9][a-z0-9_-]*)`)
	// "my command X" / "my script X".
	reMyCmd = regexp.MustCompile(`(?i)\bmy\s+(?:command|script)\s+([a-z0-9][a-z0-9_-]*)`)
	// A .sh token anywhere in the line.
	reShFile = regexp.MustCompile(`[^\s,;]+\.sh\b`)
	// The introducer that may precede the .sh token, at
	// the very end of the text before it: "…and it is in
	// file ", "…it is in ", "…in the file ", "…the file
	// is " — each alternative runs right up to the token.
	reFileIntro = regexp.MustCompile(`(?i)(?:\band\s+)?(?:it\s+is\s+in(?:\s+file)?|in\s+(?:the\s+)?file|the\s+file\s+is)\s+$`)
	// A file named without .sh: "in file X", "the file
	// is X", "it is in X".
	reFilePhrase = regexp.MustCompile(`(?i)\b(?:in\s+the\s+file|in\s+file|the\s+file\s+is|it\s+is\s+in)\s+([^\s,;]+)`)
	reThat       = regexp.MustCompile(`(?i)\bthat\b`)
	reAndIt      = regexp.MustCompile(`(?i)\band\s+it\b`)
)

// notNames are words that can follow "my command" without
// being a name ("my command is…", "my script that…"): the
// my-pattern does not fire on them.
var notNames = map[string]bool{
	"is": true, "that": true, "does": true, "will": true,
	"to": true, "runs": true, "called": true, "named": true,
	"the": true, "a": true, "an": true, "it": true,
}

// parseDeclaration recognises a declaration sentence and
// extracts its three facts. ok is false when the line is
// not a declaration — the caller then treats it as an
// ordinary line.
func parseDeclaration(line string) (parsedDeclaration, bool) {
	var p parsedDeclaration
	nameEnd := -1
	viaMy := false
	if loc := reNamedCmd.FindStringSubmatchIndex(line); loc != nil {
		p.name = line[loc[2]:loc[3]]
		nameEnd = loc[3]
	} else if loc := reMyCmd.FindStringSubmatchIndex(line); loc != nil {
		cand := line[loc[2]:loc[3]]
		if notNames[strings.ToLower(cand)] {
			return p, false
		}
		p.name = cand
		nameEnd = loc[3]
		viaMy = true
	} else {
		return p, false
	}

	// The file: a .sh token if there is one (with its
	// introducer, which belongs to the file clause, not
	// the description), else a file phrase's next word.
	fileClause := -1 // where the file clause starts, if any
	if loc := reShFile.FindStringIndex(line); loc != nil {
		p.file = line[loc[0]:loc[1]]
		fileClause = loc[0]
		if m := reFileIntro.FindStringIndex(line[:loc[0]]); m != nil {
			fileClause = m[0]
		}
	} else if loc := reFilePhrase.FindStringSubmatchIndex(line); loc != nil {
		p.file = strings.Trim(line[loc[2]:loc[3]], `"'`)
		fileClause = loc[0]
		if fileClause >= 4 && strings.EqualFold(line[fileClause-4:fileClause], "and ") {
			fileClause -= 4
		}
	}

	// The my-pattern is the loosest trigger, so it needs
	// corroboration — a file, or a doing-clause marker —
	// before a line like "my script backs up the photos"
	// (a statement about a script, no file named) is
	// taken as a declaration.
	if viaMy && p.file == "" && !strings.Contains(strings.ToLower(line), "it will") && !strings.Contains(strings.ToLower(line), "it does") {
		return parsedDeclaration{}, false
	}

	p.description = doingClause(line, nameEnd, fileClause)
	return p, true
}

// doingClause isolates what the command DOES: the text
// after the name, cut at the file clause, entered through
// a doing marker — "it will", "it does", "that", "and
// it" — when one is present. With no marker, the whole
// post-name region is the clause (the line minus its
// name and file fragments).
func doingClause(line string, nameEnd, fileClause int) string {
	end := len(line)
	if fileClause > nameEnd {
		end = fileClause
	}
	region := ""
	if nameEnd >= 0 && nameEnd <= end {
		region = line[nameEnd:end]
	}
	if desc := clauseFrom(region); desc != "" {
		return desc
	}
	// Nothing before the file clause: the doing text may
	// follow the file ("…in file copy.sh and it copies
	// files"). Try whatever the line has left.
	if fileClause >= 0 {
		rest := line[fileClause:]
		if loc := reShFile.FindStringIndex(rest); loc != nil {
			return clauseFrom(rest[loc[1]:])
		}
	}
	return ""
}

// clauseFrom extracts the doing text from one region:
// after the strongest marker present, or the region
// itself when there is none.
func clauseFrom(region string) string {
	lower := strings.ToLower(region)
	for _, marker := range []string{"it will", "it does"} {
		if i := strings.Index(lower, marker); i >= 0 {
			return cleanClause(region[i+len(marker):])
		}
	}
	if loc := reThat.FindStringIndex(lower); loc != nil {
		return cleanClause(region[loc[1]:])
	}
	if loc := reAndIt.FindStringIndex(lower); loc != nil {
		rest := region[loc[1]:]
		// "…and it will copy files": the marker chain
		// continues — drop the will/does that follows.
		lr := strings.ToLower(rest)
		for _, v := range []string{" will ", " does "} {
			if strings.HasPrefix(lr, v) {
				rest = rest[len(v)-1:]
				break
			}
		}
		return cleanClause(rest)
	}
	return cleanClause(region)
}

// cleanClause normalises an extracted clause: whitespace
// collapsed, edge punctuation trimmed, a leading
// connector the cut left behind ("and …") dropped.
func cleanClause(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	s = strings.Trim(s, " .,;:")
	for _, p := range []string{"and ", "that "} {
		if strings.HasPrefix(strings.ToLower(s), p) {
			s = s[len(p):]
		}
	}
	return strings.Trim(s, " .,;:")
}

// runnableCommand stores a declared file as the command
// that runs it: a bare file name becomes ./name (the
// runnable form), a path with slashes stays as given.
func runnableCommand(file string) string {
	file = strings.TrimSpace(file)
	if file == "" || strings.Contains(file, "/") || strings.ContainsAny(file, " \t") {
		return file
	}
	return "./" + file
}

// declarationSlots derives a declared entry's slots. One
// rule, stated plainly: {placeholders} in the command
// name the slots; otherwise a description that speaks of
// going "from" somewhere "to" somewhere else takes the
// pair [from, to]. No other inference is made.
func declarationSlots(description, command string) []string {
	if ph := placeholdersOf(command); len(ph) > 0 {
		return ph
	}
	lower := " " + strings.ToLower(description) + " "
	if strings.Contains(lower, " from ") && strings.Contains(lower, " to ") {
		return []string{"from", "to"}
	}
	return nil
}

// declarationFlow learns a parsed declaration: any fact
// the sentence did not carry is asked for with one
// follow-up question, then the entry is written through
// the shared persist path and confirmed in plain words —
// name, description as understood, command, slots.
func (d *demo) declarationFlow(p parsedDeclaration) {
	command := ""
	if p.file != "" {
		command = runnableCommand(p.file)
	} else {
		ans, ok := d.nextLine(fmt.Sprintf("what file runs %s — a .sh file, or the whole command: ", p.name))
		if !ok || strings.TrimSpace(ans) == "" {
			fmt.Fprintln(d.out, "ok — not saved.")
			return
		}
		ans = strings.TrimSpace(ans)
		if strings.HasSuffix(ans, ".sh") && !strings.ContainsAny(ans, " \t") {
			command = runnableCommand(ans)
		} else {
			command = ans
		}
	}
	desc := p.description
	if desc == "" {
		ans, ok := d.nextLine(fmt.Sprintf("what does %s do: ", p.name))
		if !ok || strings.TrimSpace(ans) == "" {
			fmt.Fprintln(d.out, "ok — not saved.")
			return
		}
		desc = strings.TrimSpace(ans)
	}
	decl := cmdmatch.Declaration{
		Name:        p.name,
		Command:     command,
		Description: desc,
		Slots:       declarationSlots(desc, command),
		Examples:    []string{desc},
	}
	if !d.persist(decl) {
		return
	}
	slots := "no slots"
	if len(decl.Slots) > 0 {
		slots = "slots: " + strings.Join(decl.Slots, ", ")
	}
	fmt.Fprintf(d.out, "learned: %s — %s\n", decl.Name, decl.Description)
	fmt.Fprintf(d.out, "command: %s (%s). it chooses %s from now on, in this session and every later one. nothing was retrained.\n", decl.Command, slots, decl.Name)
}
