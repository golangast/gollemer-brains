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
// markers, never by its topic: a command/script NAMED or
// CALLED something, "my command X", or — with no name at
// all — "i have a command that …" carrying a file marker
// or a doing-clause, in which case the name is derived
// from the file (del.sh → del) and shown in the
// confirmation. An ordinary want ("copy files from jim
// to jake") carries no marker and falls through to
// matching untouched, and when the parse is unsure it
// falls through too.

// parsedDeclaration is what a declaration sentence
// yielded. file is the file token as written ("" when no
// file could be found); description is the doing-clause
// ("" when none could be isolated — the flow asks).
// routine is true when the entry kind word was "routine"
// instead of "command"/"script": the entry is a bundle
// of steps, not a command (routine.go).
type parsedDeclaration struct {
	name        string
	file        string
	description string
	routine     bool
}

var (
	// A command/script/routine named or called X — covers
	// "i have a command named X", "there's a script
	// called X", "add a command named X", "a command
	// named X that…", "i have a routine named X".
	reNamedCmd = regexp.MustCompile(`(?i)\b(command|script|routine)\s+(?:named|called)\s+([a-z0-9][a-z0-9_-]*)`)
	// "my command X" / "my script X" / "my routine X".
	reMyCmd = regexp.MustCompile(`(?i)\bmy\s+(command|script|routine)\s+([a-z0-9][a-z0-9_-]*)`)
	// An UNNAMED declaration: "i have a command that …",
	// "there is a script that …" — the command/script is
	// stated but never named; the name is derived from
	// its file instead. (A routine cannot be declared
	// unnamed: it has no file to take a name from, so the
	// unnamed trigger does not fire for one.)
	reHaveCmd = regexp.MustCompile(`(?i)\b(?:i have|there is|there's|i've got|we have)\s+an?\s+(command|script|routine)\b`)
	// A .sh token anywhere in the line.
	reShFile = regexp.MustCompile(`[^\s,;]+\.sh\b`)
	// The file-clause INTRODUCER: the words that open the
	// clause naming the file — "and the command is in
	// file", "and it is in file", "the command is in",
	// "it is in file", "and the command is", "in file",
	// "in the file", "the file is". The description is
	// cut where the introducer STARTS, not where the
	// file token sits: cutting at the token left the
	// tail "…and the command is" inside the description
	// of John's delete declaration. Longest shapes first,
	// so "the command is in file" wins over "in file".
	reFileClauseIntro = regexp.MustCompile(`(?i)\b(?:and\s+)?(?:the\s+command\s+is\s+in\s+(?:the\s+)?file|the\s+command\s+is\s+in|the\s+command\s+is|it\s+is\s+in\s+(?:the\s+)?file|it\s+is\s+in|in\s+(?:the\s+)?file|the\s+file\s+is)\b`)
	// A file named without .sh: "in file X", "the file
	// is X", "it is in X".
	reFilePhrase = regexp.MustCompile(`(?i)\b(?:in\s+the\s+file|in\s+file|the\s+file\s+is|it\s+is\s+in)\s+([^\s,;]+)`)
	reThat       = regexp.MustCompile(`(?i)\bthat\b`)
	reAndIt      = regexp.MustCompile(`(?i)\band\s+it\b`)
	// A description that speaks of "a file" or "the
	// file": the file-slot rule's trigger.
	reFileMention = regexp.MustCompile(`(?i)\b(?:a|the)\s+file\b`)
)

// fileClauseStart returns where a line's file clause
// begins: the start of the LAST file-clause introducer
// ending at or before the file token — the introducer
// nearest the token is the clause's opening; an earlier
// "in file" inside the doing-clause itself is not. With
// no introducer, the clause starts at the token.
func fileClauseStart(line string, tokenStart int) int {
	start := tokenStart
	if tokenStart > len(line) {
		tokenStart = len(line)
	}
	for _, m := range reFileClauseIntro.FindAllStringIndex(line[:tokenStart], -1) {
		start = m[0]
	}
	return start
}

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
	unnamed := false
	if loc := reNamedCmd.FindStringSubmatchIndex(line); loc != nil {
		p.routine = strings.EqualFold(line[loc[2]:loc[3]], "routine")
		p.name = line[loc[4]:loc[5]]
		nameEnd = loc[5]
	} else if loc := reMyCmd.FindStringSubmatchIndex(line); loc != nil {
		cand := line[loc[4]:loc[5]]
		if notNames[strings.ToLower(cand)] {
			return p, false
		}
		p.routine = strings.EqualFold(line[loc[2]:loc[3]], "routine")
		p.name = cand
		nameEnd = loc[5]
		viaMy = true
	} else if loc := reHaveCmd.FindStringSubmatchIndex(line); loc != nil {
		// Unnamed: "i have a command that …". The name
		// is derived from the file below; the doing-clause
		// starts right after the command/script word. A
		// routine has no file to derive a name from, so
		// an unnamed routine is not a declaration shape
		// this door takes.
		if strings.EqualFold(line[loc[2]:loc[3]], "routine") {
			return p, false
		}
		nameEnd = loc[1]
		unnamed = true
	} else {
		return p, false
	}

	// The file: a .sh token if there is one, else a file
	// phrase's next word. Either way the file CLAUSE
	// starts at its introducer ("…and the command is in
	// file del.sh" — the clause opens at "and the
	// command is", not at "del.sh"), and the description
	// is cut there.
	fileClause := -1 // where the file clause starts, if any
	if loc := reShFile.FindStringIndex(line); loc != nil {
		p.file = line[loc[0]:loc[1]]
		fileClause = fileClauseStart(line, loc[0])
	} else if loc := reFilePhrase.FindStringSubmatchIndex(line); loc != nil {
		p.file = strings.Trim(line[loc[2]:loc[3]], `"'`)
		fileClause = fileClauseStart(line, loc[2])
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
	if isJunkClause(p.description) {
		p.description = ""
	}

	// The unnamed trigger likewise needs substance: a
	// file marker or a doing-clause. A bare "i have a
	// command" states nothing learnable and falls
	// through. With a file, the entry takes the file's
	// stem as its name — del.sh is called del — and the
	// confirmation shows it, so the derivation is never
	// silent.
	if unnamed {
		if p.file == "" && p.description == "" {
			return parsedDeclaration{}, false
		}
		if p.file != "" {
			p.name = deriveName(p.file)
		}
	}
	return p, true
}

// isJunkClause reports whether an extracted clause is
// only connective tissue ("is", "it", "that and") — the
// cut left nothing the command DOES, so the fact counts
// as missing and the flow asks for it.
func isJunkClause(s string) bool {
	if s == "" {
		return true
	}
	for _, w := range strings.Fields(strings.ToLower(s)) {
		switch w {
		case "is", "it", "the", "a", "an", "that", "and", "will", "does":
		default:
			return false
		}
	}
	return true
}

// parseTeachAnswer recognises a teach answer that is not
// a bare command but a declaration in disguise: asked
// "what command should run when you say that?", the user
// answered with a sentence — "i have a command that
// deletes … in file del.sh", or "the command is in file
// del.sh". Storing such a sentence AS the command was
// the defect John's lines exposed (the next want was
// swallowed as the answer, and its filename baked into
// the stored "command" forever). So an answer carrying
// a file marker anywhere but its first token is routed
// through the declaration parse instead. A bare command
// — "./wipe.sh", "wipe.sh --force", "mkdir {folder}" —
// leads with its program (or names no file at all) and
// keeps the teach flow's original behaviour exactly.
func parseTeachAnswer(ans string) (parsedDeclaration, bool) {
	if p, ok := parseDeclaration(ans); ok {
		return p, true
	}
	if !reShFile.MatchString(ans) && !reFilePhrase.MatchString(ans) {
		return parsedDeclaration{}, false
	}
	fields := strings.Fields(ans)
	if len(fields) < 4 || strings.Contains(fields[0], ".sh") {
		return parsedDeclaration{}, false // a bare command, not a sentence
	}
	// Build the declaration by hand: the file from its
	// marker, the name from the file's stem, the
	// description from the text before the file clause
	// (or after the file token, when the doing words
	// trail it).
	var p parsedDeclaration
	fileClause := -1
	if loc := reShFile.FindStringIndex(ans); loc != nil {
		p.file = ans[loc[0]:loc[1]]
		fileClause = fileClauseStart(ans, loc[0])
	} else if loc := reFilePhrase.FindStringSubmatchIndex(ans); loc != nil {
		p.file = strings.Trim(ans[loc[2]:loc[3]], `"'`)
		fileClause = fileClauseStart(ans, loc[2])
	}
	if p.file == "" {
		return parsedDeclaration{}, false
	}
	p.name = deriveName(p.file)
	if fileClause > 0 {
		region := ans[:fileClause]
		if loc := reHaveCmd.FindStringIndex(region); loc != nil {
			region = region[loc[1]:]
		}
		p.description = clauseFrom(region)
	}
	if p.description == "" && fileClause >= 0 {
		rest := ans[fileClause:]
		if loc := reShFile.FindStringIndex(rest); loc != nil {
			p.description = clauseFrom(rest[loc[1]:])
		}
	}
	if isJunkClause(p.description) {
		p.description = ""
	}
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

// declarationSlots derives an entry's slots — declared
// or taught; the description is the want-line in the
// teach flow, so the same rules apply there. Stated
// plainly: {placeholders} in the command name the slots;
// otherwise a description that speaks of going "from"
// somewhere "to" somewhere else takes the pair
// [from, to]; otherwise a description that mentions "a
// file" or "the file" takes the single slot [file] —
// the common script-takes-a-file shape, and the one
// John's delete declaration needs: without it the
// filename in "delete the first line of the file
// cat.txt" had nowhere to bind, and the would-run line
// was the bare ./del.sh. No other inference is made.
func declarationSlots(description, command string) []string {
	if ph := placeholdersOf(command); len(ph) > 0 {
		return ph
	}
	lower := " " + strings.ToLower(description) + " "
	if strings.Contains(lower, " from ") && strings.Contains(lower, " to ") {
		return []string{"from", "to"}
	}
	if reFileMention.MatchString(description) {
		return []string{"file"}
	}
	return nil
}

// declarationFlow learns a parsed declaration: any fact
// the sentence did not carry is asked for with one
// follow-up question, then the entry is written through
// the shared persist path and confirmed in plain words —
// name, description as understood, command, slots.
func (d *demo) declarationFlow(p parsedDeclaration) {
	if p.routine {
		d.routineDeclarationFlow(p)
		return
	}
	command := ""
	if p.file != "" {
		command = runnableCommand(p.file)
	} else {
		label := p.name
		if label == "" {
			label = "it" // an unnamed declaration whose file is still missing
		}
		ans, ok := d.nextLine(fmt.Sprintf("what file runs %s — a .sh file, or the whole command: ", label))
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
	// An unnamed declaration with a file already carries
	// its derived name (the file's stem); one that had to
	// ask for its file derives the name from the answer.
	name := p.name
	if name == "" {
		name = deriveName(command)
	}
	desc := p.description
	if desc == "" {
		ans, ok := d.nextLine(fmt.Sprintf("what does %s do: ", name))
		if !ok || strings.TrimSpace(ans) == "" {
			fmt.Fprintln(d.out, "ok — not saved.")
			return
		}
		desc = strings.TrimSpace(ans)
	}
	decl := cmdmatch.Declaration{
		Name:        name,
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
	fmt.Fprintf(d.out, "%s %s — %s\n", d.paint(ansiGood, "learned:"), decl.Name, decl.Description)
	fmt.Fprintf(d.out, "command: %s (%s). it chooses %s from now on, in this session and every later one. nothing was retrained.\n", decl.Command, slots, decl.Name)
}

// routineDeclarationFlow learns a parsed routine
// declaration — "i have a routine named nightly and it
// will back up my photos and migrate the database and
// compress the logs". The doing-clause is split into
// segments the way a plan line is, and EVERY segment
// must already resolve to a known command: a routine is
// a bundle of concepts the registry knows, never a way
// to smuggle in a new one. A segment that resolves to
// nothing refuses the whole declaration — one plain
// line naming the segment, and nothing is saved. A
// segment that resolves to another routine is accepted:
// expansion is depth-capped at render time, not here.
func (d *demo) routineDeclarationFlow(p parsedDeclaration) {
	desc := p.description
	if desc == "" {
		ans, ok := d.nextLine(fmt.Sprintf("what does %s do: ", p.name))
		if !ok || strings.TrimSpace(ans) == "" {
			fmt.Fprintln(d.out, "ok — not saved.")
			return
		}
		desc = strings.TrimSpace(ans)
	}
	segments := splitSegments(desc)
	if len(segments) < 2 {
		fmt.Fprintln(d.out, "a routine is a bundle of steps — that reads as one step, not a bundle. not saved.")
		return
	}
	names := make([]string, 0, len(segments))
	for _, seg := range segments {
		decl, _, ok := d.matchSegment(seg)
		if !ok {
			fmt.Fprintf(d.out, "i don't know '%s' yet — teach me that command first\n", seg)
			return
		}
		names = append(names, decl.Name)
	}
	decl := cmdmatch.Declaration{
		Name:        p.name,
		Description: desc,
		Examples:    []string{desc},
		Steps:       segments,
	}
	if !d.persist(decl) {
		return
	}
	fmt.Fprintf(d.out, "%s %s — %s\n", d.paint(ansiGood, "learned:"), decl.Name, decl.Description)
	fmt.Fprintf(d.out, "routine: %d steps (%s). it chooses %s from now on, in this session and every later one. nothing was retrained.\n", len(segments), strings.Join(names, ", "), decl.Name)
}
