package main

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Teach-on-ask: the conversation itself is the teaching.
// A want that matches nothing used to be a dead end — the
// user had to know about `add` and fill in its form. Now
// the REPL offers to learn the want on the spot, and the
// entry it derives needs nothing the user has not already
// said: their want-line, verbatim, becomes the new entry's
// description (their words ARE the description — the
// design's core move), and the one answer they give is
// the command that should run. The entry is written
// through the same persist path as `add`, so it survives
// restarts; and because the matcher reads descriptions
// (with synonym expansion), the want chooses the taught
// command from the next line on — in this session and
// every later one.

// looksLikeWant decides whether an unmatched line earns
// a teach offer. Two signals, either enough: some script
// is at least a neighbour of the line (a closest script
// exists with real matched mass), or the line contains
// an action word from the matcher's verb families —
// someone is asking for something to be done. Pure small
// talk has neither: "how are you" shares no words with
// any description and names no action, so it gets the
// plain don't-know and no offer.
func (d *demo) looksLikeWant(line string, ranked []cmdmatch.Scored) bool {
	if len(ranked) > 0 && ranked[0].Mass > 0 {
		return true
	}
	return cmdmatch.HasActionWord(line)
}

// teachFlow runs the offer made after an unmatched want.
// On success it confirms what was learned and immediately
// re-runs the original want through the pipeline — the
// loop closes in the same breath, with the would-run line
// the want now produces.
func (d *demo) teachFlow(want string) {
	_, learned, viaDeclaration := d.learnWant(want)
	if !learned || viaDeclaration {
		return
	}
	if decl2, score, ok := d.matcher.Match(want); ok {
		d.present(decl2, score, want)
	} else {
		fmt.Fprintln(d.out, "(saved — but that line on its own still does not match itself: a match needs two of its words in a line, so longer phrasings of it are the ones that will land.)")
	}
}

// learnWant is the teaching core, shared by the
// single-line teach offer and the plan layer's gap offer:
// it prompts for the command that should run when the
// user says the want, and writes the entry through the
// shared persist path. The user's reply is that command;
// an empty reply (or EOF) skips, and nothing is written.
// It reports what happened: the learned declaration and
// true for the bare-command path, or viaDeclaration true
// when the reply read as a declaration and was routed
// through the declaration flow instead (which prints its
// own confirmation).
func (d *demo) learnWant(want string) (decl cmdmatch.Declaration, learned, viaDeclaration bool) {
	if d.nextLine == nil {
		return decl, false, false
	}
	ans, ok := d.nextLine("want to teach me? what command should run when you say that — or press enter to skip: ")
	if !ok {
		return decl, false, false
	}
	command := strings.TrimSpace(ans)
	if command == "" {
		return decl, false, false // skipped: nothing written, nothing learned
	}
	// The guard: an answer that reads like a DECLARATION
	// — a sentence carrying a file marker — is not a bare
	// command, and storing it as one was the defect that
	// swallowed John's next want as this answer and baked
	// its filename into the stored command. Route it
	// through the declaration parse instead; the entry it
	// learns is the declared one. A bare command answer
	// ("mkdir {folder}", "./wipe.sh") never trips this.
	if p, ok := parseTeachAnswer(command); ok {
		d.declarationFlow(p)
		return decl, true, true
	}
	decl = cmdmatch.Declaration{
		Name:        d.uniqueName(deriveName(command)),
		Command:     command,
		Description: want,
		Slots:       declarationSlots(want, command),
		Examples:    []string{want},
	}
	if !d.persist(decl) {
		return cmdmatch.Declaration{}, false, false
	}
	fmt.Fprintf(d.out, "%s when you say %q, i'll run %s (saved as %s) — it chooses it from now on, in this session and every later one. nothing was retrained.\n", d.paint(ansiGood, "learned:"), want, command, decl.Name)
	return decl, true, false
}

// placeholdersOf returns the {slot} names a command
// declares inline, in order, duplicates removed. A
// taught command's slots are exactly its placeholders:
// "mkdir {folder}" takes one slot, folder.
func placeholdersOf(command string) []string {
	var out []string
	seen := map[string]bool{}
	for _, m := range rePlaceholder.FindAllStringSubmatch(command, -1) {
		name := strings.TrimSpace(m[1])
		if name != "" && !seen[name] {
			seen[name] = true
			out = append(out, name)
		}
	}
	return out
}

var rePlaceholder = regexp.MustCompile(`\{([^{}]+)\}`)

// deriveName names a taught entry after its command: the
// program it runs, without path or extension —
// "mkdir {folder}" is "mkdir", "./scripts/make-folder.sh"
// is "make-folder". The name's words join the entry's
// matching profile, like any registered name.
func deriveName(command string) string {
	fields := strings.Fields(command)
	if len(fields) == 0 {
		return "my-command"
	}
	base := fields[0]
	if i := strings.LastIndex(base, "/"); i >= 0 {
		base = base[i+1:]
	}
	if i := strings.LastIndex(base, "."); i > 0 {
		base = base[:i]
	}
	slug := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			return r
		case r >= 'A' && r <= 'Z':
			return r + ('a' - 'A')
		default:
			return '-'
		}
	}, base)
	slug = strings.Trim(slug, "-")
	for strings.Contains(slug, "--") {
		slug = strings.ReplaceAll(slug, "--", "-")
	}
	if slug == "" {
		return "my-command"
	}
	return slug
}

// uniqueName returns name, or name-2, name-3, … — the
// first variant not already in the registry. A taught
// entry must never fail to save because its derived name
// collided with a script the user already had.
func (d *demo) uniqueName(name string) string {
	if _, ok := d.reg.Find(name); !ok {
		return name
	}
	for i := 2; ; i++ {
		cand := fmt.Sprintf("%s-%d", name, i)
		if _, ok := d.reg.Find(cand); !ok {
			return cand
		}
	}
}
