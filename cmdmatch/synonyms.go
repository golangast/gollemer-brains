package cmdmatch

import (
	"strings"
	"unicode"
)

// The synonym table: word FAMILIES — groups of words that
// name the same command-domain idea — used to expand each
// script's profile at registry load (see match.go). A
// description written with one word of a family then also
// matches requests phrased with its cousins: a user who
// describes a script as "renames the photos" is understood
// when they later say "change the file names", because
// rename and change are in one family.
//
// The families are deliberately tight and command-shaped:
// the verbs commands do and the nouns they act on. They
// are NOT a thesaurus. Two families that could exist were
// kept apart on purpose: file and photo are separate
// families (a file is not a photo; merging them made the
// cleaner's word "files" evidence about photo scripts),
// and "movie" is in no family at all (movies are not
// photos either — everyday talk about a downloaded movie
// already fires the downloads script on its one exact
// word, and cousins would only widen that).
//
// The trainer's synonym table (brain/synonyms.go) is the
// precedent for the format — flat groups of near-synonyms
// applied in every direction — but its words are everyday
// small talk (like/love, happy/glad). Two of its pairs
// carry over (start/begin, make/build; "make" itself is a
// stopword here, so only build lands). The rest of this
// table is the command vocabulary that table does not
// cover.
//
// Adding words is the user-facing feature this file
// exists for: to teach the matcher a new cousin, add the
// word to the family it belongs with (or start a family).
// Families must stay disjoint — a word in two families
// would blur which idea a match rests on — and a test
// pins that, over the stemmed forms the matcher actually
// compares.
//
// A cousin match counts LESS than the word itself:
// synonymFactor in match.go. And curation is part of the
// deal — a family member that produces a wrong match in
// measurement comes out, and the removal is reported.

// verbFamilies are the actions commands do.
var verbFamilies = [][]string{
	{"rename", "change", "retitle", "relabel"},
	{"list", "show", "display", "view"},
	{"back", "backup", "copy", "save", "store", "duplicate"},
	{"clean", "clear", "delete", "remove", "erase", "wipe", "tidy", "empty", "purge"},
	{"trash", "discard", "throw"},
	{"move", "transfer", "migrate", "shift", "relocate"},
	{"start", "begin", "run", "launch", "boot", "fire"},
	{"stop", "kill", "halt", "end", "quit"},
	{"search", "find", "look", "locate", "seek"},
	{"compress", "zip", "pack", "squash", "shrink", "archive"},
	{"sync", "update", "refresh", "renew", "freshen", "reconcile"},
	{"download", "fetch", "grab"},
	{"upload", "push", "publish", "deploy", "release", "send"},
	{"create", "build", "generate"},
	{"check", "test", "verify", "validate"},
	{"fill", "load", "populate", "seed"},
	{"fake", "dummy", "mock", "synthetic"},
}

// nounFamilies are the things commands act on.
var nounFamilies = [][]string{
	{"file", "document"},
	{"photo", "picture", "image", "pic"},
	{"folder", "directory", "dir"},
	{"name", "title", "label"},
	{"website", "site"},
	{"data", "record", "dataset"},
}

// synonymFamilies is every family, verbs first — the
// table the profile expansion and the disjointness test
// work over.
var synonymFamilies = func() [][]string {
	out := make([][]string, 0, len(verbFamilies)+len(nounFamilies))
	out = append(out, verbFamilies...)
	out = append(out, nounFamilies...)
	return out
}()

// actionStems is the stemmed form of every verb in the
// table, plus "make": the trainer table's make/build pair
// carries over to matching only as build ("make" is a
// stopword in match.go, so it can never be a profile
// word), but as a signal that a line is a WANT — someone
// asking for something to be done — "make" is as much an
// action word as "create". HasActionWord is that signal;
// it plays no part in scoring.
var actionStems = buildActionStems()

func buildActionStems() map[string]bool {
	out := map[string]bool{}
	for _, fam := range verbFamilies {
		for _, w := range fam {
			out[stem(w)] = true
		}
	}
	out[stem("make")] = true
	return out
}

// HasActionWord reports whether a line contains a word
// from one of the table's verb families, in any
// inflection: the signal that a line is asking for an
// action, as opposed to small talk. It tokenises the raw
// line — stopwords included, since "make" is one — and
// stems each token with the same stem() matching uses,
// so "creates", "creating" and "create" all land on the
// family's form.
func HasActionWord(line string) bool {
	for _, tok := range strings.FieldsFunc(strings.ToLower(line), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if actionStems[stem(tok)] {
			return true
		}
	}
	return false
}

// cousins maps each stemmed family member to the stemmed
// forms of the rest of its family. Built once from
// synonymFamilies, stemmed with the same stem() the
// profiles and lines go through, so the table's base
// forms land on the forms matching actually compares.
var cousins = buildCousins()

func buildCousins() map[string][]string {
	out := map[string][]string{}
	for _, fam := range synonymFamilies {
		stemmed := make([]string, 0, len(fam))
		for _, w := range fam {
			stemmed = append(stemmed, stem(w))
		}
		for _, w := range stemmed {
			for _, c := range stemmed {
				if c == w {
					continue
				}
				dup := false
				for _, have := range out[w] {
					if have == c {
						dup = true
						break
					}
				}
				if !dup {
					out[w] = append(out[w], c)
				}
			}
		}
	}
	return out
}

// cousinsOf returns the stemmed cousins of a stemmed
// profile word (nil when the word is in no family).
func cousinsOf(stemmed string) []string {
	return cousins[stemmed]
}
