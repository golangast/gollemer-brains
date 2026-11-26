package cmdmatch

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// The matcher, in one paragraph: a typed line is reduced to
// its content words (stopwords dropped, words crudely
// stemmed). Each registered script has a PROFILE — the
// content words of its description and name (its defining
// words) plus the content words of its examples. A line
// scores against a script by how much of the line's
// weighted word mass the profile covers: words are weighted
// by inverse document frequency across the registry (a word
// only one script uses is strong evidence; a word every
// script uses is weak), and words from the description or
// name count half again as much as words that appear only
// in examples. The best script wins when it clears all
// three bars declared below; otherwise the matcher says it
// does not know. That abstention is the talk guard: an
// everyday line shares few content words with any
// description, so it covers little of its own mass against
// any profile and lands below the threshold.
//
// The bars (chosen at design time; see the verification
// notes in the repo's scratch record):
//
//   - Threshold: the winning score must reach 0.40.
//   - Mass floor: the matched weighted mass must reach 1.2,
//     so one stray common word on a short line cannot win.
//   - Margin: the winner must beat the runner-up by 0.08,
//     so a line two scripts explain equally well is a
//     question, not a coin flip.
//   - Two words: a match must rest on at least two distinct
//     matched content words. One word, however distinctive,
//     is not a claim about a whole sentence — a short line
//     of talk that happens to contain a script's topic word
//     ("i keep my notes in a paper notebook") must not fire
//     it. (Added after the first measured run, which fired
//     exactly that line on the single word "notes"; the
//     run is on record in the README.)
//
// A word the typo repair supplied counts at half weight in
// the numerator: a guess is weaker evidence than a word
// actually typed. (Also from that run: "joke" repaired to
// the stemmed "jak" of "jake", and "crashed" to "trash" —
// real words sitting one edit from a registry word. Repair
// stays, because it also carries true hits like "clear"
// for "clean"; it just speaks more quietly now.)
const (
	Threshold  = 0.40
	massFloor  = 1.2
	marginBars = 0.08
	minMatched = 2
)

// stopwords are dropped from lines and profiles alike:
// grammar words, pronouns, auxiliaries, and the shape
// prepositions (in/on/from/to carry slot structure, which
// is the extractor's business, not topic). A small set of
// all-purpose verbs (make, take, put, keep, get) goes too —
// they appear in every description and request and so carry
// no topic either.
var stopwords = map[string]bool{
	"the": true, "a": true, "an": true, "my": true, "me": true,
	"i": true, "you": true, "it": true, "its": true, "this": true,
	"that": true, "these": true, "those": true, "one": true,
	"ones": true, "and": true, "or": true, "of": true, "for": true,
	"with": true, "without": true, "in": true, "on": true,
	"at": true, "by": true, "from": true, "to": true,
	"into": true, "onto": true, "over": true, "under": true,
	"again": true, "just": true, "please": true, "can": true,
	"could": true, "would": true, "should": true, "do": true,
	"does": true, "did": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true,
	"am": true, "have": true, "has": true, "had": true,
	"will": true, "shall": true, "may": true, "might": true,
	"must": true, "not": true, "no": true, "yes": true,
	"if": true, "then": true, "than": true, "too": true,
	"very": true, "so": true, "as": true, "out": true,
	"up": true, "down": true, "off": true, "about": true,
	"between": true, "through": true, "during": true,
	"before": true, "after": true, "when": true, "while": true,
	"what": true, "which": true, "who": true, "whom": true,
	"how": true, "all": true, "any": true, "some": true,
	"more": true, "most": true, "other": true, "own": true,
	"same": true, "only": true, "also": true, "get": true,
	"got": true, "make": true, "made": true, "take": true,
	"takes": true, "put": true, "puts": true, "keep": true,
	"keeps": true,
}

// stem crudely reduces a word to a topic form: copies ->
// copy, migrates -> migrat, migrate -> migrat, backed ->
// back, logs -> log. It is not linguistics; it only has to
// be CONSISTENT, because both the line and the profiles are
// stemmed with the same rules, so a word and its common
// inflections land on the same form.
func stem(w string) string {
	switch {
	case len(w) > 4 && strings.HasSuffix(w, "ies"):
		w = w[:len(w)-3] + "y"
	case len(w) > 5 && strings.HasSuffix(w, "ing"):
		w = w[:len(w)-3]
	case len(w) > 4 && strings.HasSuffix(w, "ed"):
		w = w[:len(w)-2]
	case len(w) > 4 && strings.HasSuffix(w, "es"):
		w = w[:len(w)-2]
	case len(w) > 3 && strings.HasSuffix(w, "s") && !strings.HasSuffix(w, "ss"):
		w = w[:len(w)-1]
	}
	if len(w) > 3 && strings.HasSuffix(w, "e") {
		w = w[:len(w)-1]
	}
	return w
}

// contentWords reduces text to its distinct stemmed content
// words, in first-appearance order: lowercase, split on
// anything that is not a letter or digit, drop tokens with
// no letter, drop stopwords (checked before stemming), stem
// the rest.
func contentWords(text string) []string {
	var out []string
	seen := map[string]bool{}
	for _, tok := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		hasLetter := false
		for _, r := range tok {
			if unicode.IsLetter(r) {
				hasLetter = true
				break
			}
		}
		if !hasLetter || stopwords[tok] {
			continue
		}
		w := stem(tok)
		if w == "" || seen[w] {
			continue
		}
		seen[w] = true
		out = append(out, w)
	}
	return out
}

// profile is one script's matchable word set: defining
// words (description + name parts) and example-only words.
type profile struct {
	decl     Declaration
	defining map[string]bool // description and name words
	example  map[string]bool // words appearing only in examples
}

func (p *profile) has(w string) (defining bool, ok bool) {
	if p.defining[w] {
		return true, true
	}
	if p.example[w] {
		return false, true
	}
	return false, false
}

// Matcher scores lines against a registry's profiles. Build
// one with NewMatcher; rebuild it when the registry changes
// (it is cheap — registries are small).
type Matcher struct {
	profiles []profile
	df       map[string]int // word -> number of profiles containing it
	idfMax   float64        // idf of a word no profile contains
	vocab    map[string]bool
}

// NewMatcher builds the profiles and document frequencies
// of a registry.
func NewMatcher(r *Registry) *Matcher {
	m := &Matcher{df: map[string]int{}, vocab: map[string]bool{}}
	for _, s := range r.Scripts {
		p := profile{decl: s, defining: map[string]bool{}, example: map[string]bool{}}
		for _, w := range contentWords(s.Description) {
			p.defining[w] = true
		}
		for _, w := range contentWords(strings.ReplaceAll(s.Name, "-", " ")) {
			p.defining[w] = true
		}
		for _, ex := range s.Examples {
			for _, w := range contentWords(ex) {
				if !p.defining[w] {
					p.example[w] = true
				}
			}
		}
		for w := range p.defining {
			m.df[w]++
			m.vocab[w] = true
		}
		for w := range p.example {
			m.df[w]++
			m.vocab[w] = true
		}
		m.profiles = append(m.profiles, p)
	}
	m.idfMax = math.Log(float64(len(r.Scripts)+1)) + 0.5
	return m
}

func (m *Matcher) idf(w string) float64 {
	if df, ok := m.df[w]; ok && df > 0 {
		return math.Log(float64(len(m.profiles)+1)/float64(df)) + 0.5
	}
	return m.idfMax
}

// repair returns the registry-vocabulary word a mistyped
// line word most likely meant: the UNIQUE vocabulary word
// at edit distance 1 (one substitution, insertion,
// deletion, or adjacent transposition). No unique neighbour
// — no repair; the word stays unknown.
func (m *Matcher) repair(w string) (string, bool) {
	found := ""
	for v := range m.vocab {
		if editDistance1(w, v) {
			if found != "" {
				return "", false // ambiguous: two neighbours
			}
			found = v
		}
	}
	return found, found != ""
}

// editDistance1 reports whether a and b differ by exactly
// one elementary edit.
func editDistance1(a, b string) bool {
	ra, rb := []rune(a), []rune(b)
	switch {
	case len(ra) == len(rb):
		diff := -1
		for i := range ra {
			if ra[i] != rb[i] {
				if diff >= 0 {
					// Second difference: only a transposition
					// of the two differing neighbours repairs it.
					if i == diff+1 && ra[diff] == rb[i] && ra[i] == rb[diff] {
						// verify the rest matches
						for j := i + 1; j < len(ra); j++ {
							if ra[j] != rb[j] {
								return false
							}
						}
						return true
					}
					return false
				}
				diff = i
			}
		}
		return diff >= 0
	case len(ra)+1 == len(rb):
		return oneInsertion(rb, ra)
	case len(rb)+1 == len(ra):
		return oneInsertion(ra, rb)
	}
	return false
}

// oneInsertion reports whether short is long with exactly
// one rune removed.
func oneInsertion(long, short []rune) bool {
	i, j := 0, 0
	skipped := false
	for i < len(long) && j < len(short) {
		if long[i] == short[j] {
			i++
			j++
			continue
		}
		if skipped {
			return false
		}
		skipped = true
		i++
	}
	return true
}

// Scored is one script's score against a line.
type Scored struct {
	Script  Declaration
	Score   float64 // weighted coverage of the line, 0..1
	Mass    float64 // matched weighted mass (the floor's measure)
	Matched int     // distinct matched content words (the two-words bar)
}

// Rank scores a line against every registered script,
// best first. It applies the typo repair to line words no
// profile knows before scoring.
func (m *Matcher) Rank(line string) []Scored {
	words := contentWords(line)
	// Typo repair: an unknown word with exactly one
	// vocabulary neighbour at distance 1 is scored as that
	// neighbour, at half weight — a guess is weaker
	// evidence than a word actually typed. (A word known
	// to ANY profile is left alone, even if another
	// profile would want it.)
	repaired := make([]string, len(words))
	wasRepaired := make([]bool, len(words))
	for i, w := range words {
		repaired[i] = w
		if _, known := m.df[w]; !known {
			if fix, ok := m.repair(w); ok {
				repaired[i] = fix
				wasRepaired[i] = true
			}
		}
	}
	denom := 0.0
	for _, w := range repaired {
		denom += m.idf(w)
	}
	out := make([]Scored, 0, len(m.profiles))
	for _, p := range m.profiles {
		mass := 0.0
		matched := 0
		for i, w := range repaired {
			defining, ok := p.has(w)
			if !ok {
				continue
			}
			matched++
			weight := m.idf(w)
			if defining {
				weight *= 1.5
			}
			if wasRepaired[i] {
				weight *= 0.5
			}
			mass += weight
		}
		score := 0.0
		if denom > 0 {
			score = mass / denom
		}
		out = append(out, Scored{Script: p.decl, Score: score, Mass: mass, Matched: matched})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out
}

// Match returns the script a line means, its score, and
// true — or false when the line clears none of the bars
// (below threshold, below the mass floor, fewer than two
// matched words, or too close to call between two
// scripts): the matcher does not know, and says so.
func (m *Matcher) Match(line string) (Declaration, float64, bool) {
	ranked := m.Rank(line)
	if len(ranked) == 0 {
		return Declaration{}, 0, false
	}
	best := ranked[0]
	if best.Score < Threshold || best.Mass < massFloor || best.Matched < minMatched {
		return Declaration{}, best.Score, false
	}
	if len(ranked) > 1 && best.Score-ranked[1].Score < marginBars {
		return Declaration{}, best.Score, false
	}
	return best.Script, best.Score, true
}
