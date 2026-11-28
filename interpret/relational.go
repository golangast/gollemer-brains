package interpret

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer-brains/cmdmatch"
)

// Relation-aware inference: the relation as a third
// factor. The factorized fallback composes kind +
// object words; this composition runs on the COMMAND
// side and composes relation + object: when the matcher
// abstains on a line, the line's own relational shape
// (RelationCues — the relations its words name) plus
// the object words it shares with a declaration can
// still adopt that declaration, when exactly one
// declaration fits both factors. The discipline is the
// fallback's: uniqueness or abstention.
//
// Only STATED relations compose. A declaration that
// never named its relations participates exactly as it
// always did — through the matcher alone — so a
// registry with no stated relations reads every line
// the way it did before this round.

// declObjectWords is the set of typed object words a
// declaration's own text carries — the object factor
// of the relational composition.
func declObjectWords(decl cmdmatch.Declaration) map[string]bool {
	out := map[string]bool{}
	scan := func(text string) {
		for _, w := range strings.Fields(strings.ToLower(text)) {
			w = strings.Trim(w, ".,;:!?\"'")
			if _, ok := objectWordType[w]; ok {
				out[w] = true
			}
		}
	}
	scan(decl.Description)
	for _, ex := range decl.Examples {
		scan(ex)
	}
	return out
}

// statedRelationClosure is a declaration's stated
// relation names with their inverses folded in: an edge
// declared once is read in both directions, here too.
func statedRelationClosure(decl cmdmatch.Declaration) map[string]bool {
	out := map[string]bool{}
	for _, r := range decl.Relations {
		out[r.Name] = true
		if inv := cmdmatch.InverseRelation(r.Name); inv != "" {
			out[inv] = true
		}
	}
	return out
}

// ComposeRelational composes a command read from a
// line the matcher left behind: the line's relation
// cues must ALL be covered by a declaration's stated
// relations (read in either direction), and the line
// must share at least one typed object word with the
// declaration's own text. Exactly one fitting
// declaration adopts; zero or several abstain. The
// trace names the factors that composed the read.
func (in *Interpreter) ComposeRelational(line string) (cmdmatch.Declaration, string, bool) {
	if in.reg == nil {
		return cmdmatch.Declaration{}, "", false
	}
	cues := RelationCues(line)
	if len(cues) == 0 {
		return cmdmatch.Declaration{}, "", false
	}
	words := LineObjectWords(line)
	var found cmdmatch.Declaration
	n := 0
	shared := ""
	for _, decl := range in.reg.Scripts {
		if len(decl.Relations) == 0 {
			continue
		}
		closure := statedRelationClosure(decl)
		covers := true
		for _, cue := range cues {
			if !closure[cue] {
				covers = false
				break
			}
		}
		if !covers {
			continue
		}
		obj := ""
		for w := range declObjectWords(decl) {
			if words[w] {
				obj = w
				break
			}
		}
		if obj == "" {
			continue
		}
		n++
		found = decl
		shared = obj
	}
	if n != 1 {
		return cmdmatch.Declaration{}, "", false
	}
	return found, fmt.Sprintf("relation fallback: cues %s + object %s compose %s (relations stated on the declaration)", strings.Join(cues, "+"), shared, found.Name), true
}

// RelationConflict is disjointness at the relation
// level: a line cannot both move a thing and cancel
// that same thing. When the matched declaration states
// a move relation, the line carries a cancels cue, and
// the cue's target — the words right after the cancel
// word — IS one of the values seated in the move's
// slots, the read is torn in the round-4 sense and must
// be withheld, not blended. It returns a plain
// description of the tear.
func RelationConflict(line string, decl cmdmatch.Declaration, slots map[string]string) (string, bool) {
	moves := false
	for _, r := range decl.Relations {
		if r.Name == "moves_from" || r.Name == "moves_to" {
			moves = true
		}
	}
	if !moves || len(slots) == 0 {
		return "", false
	}
	words := SpanWords(line)
	for i, w := range words {
		if w.Norm != "cancel" && w.Norm != "cancels" && w.Norm != "stop" && w.Norm != "stops" {
			continue
		}
		// The target: the next content words after the
		// cancel word, up to two, fillers skipped.
		var target []string
		for j := i + 1; j < len(words) && len(target) < 2; j++ {
			if spanFillers[words[j].Norm] {
				continue
			}
			target = append(target, words[j].Norm)
		}
		if len(target) == 0 {
			continue
		}
		for slot, value := range slots {
			if slot != "from" && slot != "to" {
				continue
			}
			v := objectKey(value)
			if v == target[0] || (len(target) == 2 && v == target[0]+" "+target[1]) {
				return fmt.Sprintf("the line both %s %s and cancels it", relationOfSlot(decl, slot), value), true
			}
		}
	}
	return "", false
}

// relationOfSlot names the move relation a slot carries
// on a declaration, for conflict wording.
func relationOfSlot(decl cmdmatch.Declaration, slot string) string {
	for _, r := range decl.Relations {
		if r.Slot == slot {
			return r.Name
		}
	}
	return "moves"
}
