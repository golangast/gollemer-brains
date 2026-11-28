// Package cmdmatch is the describe-it-once mechanism behind
// examples/customcmd: a user declares their own commands —
// a name, the command line, and a plain description of what
// it does — in a registry file, and this package matches
// what the user later types against those descriptions at
// runtime. Nothing is trained per command: adding a script
// to the registry makes it matchable on the next line typed.
//
// The division of labour is deliberate. MEANING — which
// script a sentence refers to — is decided here, lexically,
// against descriptions the user wrote. ARGUMENTS — the
// folder names and other values inside the sentence — are
// a different problem, handled by a small trained extractor
// brain that knows sentence shapes ("from X to Y") but has
// never seen the user's scripts. This package is the half
// that needs no training at all.
package cmdmatch

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// Declaration is one user script, as written in the
// registry file: the whole user-facing surface. Name and
// Command and Description are required; Slots names the
// arguments the command takes, in the order they are
// appended to the command line; Examples are a few
// sentences in the user's own words that mean this script
// (they widen what the matcher recognises — the description
// alone is enough).
//
// The remaining fields are the PLAN layer's metadata —
// all optional, all ignored by the matcher. Stage is a
// rough order tag (inspect, prepare, execute, verify,
// cleanup) saying where in a multi-step job this command
// belongs; After names other commands that should run
// before this one when both appear in a plan; ActsOn is
// a hint about what kind of thing the command acts on
// (file, tests, repo, logs…), so steps in a plan can
// share context. They exist so a plan can be ORDERED and
// its gaps spotted; they change nothing about which
// script a single sentence means.
//
// Needs names concepts that should run EARLIER than
// this one in a job — a relation, not an ordering the
// plan enforces: when a plan renders, a Need that is
// absent from the plan, or sitting later in it, earns a
// printed note (unless an After edge already encodes
// the same relation). The matcher ignores it like the
// rest of the plan metadata.
//
// Steps, when present, makes the entry a ROUTINE
// instead of a script: a named bundle of segment texts,
// each the kind of line that would match one of the
// registry's commands. A routine has no Command of its
// own — a want that matches it expands into a plan of
// its steps. The matcher still reads the entry by its
// description and examples, exactly like a script.
type Declaration struct {
	Name        string   `json:"name"`
	Command     string   `json:"command"`
	Description string   `json:"description"`
	Slots       []string `json:"slots,omitempty"`
	Examples    []string `json:"examples,omitempty"`
	Stage       string   `json:"stage,omitempty"`
	After       []string `json:"after,omitempty"`
	ActsOn      string   `json:"acts_on,omitempty"`
	Needs       []string `json:"needs,omitempty"`
	Steps       []string `json:"steps,omitempty"`

	// Relations names the typed relationships this
	// command stands in (see relations.go): each names
	// a relation type, the slot it runs through, and —
	// for edges whose far side is a name, like requires
	// — its target. Optional; a declaration that states
	// none still carries the relations its shape implies
	// (DeriveRelations names them), but only STATED
	// relations are validated and composed over.
	Relations []Relation `json:"relations,omitempty"`
	// SlotTypes declares the value type of individual
	// slots (person, folder, file, time, place…): the
	// object layer's contract. A seated value carries
	// its slot's type, and relation ranges are checked
	// against it. Optional per slot.
	SlotTypes map[string]string `json:"slot_types,omitempty"`
}

// Registry is a loaded registry file: an optional comment
// (the shipped file explains itself there) and the scripts,
// in file order.
type Registry struct {
	Comment string        `json:"comment,omitempty"`
	Scripts []Declaration `json:"scripts"`
}

// LoadRegistry reads and validates a registry file. A
// malformed registry is an error that names the problem —
// a registry the program silently half-reads would match
// against scripts the user thinks exist and do not.
func LoadRegistry(path string) (*Registry, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r Registry
	if err := json.Unmarshal(raw, &r); err != nil {
		return nil, fmt.Errorf("cmdmatch: bad registry %s: %w", path, err)
	}
	if err := r.Validate(); err != nil {
		return nil, fmt.Errorf("cmdmatch: registry %s: %w", path, err)
	}
	return &r, nil
}

// Validate checks the registry's own consistency: every
// script has the required fields, names are unique, and
// slot names within a script are unique.
func (r *Registry) Validate() error {
	seen := map[string]bool{}
	for i, s := range r.Scripts {
		if strings.TrimSpace(s.Name) == "" {
			return fmt.Errorf("script %d has no name", i)
		}
		if seen[s.Name] {
			return fmt.Errorf("script name %q appears twice", s.Name)
		}
		seen[s.Name] = true
		if len(s.Steps) > 0 {
			// A routine: segment texts instead of a
			// command. Both at once is a contradiction —
			// the entry either runs one command or
			// expands into a plan of steps.
			if strings.TrimSpace(s.Command) != "" {
				return fmt.Errorf("routine %q declares both steps and a command", s.Name)
			}
			for _, step := range s.Steps {
				if strings.TrimSpace(step) == "" {
					return fmt.Errorf("routine %q has an empty step", s.Name)
				}
			}
		} else if strings.TrimSpace(s.Command) == "" {
			return fmt.Errorf("script %q has no command", s.Name)
		}
		if strings.TrimSpace(s.Description) == "" {
			return fmt.Errorf("script %q has no description — the description is what the matcher matches against", s.Name)
		}
		slots := map[string]bool{}
		for _, slot := range s.Slots {
			if slots[slot] {
				return fmt.Errorf("script %q declares slot %q twice", s.Name, slot)
			}
			slots[slot] = true
		}
		for _, rel := range s.Relations {
			if !KnownRelation(rel.Name) {
				return fmt.Errorf("script %q declares unknown relation %q (known: %s)", s.Name, rel.Name, strings.Join(RelationNames(), ", "))
			}
			if rel.Slot != "" && !slots[rel.Slot] {
				return fmt.Errorf("script %q declares relation %q on slot %q, which it does not declare", s.Name, rel.Name, rel.Slot)
			}
		}
		for slot, typ := range s.SlotTypes {
			if !slots[slot] {
				return fmt.Errorf("script %q declares a type for slot %q, which it does not declare", s.Name, slot)
			}
			if !KnownValueType(typ) {
				return fmt.Errorf("script %q declares unknown value type %q for slot %q", s.Name, typ, slot)
			}
		}
	}
	return nil
}

// Save writes the registry back to path (used when a script
// is added at runtime: the addition is a file edit, so it
// persists and is visible to the next program that loads
// the registry).
func (r *Registry) Save(path string) error {
	raw, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}

// Add appends a declaration after validating the whole
// registry still holds together.
func (r *Registry) Add(d Declaration) error {
	r.Scripts = append(r.Scripts, d)
	if err := r.Validate(); err != nil {
		r.Scripts = r.Scripts[:len(r.Scripts)-1]
		return err
	}
	return nil
}

// Find returns the declaration with the given name.
func (r *Registry) Find(name string) (Declaration, bool) {
	for _, s := range r.Scripts {
		if s.Name == name {
			return s, true
		}
	}
	return Declaration{}, false
}

// Names returns the script names in registry order.
func (r *Registry) Names() []string {
	out := make([]string, len(r.Scripts))
	for i, s := range r.Scripts {
		out[i] = s.Name
	}
	return out
}

// sortedNames is Names, sorted — for deterministic output
// in tests and diagnostics.
func (r *Registry) sortedNames() []string {
	out := r.Names()
	sort.Strings(out)
	return out
}
