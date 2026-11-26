package main

import "os"

// Colour for the program's answers — John's ask
// (2026-11-26): the responses are coloured, the rest is
// not. Plain ANSI, hand-rolled (no colour library, per
// the project's standing rule), and only when the output
// is a real terminal: piped output, test buffers, and
// NO_COLOR all stay plain, so nothing downstream ever
// has to read escape codes.

const (
	ansiLabel = "\x1b[1;36m" // bold cyan: the labels on answer blocks
	ansiGood  = "\x1b[32m"   // green: something was learned
	ansiWarn  = "\x1b[33m"   // yellow: the caution line
	ansiReset = "\x1b[0m"
)

// paint wraps s in a colour when colour is on for this
// demo, and returns it untouched when it is not.
func (d *demo) paint(code, s string) string {
	if !d.color {
		return s
	}
	return code + s + ansiReset
}

// colorEnabled reports whether stdout is a terminal the
// user is looking at (and NO_COLOR is not set).
func colorEnabled() bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	fi, err := os.Stdout.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}
