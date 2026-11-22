package brain

import "testing"

// The .latest checkpoint cadence: cadence 1 (and any
// unset/invalid value) keeps the historical every-epoch
// behaviour; higher cadences write only on multiples.
func TestLatestDue(t *testing.T) {
	cases := []struct {
		epoch, every int
		want         bool
	}{
		{1, 1, true},
		{2, 1, true},
		{7, 1, true},
		{1, 0, true},  // unset means every epoch
		{5, -3, true}, // invalid means every epoch
		{1, 10, false},
		{9, 10, false},
		{10, 10, true},
		{11, 10, false},
		{20, 10, true},
		{19, 20, false},
		{20, 20, true},
		{40, 20, true},
	}
	for _, c := range cases {
		if got := latestDue(c.epoch, c.every); got != c.want {
			t.Errorf("latestDue(%d, %d) = %v, want %v", c.epoch, c.every, got, c.want)
		}
	}
}
