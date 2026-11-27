package cmdmatch

import "testing"

func TestCousinsAccessor(t *testing.T) {
	// The generator-facing surface: written forms, table
	// order, the word itself excluded, inflections seen
	// through the stem.
	got := Cousins("renames")
	want := []string{"change", "retitle", "relabel"}
	if len(got) != len(want) {
		t.Fatalf("Cousins(renames) = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Cousins(renames) = %v, want %v", got, want)
		}
	}
	if got := Cousins("photos"); len(got) != 3 {
		t.Errorf("Cousins(photos) = %v, want the photo family's other three", got)
	}
	if got := Cousins("zebra"); got != nil {
		t.Errorf("Cousins(zebra) = %v, want nil for a word in no family", got)
	}
}
