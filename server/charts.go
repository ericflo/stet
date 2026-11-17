package main

// Charts are drawn on the server as inline SVG: no chart library, nothing to load, and
// they print. Marks take their colour from the stylesheet (--mark for the thing being
// shown, --mark-neg for a loss against the baseline), text wears the page's text colours,
// and every mark carries a data-tip that site.js turns into a tooltip.

import (
	"fmt"
	"html/template"
	"math"
	"strings"
	"time"
)

type svg struct{ strings.Builder }

func (b *svg) f(format string, args ...any) { fmt.Fprintf(&b.Builder, format, args...) }

func (b *svg) open(w, h int, class, label string) {
	// A chart's text shrinks with it. It may shrink a little; past that the page lets it
	// scroll sideways instead (.figure .plot), so the labels stay readable on a phone.
	floor := ""
	if class == "chart" {
		floor = fmt.Sprintf(` style="min-width:%dpx"`, w*17/20)
	}
	b.f(`<svg class="%s" viewBox="0 0 %d %d" width="%d" height="%d"%s role="img" aria-label="%s" preserveAspectRatio="xMidYMid meet">`,
		class, w, h, w, h, floor, template.HTMLEscapeString(label))
}

func (b *svg) done() template.HTML {
	b.WriteString(`</svg>`)
	return template.HTML(b.String())
}

func esc(s string) string { return template.HTMLEscapeString(s) }

// niceStep picks a round tick spacing giving about n ticks across span.
func niceStep(span float64, n int) float64 {
	if !(span > 1e-9) || math.IsInf(span, 0) {
		return 1 // no span worth dividing, or not a number at all
	}
	raw := span / float64(n)
	mag := math.Pow(10, math.Floor(math.Log10(raw)))
	switch f := raw / mag; {
	case f <= 1:
		return mag
	case f <= 2:
		return 2 * mag
	case f <= 5:
		return 5 * mag
	}
	return 10 * mag
}

func tickLabel(v, step float64) string {
	prec := 0
	if step < 1 {
		prec = int(math.Ceil(-math.Log10(step)))
	}
	s := fmt.Sprintf("%.*f", prec, v)
	if v < 0 {
		s = "−" + s[1:]
	}
	return s
}

// sparkSVG is a row's history at a glance: its gain in each round it has run, against the
// zero line of the unedited baseline.
func sparkSVG(gains []float64) template.HTML {
	const w, h, pad = 96, 26, 3
	if len(gains) == 0 {
		return ""
	}
	lo, hi := 0.0, 0.0
	for _, g := range gains {
		lo, hi = math.Min(lo, g), math.Max(hi, g)
	}
	if hi-lo < 1e-9 {
		hi = lo + 1
	}
	y := func(v float64) float64 { return pad + (hi-v)/(hi-lo)*(h-2*pad) }
	x := func(i int) float64 {
		if len(gains) == 1 {
			return w - pad
		}
		return pad + float64(i)/float64(len(gains)-1)*(w-2*pad)
	}
	var b svg
	b.open(w, h, "spark", fmt.Sprintf("gain over %d rounds", len(gains)))
	b.f(`<line class="axis" x1="0" x2="%d" y1="%.1f" y2="%.1f"/>`, w, y(0), y(0))
	if len(gains) > 1 {
		b.WriteString(`<polyline class="ln" points="`)
		for i, g := range gains {
			b.f("%.1f,%.1f ", x(i), y(g))
		}
		b.WriteString(`"/>`)
	}
	last := len(gains) - 1
	b.f(`<circle class="dot" cx="%.1f" cy="%.1f" r="2.4"/>`, x(last), y(gains[last]))
	return b.done()
}

// ticks lists the gridline values from lo to hi. It counts them rather than adding step
// until hi is passed: the numbers come from graders, and adding a step too small to change
// a float64 would never end.
func ticks(lo, hi, step float64) []float64 {
	n := (hi - lo) / step
	if !(n >= 1) {
		n = 1
	}
	n = math.Min(math.Round(n), 40)
	out := make([]float64, 0, int(n)+1)
	for i := 0; i <= int(n); i++ {
		out = append(out, lo+float64(i)*step)
	}
	return out
}

// frame draws the plot area's horizontal rules and y labels, and returns the y scale.
func (b *svg) frame(w, h, left, right, top, bottom int, lo, hi float64, unit string) func(float64) float64 {
	step := niceStep(hi-lo, 4)
	lo, hi = math.Floor(lo/step)*step, math.Ceil(hi/step)*step
	if hi == lo {
		hi = lo + step
	}
	y := func(v float64) float64 { return float64(top) + (hi-v)/(hi-lo)*float64(h-top-bottom) }
	for _, v := range ticks(lo, hi, step) {
		cls := "grid"
		if math.Abs(v) < step/1e6 {
			cls = "axis"
		}
		b.f(`<line class="%s" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>`, cls, left, w-right, y(v), y(v))
		b.f(`<text x="%d" y="%.1f" text-anchor="end">%s</text>`, left-8, y(v)+4, tickLabel(v, step))
	}
	if unit != "" {
		b.f(`<text x="%d" y="%d" text-anchor="start">%s</text>`, 0, 11, esc(unit))
	}
	return y
}

// recordChart is the record's progression: each time it changed hands, and the rating it
// changed hands at.
func recordChart(recs []RecordRow) template.HTML {
	if len(recs) == 0 {
		return ""
	}
	const w, h, left, right, top, bottom = 720, 250, 46, 24, 22, 30
	lo, hi := 0.0, 0.0
	for _, r := range recs {
		lo, hi = math.Min(lo, r.Rating), math.Max(hi, r.Rating)
	}
	hi += (hi - lo) * 0.12
	t0, t1 := recs[0].At, now()
	if t1-t0 < 3600 {
		t1 = t0 + 3600
	}
	x := func(t int64) float64 { return left + float64(t-t0)/float64(t1-t0)*float64(w-left-right) }
	var b svg
	b.open(w, h, "chart", "The record over time")
	y := b.frame(w, h, left, right, top, bottom, lo, hi, "rating, nats better than unedited")
	b.WriteString(`<path class="ln" d="`)
	for i, r := range recs {
		if i == 0 {
			b.f("M%.1f,%.1f", x(r.At), y(r.Rating))
		} else {
			b.f(" H%.1f V%.1f", x(r.At), y(r.Rating))
		}
	}
	b.f(` H%.1f"/>`, x(t1))
	for i, r := range recs {
		tip := fmt.Sprintf("%s v%d by %s · %s nats · %s", r.Name, r.Version, r.Owner, signed(r.Rating, 3), time.Unix(r.At, 0).UTC().Format("2 Jan 2006"))
		b.f(`<circle class="dot ring" cx="%.1f" cy="%.1f" r="4.5" data-tip="%s"/>`, x(r.At), y(r.Rating), esc(tip))
		b.f(`<circle class="hit" cx="%.1f" cy="%.1f" r="14" data-tip="%s"/>`, x(r.At), y(r.Rating), esc(tip))
		if i == len(recs)-1 || len(recs) <= 4 {
			anchor, dx := "start", 8.0
			if x(r.At) > float64(w)*0.7 {
				anchor, dx = "end", -8.0
			}
			b.f(`<text class="lbl" x="%.1f" y="%.1f" text-anchor="%s">%s</text>`, x(r.At)+dx, y(r.Rating)-9, anchor, esc(clip(r.Name, 28)))
		}
	}
	b.f(`<text x="%d" y="%d" text-anchor="start">%s</text>`, left, h-8, time.Unix(t0, 0).UTC().Format("2 Jan 2006"))
	b.f(`<text x="%d" y="%d" text-anchor="end">now</text>`, w-right, h-8)
	return b.done()
}

// gainChart shows an editor's gain in each of its rounds, oldest first. Bars above the
// zero line beat the unedited corpus; bars below made it worse.
func gainChart(entries []EntryView) template.HTML {
	var rows []EntryView
	for i := len(entries) - 1; i >= 0; i-- { // entries come newest first
		if entries[i].Settled {
			rows = append(rows, entries[i])
		}
	}
	if len(rows) == 0 {
		return ""
	}
	const w, h, left, right, top, bottom = 720, 230, 46, 12, 22, 28
	lo, hi := 0.0, 0.0
	for _, r := range rows {
		lo, hi = math.Min(lo, r.Gain), math.Max(hi, r.Gain)
	}
	if hi-lo < 1e-9 {
		hi = 0.1
	}
	var b svg
	b.open(w, h, "chart", "Gain in each round")
	y := b.frame(w, h, left, right, top, bottom, lo, hi, "gain, nats")
	slot := float64(w-left-right) / float64(len(rows))
	bar := math.Min(24, math.Max(3, slot-2))
	for i, r := range rows {
		cx := left + slot*(float64(i)+0.5)
		tip := fmt.Sprintf("%s · %s nats", r.Round, signed(r.Gain, 3))
		if !r.Valid {
			tip = r.Round + " · invalid, scored as unedited"
			b.f(`<circle class="void" cx="%.1f" cy="%.1f" r="3.5"/>`, cx, y(0))
		} else {
			cls, y0, y1 := "bar", y(r.Gain), y(0)
			if r.Gain < 0 {
				cls, y0, y1 = "bar neg", y(0), y(r.Gain)
			}
			b.WriteString(barPath(cls, cx-bar/2, y0, bar, math.Max(1, y1-y0), r.Gain >= 0))
		}
		b.f(`<rect class="hit" x="%.1f" y="%d" width="%.1f" height="%d" data-tip="%s"/>`, cx-slot/2, top, slot, h-top-bottom, esc(tip))
	}
	b.f(`<text x="%d" y="%d" text-anchor="start">%s</text>`, left, h-8, esc(rows[0].Round))
	if len(rows) > 1 {
		b.f(`<text x="%d" y="%d" text-anchor="end">%s</text>`, w-right, h-8, esc(rows[len(rows)-1].Round))
	}
	return b.done()
}

// barPath is a bar with its data end rounded and its base square on the baseline.
func barPath(class string, x, y, w, h float64, up bool) string {
	r := math.Min(4, math.Min(w/2, h))
	if up {
		return fmt.Sprintf(`<path class="%s" d="M%.1f,%.1f V%.1f Q%.1f,%.1f %.1f,%.1f H%.1f Q%.1f,%.1f %.1f,%.1f V%.1f Z"/>`,
			class, x, y+h, y+r, x, y, x+r, y, x+w-r, x+w, y, x+w, y+r, y+h)
	}
	return fmt.Sprintf(`<path class="%s" d="M%.1f,%.1f V%.1f Q%.1f,%.1f %.1f,%.1f H%.1f Q%.1f,%.1f %.1f,%.1f V%.1f Z"/>`,
		class, x, y, y+h-r, x, y+h, x+r, y+h, x+w-r, x+w, y+h, x+w, y+h-r, y)
}

// sourceChart compares the exam loss on each kind of text: the unedited corpus (hollow)
// and this editor's (filled), joined by the distance between them.
func sourceChart(sources []string, base, edited map[string]float64) template.HTML {
	if len(base) == 0 || len(edited) == 0 {
		return ""
	}
	// Drawn at the width of the half-page column it sits in, so the text is not scaled down.
	const w, rowH, left, right, top, bottom = 500, 32, 112, 66, 30, 28
	h := top + bottom + rowH*len(sources)
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, s := range sources {
		for _, v := range []float64{base[s], edited[s]} {
			if v > 0 {
				lo, hi = math.Min(lo, v), math.Max(hi, v)
			}
		}
	}
	if math.IsInf(lo, 0) {
		return ""
	}
	step := niceStep(hi-lo, 5)
	lo, hi = math.Floor(lo/step)*step, math.Ceil(hi/step)*step
	if hi == lo {
		hi = lo + step
	}
	x := func(v float64) float64 { return left + (v-lo)/(hi-lo)*float64(w-left-right) }
	var b svg
	b.open(w, h, "chart", "Exam loss by kind of text, unedited against this curator")
	for _, v := range ticks(lo, hi, step) {
		b.f(`<line class="grid" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>`, x(v), x(v), top-6, h-bottom+4)
		b.f(`<text x="%.1f" y="%d" text-anchor="middle">%s</text>`, x(v), h-10, tickLabel(v, step))
	}
	b.f(`<circle class="hollow" cx="%d" cy="12" r="4.5"/><text x="%d" y="16">unedited</text>`, left+6, left+16)
	b.f(`<circle class="dot" cx="%d" cy="12" r="4.5"/><text x="%d" y="16">this curator</text>`, left+96, left+106)
	for i, s := range sources {
		cy := float64(top + rowH*i + rowH/2)
		bv, ev := base[s], edited[s]
		tip := fmt.Sprintf("%s · unedited %.3f, this curator %.3f (%s)", s, bv, ev, signed(bv-ev, 3))
		b.f(`<text class="lbl" x="%d" y="%.1f" text-anchor="end">%s</text>`, left-14, cy+4, esc(s))
		cls := "link"
		if ev > bv {
			cls = "link neg"
		}
		b.f(`<line class="%s" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/>`, cls, x(bv), x(ev), cy, cy)
		b.f(`<circle class="hollow" cx="%.1f" cy="%.1f" r="4.5"/>`, x(bv), cy)
		b.f(`<circle class="dot ring" cx="%.1f" cy="%.1f" r="5"/>`, x(ev), cy)
		gcls := "val"
		if ev > bv {
			gcls = "val neg"
		}
		b.f(`<text class="%s" x="%d" y="%.1f" text-anchor="end">%s</text>`, gcls, w-8, cy+4, signed(bv-ev, 3))
		b.f(`<rect class="hit" x="0" y="%.1f" width="%d" height="%d" data-tip="%s"/>`, cy-rowH/2, w, rowH, esc(tip))
	}
	return b.done()
}

// baselineChart shows how hard each kind of exam text is for the unedited corpus.
func baselineChart(sources []string, base map[string]float64) template.HTML {
	if len(base) == 0 {
		return ""
	}
	const w, rowH, left, right, top, bottom = 720, 30, 150, 70, 8, 26
	h := top + bottom + rowH*len(sources)
	hi := 0.0
	for _, s := range sources {
		hi = math.Max(hi, base[s])
	}
	if hi <= 0 {
		return ""
	}
	step := niceStep(hi, 5)
	hi = math.Ceil(hi/step) * step
	x := func(v float64) float64 { return left + v/hi*float64(w-left-right) }
	var b svg
	b.open(w, h, "chart", "Baseline exam loss by kind of text")
	for _, v := range ticks(0, hi, step) {
		b.f(`<line class="grid" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>`, x(v), x(v), top, h-bottom+4)
		b.f(`<text x="%.1f" y="%d" text-anchor="middle">%s</text>`, x(v), h-8, tickLabel(v, step))
	}
	for i, s := range sources {
		cy := float64(top + rowH*i + rowH/2)
		v := base[s]
		r := 4.0
		b.f(`<text class="lbl" x="%d" y="%.1f" text-anchor="end">%s</text>`, left-14, cy+4, esc(s))
		b.f(`<path class="bar quiet" d="M%.1f,%.1f H%.1f Q%.1f,%.1f %.1f,%.1f V%.1f Q%.1f,%.1f %.1f,%.1f H%.1f Z"/>`,
			x(0), cy-6, x(v)-r, x(v), cy-6, x(v), cy-6+r, cy+6-r, x(v), cy+6, x(v)-r, cy+6, x(0))
		b.f(`<text class="val" x="%.1f" y="%.1f">%.3f</text>`, x(v)+8, cy+4, v)
		b.f(`<rect class="hit" x="0" y="%.1f" width="%d" height="%d" data-tip="%s"/>`, cy-rowH/2, w, rowH, esc(fmt.Sprintf("%s · %.3f nats per token", s, v)))
	}
	return b.done()
}

// curveChart is the student's training loss as it works through the edited corpus.
func curveChart(curve [][]float64) template.HTML {
	var pts [][2]float64
	for _, p := range curve {
		if len(p) >= 2 && finite(p[0]) && finite(p[1]) {
			pts = append(pts, [2]float64{p[0], p[1]})
		}
	}
	if len(pts) < 3 {
		return ""
	}
	const w, h, left, right, top, bottom = 500, 250, 44, 14, 24, 28
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, p := range pts[1:] { // the first step is the untrained model: off the scale and uninformative
		lo, hi = math.Min(lo, p[1]), math.Max(hi, p[1])
	}
	x0, x1 := pts[0][0], pts[len(pts)-1][0]
	if x1 <= x0 {
		return ""
	}
	x := func(v float64) float64 { return left + (v-x0)/(x1-x0)*float64(w-left-right) }
	var b svg
	b.open(w, h, "chart", "Training loss by step")
	y := b.frame(w, h, left, right, top, bottom, lo, hi, "training loss, nats per token")
	ceil := y(hi + (hi-lo)*0.2)
	b.WriteString(`<polyline class="ln" points="`)
	for _, p := range pts {
		b.f("%.1f,%.1f ", x(p[0]), math.Max(ceil, y(p[1])))
	}
	b.WriteString(`"/>`)
	for i, p := range pts {
		xa, xb := x(p[0]), x(p[0])
		if i > 0 {
			xa = (x(pts[i-1][0]) + x(p[0])) / 2
		}
		if i < len(pts)-1 {
			xb = (x(p[0]) + x(pts[i+1][0])) / 2
		}
		b.f(`<rect class="hit" x="%.1f" y="%d" width="%.1f" height="%d" data-tip="%s" data-x="%.1f" data-y="%.1f"/>`,
			xa, top, math.Max(1, xb-xa), h-top-bottom, esc(fmt.Sprintf("step %s · loss %.3f", comma(int64(p[0])), p[1])), x(p[0]), math.Max(ceil, y(p[1])))
	}
	b.f(`<text x="%d" y="%d" text-anchor="start">step %s</text>`, left, h-8, comma(int64(x0)))
	b.f(`<text x="%d" y="%d" text-anchor="end">step %s</text>`, w-right, h-8, comma(int64(x1)))
	return b.done()
}
