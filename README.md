# corotations

Finding where a galaxy's spiral arms and its stars turn in step — automatically, from two images.

This is the reference implementation of an unsupervised, purely photometric pipeline that
measures the **corotation radius** (R_CR) of a spiral galaxy: the radius at which the spiral
pattern's angular speed matches that of the disk material.

The method follows the colour-gradient framework of Puerari & Dottori (1997). Young stars
(blue band) and old stars (red band) are azimuthally offset from one another on opposite sides
of corotation, so the sign of that offset flips as the radius crosses R_CR. The pipeline
measures the offset as a function of radius and reports every sign change as a candidate.

## How it works

1. **Deprojection.** Both bands are resampled onto a common sky grid and deprojected with a
   single geometric transform derived from the second moments of the longer-wavelength image.
   Using one transform for both bands means residual deprojection error cancels in the phase
   difference.
2. **Azimuthal profiles.** The disk is sampled with 34 annuli spanning `0.08–1.25 R₂₅`, and the
   mean surface brightness in each is binned into 120 azimuthal bins. Rings touching any pixel
   outside the real image footprint are skipped, leaving a median of 33 usable rings per galaxy.
3. **Phase fitting.** Each profile is fitted independently with `A·cos(mθ + c) + D` at `m = 2`.
   The two bands are fitted from their own starting points — seeding one with the other would
   start the optimizer at the condition being tested. Fits are canonicalized to `A > 0` and the
   phase difference is wrapped into `(−π/m, +π/m]`.
4. **Candidate detection.** `Δφ(R) = c_blue − c_red` is scanned for sign changes between
   adjacent radii; each bracketing pair gives one candidate, located by linear interpolation.
   Candidates are kept in `0.15 < R/R₂₅ < 0.60`, outside which the fitted phase is poorly
   constrained by the bulge or the sky background.
5. **Uncertainty.** Each candidate gets an interval spanning the radii over which `|Δφ|` stays
   below a threshold that widens where the curve flattens, so a shallow crossing is bracketed
   loosely and a steep one tightly.

Band pairs are SDSS *g* − *i* where available, and DSS2 Blue − WISE 3.4 µm otherwise.

## Reproducing the published numbers

The scored measurement set is committed, so the benchmark runs with no downloads:

```bash
pip install numpy pandas scipy
python3 recall_test.py
```

This writes `results/recall_test.csv` and prints:

| method | n | gold radii | found | recall | purity | F₁ | completeness |
|---|---|---|---|---|---|---|---|
| This work | 90 | 195 | 124 | 63.6% ± 3.8 | 52.5% | 0.58 ± 0.03 | 80.0% |
| Buta & Zhang (2009) | 40 | 78 | 45 | 57.7% ± 6.5 | 35.9% | 0.44 ± 0.05 | 72.5% |
| Abdeen et al. (2020) | 16 | 40 | 21 | 52.5% ± 8.5 | 56.2% | 0.54 ± 0.09 | 68.8% |
| Williams et al. (2021) | 19 | 37 | 15 | 40.5% ± 10.3 | 47.5% | 0.44 ± 0.10 | 47.4% |
| Ruiz-García et al. (2024) | 21 | 39 | 15 | 38.5% ± 6.4 | 52.4% | 0.44 ± 0.08 | 52.4% |
| Elmegreen & Elmegreen (1995) | 44 | 102 | 30 | 29.4% ± 4.6 | 51.1% | 0.37 ± 0.05 | 50.0% |
| Rautiainen & Laurikainen (2008) | 19 | 35 | 9 | 25.7% ± 8.7 | 47.4% | 0.33 ± 0.10 | 47.4% |

### How the benchmark works

Methods are scored against a reference built **only from published measurements**, never from
this pipeline's output. A galaxy enters the reference when the intervals reported by two or more
independent papers intersect; the reference value is the midpoint of that intersection. A
prediction counts as correct when it falls within `0.10 R₂₅` of a reference value — a criterion
that depends on where a radius is placed, not on how wide an uncertainty the method quotes, so
it treats interval-reporting and point-reporting methods alike.

Every published method is scored **leave-one-out**: its own measurements are removed from the
reference before it is evaluated, so no method is graded against itself.

- **Completeness** — fraction of galaxies with at least one correct R_CR (per galaxy).
- **Recall** — fraction of reference radii recovered (per radius).
- **Purity** — fraction of predicted radii that are correct (per radius).
- **F₁** — harmonic mean of recall and purity, high only when a method finds most known
  resonances *and* is reliable about it.

Two caveats worth stating plainly. All three rates are **conditional on coverage**: the
denominator counts only the galaxies a method actually reports on, which is 90 here and 16–44
for the published methods, so each method is measured on the sample it chose. And the errors
are wide — the F₁ and recall leads above are not separable from second place at these sample
sizes. Errors are bootstrap over galaxies, 2000 resamples.

## Re-running the measurements from images

Only needed to rebuild `run/v2_g_wide.json` from scratch; requires `astropy`, `opencv-python`
and `astroquery`.

```bash
export FITS_ROOT=/path/to/fits          # defaults to ./fits
python3 fetch_r25.py                    # R₂₅ from HyperLEDA/Vizier -> data/r25_pgc.csv
python3 download_wide.py                # SkyView cutouts, 3 R₂₅ across -> $FITS_ROOT/wide_FITS
python3 corotation_pipeline2.py --blue g --out run/v2_g_wide.json
python3 recall_test.py
```

## Layout

```
corotation_pipeline.py    v1 pipeline; v2 imports its I/O helpers and path roots
corotation_pipeline2.py   measurement: deprojection, annuli, per-band cosine fits
select_v2.py              candidate selection and uncertainty intervals
eval_lib.py               gold standard construction and metrics
eval_fixed.py             de-duplicates the literature table (see below)
score_est.py              literature tables and interval helpers
recall_test.py            the benchmark -> results/recall_test.csv
download_wide.py          SkyView cutout fetch
fetch_r25.py              R₂₅ lookup
split.csv                 galaxy -> SDSS / non-SDSS band assignment
data/total_crs.csv        the published corotation compilation
data/r25_pgc.csv          R₂₅ per galaxy, arcsec
run/v2_g_wide.json        per-galaxy, per-ring fitted phases — the scored measurement set
run/winding_check.csv     SpArcFiRe chirality, for the optional rotation-sense rule
```

### Settings

`recall_test.py` calls the selector with the published configuration:

```python
predictions(blob, wrap=False, sense=None, arms=False, sigma=False, bar=None, kernel=1)
```

`select_v2.candidates` also implements a phase-wrap guard, a rotation-sense rule, an arm
amplitude cut, covariance-propagated intervals and bar handling. None are applied to the
published results; they are kept because they are the ablations behind that choice. `kernel=1`
means no smoothing of `Δφ` — a three-point average cuts candidates per galaxy from 3.3 to 2.2
but drops galaxy-level agreement from 80.0% to 70.1%.

### A note on the literature table

`data/total_crs.csv` files each measurement under both a catalogue name and an article name.
Read naively this stores some rows twice and invents galaxies from name mismatches.
`eval_fixed.py` rebuilds the tables keying on the catalogue name only, and every script here
imports it before use.

## Paths

No absolute paths. Image directories resolve from `$FITS_ROOT` (default `./fits`), everything
else relative to the repository root.

## Credit

The corotation work this builds on, including the earlier SpArcFiRe-based implementation of the
phase-offset measurement, was developed with Sahel. Galaxy deprojection geometry and chirality
come from [SpArcFiRe](https://github.com/waynebhayes/SpArcFiRe) (Davis & Hayes 2014).

## License

MIT — see [LICENSE](LICENSE).
