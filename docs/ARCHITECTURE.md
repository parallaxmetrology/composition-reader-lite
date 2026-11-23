# Architecture and measurement contract

## One representation, multiple interpretations

```
Original bytes + explicit corrections
        | EXIF, alpha, crop, bounded working scale
        v
Evidence fields: tone, chroma, relative/absolute Notan mass, edges, learned foreground
        |
        v
Scene: separate foreground subjects and tonal masses, contours, lines, frame, diagnostics
        |
        +--> placement / armature / arrangement / symmetry / curve hypotheses
        |        + evidence references + geometry + eligible/insufficient
        |        + controls + sensitivity + explicit limitations
        |
        +--> original VTL kernel and color coordinates at up to 1536 px
        +--> original Notan construction measurements
        +--> optional full ISR with separate islands and dependency readings
        v
Versioned report -> overlays / comparison / exports / labeled evaluation
```

The scene is independent of the catalog. `registry.py` declares stable detector IDs, families, evidence requirements, method versions and control scope. Every run emits all 22 detector records, including insufficient records. The runtime checks that this contract is complete. UI families, exports and comparisons consume the common record shape. New methods add evidence consumers rather than changing the image loader or inventing a new output format.

Catalog entries can map to a detector, a partial source of evidence, a frame context or a future/reference construction. Multiple names can refer to related evidence without pretending to be distinct validated classifiers. Dataset categories require an explicit separate mapping.

## Coordinates and resolutions

- Geometry: top-left origin, x right and y down; normalized [0,1] within the current analysis frame.
- Distances: actual pixel proportions divided by image diagonal. An image's aspect ratio is retained when calculating directions, circles and perpendiculars.
- Analysis: maximum side 512, never upscaled. Notan and detector tolerances are defined at that working scale. Very small images therefore have a different effective squint scale; this remains a documented limitation.
- VTL: the original is independently decoded to at most 1536 pixels before `vtl.measure` and `vtl.measure_color`. The input size is recorded. This avoids quietly substituting the 512-pixel detector input for the canonical kernel input. VTL's centered offsets are kept separate from normalized overlay coordinates.
- ISR extended read: receives the analysis raster. Persistence, consensus-family and saccade options remain disabled; this uses the complete standard read with those expensive extensions explicitly off. Its native IDs, centered coordinates and reading schema remain intact. The UI translates island coordinates only for display.
- Crop: normalized coordinates in the EXIF-corrected source frame, recorded with the actual rounded analysis pixel bounds. Original bytes are preserved. Comparison warns when options, frames or engine implementations differ.

## Evidence extraction

The copied Notan code computes three-scale luminance/chroma mass at sigma 4, 8 and 16 with chroma weight 0.5. Its relative field is contrast-normalized; its absolute field remains physical. A 99th-percentile absolute signal below 0.004 refuses a structural reading rather than amplifying near-flat noise.

The application adapter uses a 0.06 relative mass mask. Watershed seeds come from connected regional maxima with prominence 0.06 after sigma-3 smoothing. Light and dark polarities are split against median tone. Regions below 0.3% image area or 1.2% total field mass are omitted; at most 20 are reported. Reported coverage exposes missing diffuse mass. These are structural regions, not object labels.

Placement now uses a separate, pinned U²-NetP foreground model running locally with ONNX Runtime on CPU. Inputs use the upstream 320×320 RGB normalization. The raw output is not stretched to manufacture confidence. Connected components at output >=0.5 must occupy 0.1–85% of the image; a weak or near-uniform map abstains. At most eight subjects are retained. Bounding-box centers locate the complete foreground components. Their placement weights are proportional to square-root area times mean foreground strength squared. Selection never sees a composition guide. Tonal regions remain unchanged for the other geometry methods. The model file SHA-256 and inference contract accompany each automatic reading.

OpenCV LSD extracts tone and fixed-scale chroma segments after sigma-1.8/2.5 smoothing. Minimum length is 0.065 of the image diagonal. Near-duplicate pose/position pairs are removed, and at most 64 longest segments are retained. Canny luminance edges supply the curve-alignment view. Protected boundary mode halves mass near the outer 2% and ramps to full weight over the next 5%, then sets aside tonal masses with more than 60% of their pixels inside that outer 7% band (a mat or frame otherwise survives as a ring whose centroid lies in the empty centre); bilateral symmetry ramps its gradients from zero at 2% to full at 7% in the same mode. It does not crop the image or suppress line evidence or foreground subjects. Full ISR receives its own original edge-aware mode.

Manual anchors replace automatic anchor positions for anchor-based hypotheses. Their weights come from local Notan mass neighborhoods. They do not rewrite the mass field, edge field, or VTL/ISR outputs. Near-empty neighborhoods receive a small weight floor. Omitted automatic labels refer to that extraction; region identities can change under perturbation, so omission sensitivity should be interpreted cautiously.

## Measurement definitions

| Family | Actual calculation | Interpretation boundary |
|---|---|---|
| Placement | Foreground-weighted Gaussian affinity to fixed intersections (sigma 0.045 diagonal); box thirds also tests grid lines (sigma 0.035 frame units). Thirds can substitute the focal path below. | Source and geometry are explicit; face/sky inference is local, with no gaze or identity inference |
| Armature | Segment midpoint distance (sigma 0.025 diagonal) and orientation (sigma 0.14 radians), length-weighted | Alignment, not evidence an artist used that construction |
| Direction | Length-weighted orientation affinity; sigma 0.15 radians | Long edges, not implied narrative motion |
| Triangle | Best of top-12-anchor triples: mass coverage × triangular spread × area gate × weight-balance factor | Favors well-spread triples; scalene/narrow intentional triangles can score lower |
| Bilateral | Opposite reflected normal gradients at two luminance-detail scales, with transverse-energy and boundary penalties; 36 coarse axes plus bounded refinements; fit multiplied by a frame-coverage gate (0 at <=50% of the frame having a mirror partner, 1 at >=90%) | Broad parallel tonal bands are insufficient; an off-centre symmetric object is reported as subject symmetry (diagnostic), not frame symmetry; semantic pairing still differs |
| Texture symmetry | Unchanged `imgsym.score_along_axis(..., method='hog')` along the chosen structural axis | Separate supplementary HOG result, never averaged into the structural score |
| Circle | Least-squares circle on selected Notan contours, residual relative to radius and angular coverage | Circular boundary, not a whole-image narrative |
| Spiral | Eight fixed true logarithmic spirals, growth 2 log(phi)/pi, arclength-weighted edge proximity | Restricted frame templates; alignment does not establish golden-ratio intent |
| Convergence | Pair-derived intersections among top 32 segments; support from at least 3 with direction diversity | Convergence alone cannot distinguish perspective from radial rays |
| Counterweight | Lead mass = greatest moment (tonal-mass share x offset from frame centre); fit = share of that moment cancelled by one opposing mass x lead-offset gate (full at 0.15 diagonal) x asymmetry (1 minus weight ratio x distance ratio of lead and counter) | Even spread is not one counterweight; equal twins at equal distances are symmetry; weight is tonal mass, not colour, isolation or subject interest |
| Negative space | Fraction of Notan mass field below 0.06 | Flat field, not necessarily background or intentional empty space |
| Radial | Mass-gradient squared-cosine agreement with rays from mass centroid; fit = departure from isotropic 0.5 in either direction (concentric or radiating) x reach gate (gradient-weighted mean radius / 0.15 diagonal) x surround gate (share of 12 direction sectors holding structure / 0.75) | A large round body reads concentric; convergent perspective lines read radiating |
| Quadrant | Four mass fractions | Diagnostic distribution, not a composition class |

Frame ratios compare long-side/short-side ratio to sqrt(2), sqrt(3), 2, sqrt(5), phi and 3:2, with a 1.5% relative-error match window. Internal dynamic-symmetry constructions are not inferred from a matching frame.

## Controls, sensitivity and statuses

For non-placement geometry, the control preserves anchor count/weights and line lengths, randomizes positions and permitted segment orientations, keeps complete segments within the frame, and repeats the full search for each sample. Both per-method and within-family maximum tails are returned. A supported status requires fit >= 0.5 and family-maximum tail <= 0.1; fit >= 0.45 without that evidence is ambiguous. These thresholds are engineering choices, not fitted human-label probabilities.

Spiral controls preserve the edge texture under random toroidal translations and repeat all eight variants for each translation. Its control therefore addresses incidental support in dense textures. It does not test other curve families or an artist's process.

Placement v0.3.1 conditions translations on each full subject bounding-box footprint and observed frame contacts: counts, sizes and weights are fixed and boxes stay inside the frame. Anchor offsets within boxes are preserved. A nonzero footprint within one working pixel of a frame edge is pinned on that axis; zero-area manual anchors do not imply clipping. The per-guide upper-tail ranks of observed plus null draws are compared against the minimum rank across all four placement guides. This avoids letting the denser fifths grid set a shared raw-fit threshold. Ties use conservative upper ranks. The raw family maximum remains a diagnostic, while `adjusted_tail_fraction` drives support. Placement uses at least 256 controls when controls are requested, with independent random generator seed 100176 (base seed 173 plus 100003), leaving non-placement random draws unchanged.

Supported placement requires fit >=0.5, adjusted family tail <=0.1, and excess over the conditional null mean >=0.10. Centering additionally needs each one-dimensional axis affinity to exceed its footprint-conditioned mean by >=0.05. Thirds can be descriptive at fit >=0.65 and excess over its null mean >=0.20; the UI explicitly states that coincidence has not been ruled out. A descriptive match is not evidence of significance. Thirds line support has a 0.85 base with an independent aspect/extent boost up to 1.0; the best intersection or axis support is selected per subject and the full search repeats in every control. The original gates were selected on older audits; the frame-contact correction was frozen before the independent v4 holdout.

The inherited box controls translate components independently and use boxes as footprint approximations. They do not model occlusion, object identity, eye position or relationships between fragmented components. Large objects can have genuine intentional centering that this conservative null cannot distinguish from constrained placement.

The finite-sample tail is `(1 + count(control >= observed)) / (samples + 1)`, seed 173. Default 64 requested samples; 32/128/256 are available. Placement enforces the 256-sample minimum described above. Zero disables geometric controls and leaves descriptive fits only. These are explicit geometric nulls, not natural-image null distributions. They do not account for every researcher's choice across all families, parameters or crop attempts.

Placement, quick geometry methods and structural symmetry are re-extracted after blur sigma 1.4 and a 75% down/up sample. Maximum fit change >0.15 marks a supported result, or a descriptive placement/structural-symmetry result, as sensitive. Placement also becomes sensitive when the constrained subject weight changes by more than 0.10 on either axis under the perturbations. Both fit and contact changes are reported. Other methods report no sensitivity check rather than implying one was performed. Other descriptive methods use fit >=0.55 for display emphasis; structural symmetry uses its development-selected 0.45 threshold. Unsupported/insufficient results remain visible. Raw fits across different families are not commensurate and are not combined into a quality grade.

## Extending without changing the foundation

1. Define the observable claim and its counterexamples; map catalog names separately.
2. Add a `DetectorSpec` with a stable ID, requirements, family, version and control scope.
3. Implement a scene consumer returning geometry, supporting element IDs, parameters, eligibility and limitations through `result`.
4. Add it to evaluation and its declared control/sensitivity schedule. Report unavailable checks explicitly.
5. Add independent known-answer and negative fixtures. Freeze dataset mappings and thresholds before holdout evaluation.
6. The existing UI, report export, comparison and benchmark runner consume the new record. Extend geometry only when the evidence really requires a new primitive.

U²-NetP foreground inference is integrated for placement with explicit model provenance. Other segmentation or DeepLSD implementations can later supplement extraction with their own versioned contracts. They should not silently change what an existing versioned detector means. The rule-of-thirds GitHub model, VFCNet and PhotoGuider are possible separately evaluated adapters; they are not installed or represented as active dependencies in this build.

The armature constructor follows the [dedicated 14-line construction](https://gridmakerpro.com/grids/advanced-composition/armature-14-line/): two diagonals, eight corner-to-opposite-midpoint segments and four midpoint diamond edges. It has 37 unique crossings. Its dedicated definition is more precise than the catalog overview’s loose mention of midlines. An independent intersection test checks this geometry.

The v0.2 symmetry measurement searches 36 coarse axes and 81 bounded local refinements. Its 0.45 descriptive-fit threshold was selected on the previous CADB audit for low false-positive rate, before the fresh holdout lock. The inherited 0.55 threshold applies to other descriptive measurements. See V2_VALIDATION.md for the development tradeoff.


## Lite build

This build has no face, horizon, object-detection or subject-selection layer. Placement readings use the foreground map from the single U²-NetP model, weighted by area and strength. ISR is not included.

## Fixes of 22 Nov 2026

- Bilateral symmetry is gated by frame coverage; the best axis with no coverage requirement is reported as subject symmetry (diagnostic).
- Radial organization counts radiating as well as concentric structure, with reach and surround requirements.
- Counterweight is the lever reading above; it no longer saturates when removing any one large mass shifts the balance.
- The diagonal-and-perpendiculars template (golden triangle) is tested in both mirror forms; the better one is reported.
- A foreground component is set aside as sky when it touches the top edge and a side (or spans half the width), does not reach the bottom, has its centre in the upper 40% and is brighter than the picture median; it is recorded in the subject selection and a warning.
- When no reading passes, the report lists up to three near misses (eligible, fit >= 0.45, ambiguous or sensitive) with the rule that held each back. The readable export includes the summary and near misses.
- Spiral readings carried by texture say so when shifted copies of the edge field fit as well in more than half of trials.
