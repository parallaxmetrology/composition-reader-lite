# Validation record — 20 November 2026

## External evaluation update

The subsequent frozen [PICD/CADB audit](EXTERNAL_VALIDATION_RESULTS.md) analyzed 950 CADB test images and 1,534 PICD images, with two documented oversized-input rejections. Including 16 additional CADB negatives and 288 control/baseline jobs, it produced 2,788 full-engine reports. The separate expert-anchor diagnostic covers all 603 CADB positive placement labels. This supplies category-specific evidence and reveals serious limitations; it is not a blanket validation of the atlas. The complete local illustrated report is `output/validation-picd-cadb-v1/report.html`.

**27 automated tests passed** after adding audit checks for tied ranking metrics, deterministic pixel-preserving controls and the distinction between raw and UI decisions. The engine itself was unchanged during the external audit. The initial engineering record below is retained for provenance.

## Executed checks

- **24 automated tests passed** in the complete test run (41.31 seconds on this machine). Tests exercise actual image pixels, geometric scorers, unchanged vendor calls, API lifecycle/persistence and labeled evaluation. The storage-default follow-up also passed the service tests.
- All six existing local study images were analyzed with 64 controls and blur/resampling checks. The crow baseline also ran the standard ISR region/dependency read. All returned 22 records and a finite real `imgsym` HOG score. Runs took approximately 2.4–6.9 seconds in the recorded validation pass; initial imports/font-cache creation can take longer. Exact times and source hashes are in `example-validation.json`.
- The known-answer cases cover uniform black/gray/white images, subthreshold noise, centered versus thirds-positioned shapes, spread triangles versus collinear anchors, bilateral symmetry, converging versus parallel lines, dense-edge spiral false positives, corrected/omitted anchors, crop geometry, alpha and EXIF handling, malformed inputs, aspect-correct perpendiculars, and the 14-line armature's 37 crossings.
- The geometric null is deterministic and the family maximum tail is no smaller than its individual-method tail. Dense edge fields do not pass the spiral shift control merely because raw support is high.
- The benchmark tests execute real images, check confusion metrics, omit unknown labels and reject byte duplicates. No classifier is stubbed in these checks.
- All copied source SHA-256 values match `vendor/SOURCES.json`.
- The package builds as a wheel. A separate target installation imports Composition Reader and all four vendor packages from that target, runs a real blank-image analysis, loads all 82 catalog entries and includes its web assets, provenance manifest and three upstream MIT notices.
- The browser workflow was exercised at desktop size and 390-pixel width. The narrow viewport had no horizontal overflow. Actual analysis, field switching, ISR graph display, atlas filtering, before/after comparison, manual anchor rerun and crop rerun were observed working. A manually placed anchor produced one recorded anchor; the crop produced a 280 × 427 analysis raster.
- JavaScript syntax and both launcher scripts pass their syntax checks. The FastAPI application binds only to loopback; foreign browser origins and unknown hosts are rejected by tested boundaries.

The test environment emits one upstream Starlette/httpx deprecation warning. It does not fail the service tests. The full dependency versions are in `requirements-tested.txt`.

## Artifacts

`output/example-validation` contains six standalone HTML evidence reports, six JSON reports, the five actual raster fields per study, and three comparison JSON files. Rerun with:

```sh
.venv/bin/python scripts/validate_examples.py
```

Add `--store` to also place those fresh reports in the workbench history. Existing reports are retained. Engine source hashes are frozen at process startup so a running development process cannot stamp old loaded code with a newer source hash.

HTML report structure and escaped data were checked by the API tests. Browser security policy does not allow opening local `file:` artifacts in the automated browser, so offline HTML visual rendering was not verified there. The live workbench was visually verified; `docs/workbench-preview.jpg` records its final evidence view. Browser-side PNG export was invoked without console errors, but the automated download wait stalled and a saved download file was not confirmed. Server-generated JSON, session, HTML and comparison CSV were verified through API tests.

## What these results establish

This is implemented software with engineering checks and inspectable evidence. These tests establish expected responses on deliberately constructed cases and successful execution on the existing study images. They do **not** establish expert-category accuracy, perceptual validity, causal eye movement, historical construction, or an aesthetic quality score.

At the initial engineering milestone, the local study images were not a labeled holdout benchmark and no PICD/CADB evaluation had run. The external audit linked above now adds conditional category metrics and confidence intervals. Observer agreement, broad cross-domain robustness and learned-segmentation accuracy remain unestablished. A high symmetry or circle fit can describe one object; a high geometric-control result can still be an incidental composition-level resemblance.

The next research gate is to repair the demonstrated selection/weighting and specificity failures on development data, then test a new untouched holdout. Per-image reports continue to expose raw support, controls, sensitivity and limitations rather than calibrated category probabilities.
