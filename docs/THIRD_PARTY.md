# Third-party components

Vendored source is unchanged; `vendor/SOURCES.json` records each file's SHA-256 and the test suite verifies it.

| Component | Role | Licence |
|---|---|---|
| Parallax Notan | Mass field, value map, construction read | MIT (`vendor/parallax-notan-LICENSE`) |
| Visual Thinking Lens | Frozen Canon 3 v8 kernel and colour coordinates | MIT (`vendor/vtl-LICENSE`) |
| imgsym | HOG symmetry along the selected axis | MIT (`vendor/imgsym-LICENSE`) |

## Model (downloaded at install, checksum-verified, not in the repo)

| Model | Role | Licence |
|---|---|---|
| U²-NetP (rembg release) | Foreground map for placement | Architecture Apache-2.0; the converted file has no model-specific licence record |

`src/composition_reader/data/focal-models.json` lists URLs, hashes and sizes. Application licensing does not override model licences.
