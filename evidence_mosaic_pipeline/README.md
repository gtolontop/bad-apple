# Bad Apple Official-Image Photomosaic Pipeline

This is a local, cache-heavy Python pipeline for rendering a 1920x1080 Bad Apple-style photomosaic video from official/public Epstein-related government release images.

The intended output path is:

```powershell
outputs/bad_apple_epstein_evidence_mosaic.mp4
```

## Safety And Source Rules

- Use only official/public source material, such as DOJ and House Oversight releases listed in `SOURCES.md`.
- Do not use reposts, social media, YouTube screenshots, Reddit, news article images, or third-party compilations.
- Prefer mundane evidence photos: rooms, houses, objects, landscapes, non-sensitive scene material.
- The script strips EXIF by re-encoding every cached tile.
- The script skips images that appear to contain faces, dense readable text, likely IDs/passports/addresses, or large skin-tone regions.
- The filters are conservative heuristics, not a legal or privacy guarantee. Review your source set before rendering.
- The final video adds only neutral interface text: `OFFICIAL IMAGE REVIEW`, frame count, and dataset count.

## Setup

Install Python 3.10+ and ffmpeg. On Windows, make sure `ffmpeg.exe` is on your `PATH`, or pass its full path with `--ffmpeg`.

One Windows install option:

```powershell
winget install Gyan.FFmpeg
```

```powershell
cd evidence_mosaic_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional OCR filtering uses `pytesseract`, but it also needs the Tesseract OCR binary installed separately. Without that binary, the default `--enable-ocr auto` mode continues with the non-OCR safety filters.

## Inputs

Put these in the project folder:

```text
bad_apple.mp4
source_images/
```

`source_images/` may contain nested folders and `.zip` archives. The pipeline extracts zips into `cache/extracted/` and scans for:

```text
.jpg .jpeg .png .webp
```

## Dry Run

Run this first:

```powershell
python build_mosaic.py --dry-run
```

It scans and filters `source_images/`, creates cached thumbnails in `cache/tiles/`, then reports usable image counts in dark-to-light brightness buckets.

For stricter filtering:

```powershell
python build_mosaic.py --dry-run --strict-filter
```

To require OCR to be available:

```powershell
python build_mosaic.py --dry-run --enable-ocr always
```

## Render

Default 48x27 grid:

```powershell
python build_mosaic.py --video bad_apple.mp4 --images source_images --out outputs/bad_apple_epstein_evidence_mosaic.mp4 --grid 48x27
```

If ffmpeg is installed but not on `PATH`:

```powershell
python build_mosaic.py --ffmpeg "C:\path\to\ffmpeg.exe"
```

Higher-density 64x36 grid:

```powershell
python build_mosaic.py --video bad_apple.mp4 --images source_images --out outputs/bad_apple_epstein_evidence_mosaic.mp4 --grid 64x36
```

The default render cap is 220 seconds, matching 3:40. Use `--full-video` if your input should be rendered to its actual end.

Rendered frame images go to:

```text
frames/
```

Final video goes to:

```text
outputs/
```

If `bad_apple.mp4` has audio, ffmpeg maps it into the final video. If no audio stream exists, the output is silent.

## Optional Official Downloader

The downloader is opt-in and intentionally limited. It only scans official allowed hosts for direct image/archive links and does not bypass age gates, JavaScript search interfaces, captchas, or access controls.

```powershell
python build_mosaic.py --download-official --dry-run
```

You can add official pages:

```powershell
python build_mosaic.py --download-official --official-url "https://www.justice.gov/epstein" --dry-run
```

Manual placement in `source_images/` is the recommended path when the official release interface is dynamic or protected.

## Cache And Resume Behavior

- `cache/tile_index.json` records accepted and rejected source image signatures.
- `cache/tiles/` stores square, EXIF-stripped thumbnails.
- `cache/extracted/` stores extracted archive contents.
- Existing rendered frames are skipped unless `--force-frames` is set.
- Cached tiles are reused unless `--force-tiles` is set.

Useful rerender commands:

```powershell
python build_mosaic.py --force-frames
python build_mosaic.py --force-tiles --dry-run
```

## Tuning

- `--grid 48x27`: readable silhouette, larger visible image tiles.
- `--grid 64x36`: more pixel-like detail, smaller tiles.
- `--tone-strength 0.34`: higher values make the Bad Apple silhouette more readable, lower values preserve more raw photo texture.
- `--recent-window 180`: raises or lowers tile reuse avoidance.
- `--no-zoom`: disables quick detail-review zoom transitions.
- `--fps 24`: overrides source fps for faster/lighter renders.
- `--frame-format png`: larger lossless frame cache.

## Expected Folder Layout

```text
evidence_mosaic_pipeline/
  build_mosaic.py
  README.md
  SOURCES.md
  requirements.txt
  bad_apple.mp4
  source_images/
  cache/
  frames/
  outputs/
```
