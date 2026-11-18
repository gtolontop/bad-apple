# Bad Apple experiments

Bad Apple played through things that were never meant to play video.

| Folder | What |
|---|---|
| `Cookie Banners/` | the video rendered as a cookie consent banner |
| `reCAPTCHA/` | the video rendered inside a CAPTCHA grid |
| `Smoke/` | the video rendered as smoke |
| `evidence_mosaic_pipeline/` | Python pipeline that rebuilds each frame as a photomosaic of official public-release images |
| `pc_screenshots_mosaic/` | the same mosaic, fed with desktop screenshots |

## Running the HTML labs

Each lab loads `badapple.mp4` from its own folder. The video is not included: put your own copy of it next to the HTML file, then open the page in a browser.

## Mosaic pipeline

See [`evidence_mosaic_pipeline/README.md`](evidence_mosaic_pipeline/README.md). Source images, frames and rendered videos are generated locally and are not part of the repository.
