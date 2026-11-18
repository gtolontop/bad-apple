#!/usr/bin/env python3
"""
Build a Bad Apple-style evidence-photo photomosaic video.

Default usage:
    python build_mosaic.py --video bad_apple.mp4 --images source_images --out outputs/bad_apple_epstein_evidence_mosaic.mp4 --grid 48x27

The downloader is intentionally opt-in. The main ingestion path is a local
source_images/ directory populated with files or zips from official releases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

try:
    import cv2
except Exception:  # pragma: no cover - reported by require_runtime_dependencies
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover - reported by require_runtime_dependencies
    np = None

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except Exception:  # pragma: no cover - reported by require_runtime_dependencies
    Image = None
    ImageOps = None
    UnidentifiedImageError = OSError

try:
    import requests
except Exception:  # pragma: no cover - optional downloader only
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional downloader only
    BeautifulSoup = None

try:
    import pytesseract
except Exception:  # pragma: no cover - OCR is optional unless required
    pytesseract = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ARCHIVE_EXTS = {".zip"}
ALLOWED_DOWNLOAD_HOSTS = {
    "justice.gov",
    "www.justice.gov",
    "oversight.house.gov",
    "www.oversight.house.gov",
}
DEFAULT_OFFICIAL_URLS = [
    "https://www.justice.gov/epstein",
    "https://www.justice.gov/opa/pr/department-justice-publishes-35-million-responsive-pages-compliance-epstein-files",
    "https://oversight.house.gov/",
]
PII_PATTERNS = [
    r"\bpassport\b",
    r"\bdriver'?s?\s+licen[cs]e\b",
    r"\bsocial\s+security\b",
    r"\bssn\b",
    r"\bdate\s+of\s+birth\b",
    r"\bdob\b",
    r"\baddress\b",
    r"\b(street|st\.|avenue|ave\.|road|rd\.|boulevard|blvd\.|drive|dr\.)\b",
    r"\b\d{3}[-.\s]\d{2}[-.\s]\d{4}\b",
    r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
]


@dataclass(frozen=True)
class TileRecord:
    source: str
    tile_path: str
    luminance: float
    contrast: float
    bucket: int
    width: int
    height: int


def parse_grid(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", value.lower().strip())
    if not match:
        raise argparse.ArgumentTypeError("Grid must look like 48x27 or 64x36.")
    grid_w, grid_h = int(match.group(1)), int(match.group(2))
    if grid_w < 2 or grid_h < 2:
        raise argparse.ArgumentTypeError("Grid dimensions are too small.")
    return grid_w, grid_h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Bad Apple photomosaic from local official-source image material."
    )
    parser.add_argument("--video", default="bad_apple.mp4", help="Input Bad Apple video.")
    parser.add_argument("--images", default="source_images", help="Folder containing official images or .zip archives.")
    parser.add_argument("--out", default="outputs/bad_apple_epstein_evidence_mosaic.mp4", help="Output mp4 path.")
    parser.add_argument("--grid", default="48x27", type=parse_grid, help="Mosaic grid, for example 48x27 or 64x36.")
    parser.add_argument("--resolution", default="1920x1080", type=parse_grid, help="Output resolution. Default: 1920x1080.")
    parser.add_argument("--cache", default="cache", help="Cache directory.")
    parser.add_argument("--frames", default="frames", help="Rendered frame directory.")
    parser.add_argument("--buckets", type=int, default=16, help="Brightness bucket count.")
    parser.add_argument("--dry-run", action="store_true", help="Only ingest images and report usable bucket counts.")
    parser.add_argument("--force-tiles", action="store_true", help="Rebuild cached thumbnails.")
    parser.add_argument("--force-frames", action="store_true", help="Re-render existing frame images.")
    parser.add_argument("--fps", type=float, default=None, help="Override render fps. Default: source fps.")
    parser.add_argument("--duration", type=float, default=220.0, help="Max seconds to render. Default: 220 seconds.")
    parser.add_argument("--full-video", action="store_true", help="Render the full input video instead of --duration.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--tile-size", type=int, default=None, help="Cached square thumbnail size. Default: derived from grid.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for tile selection.")
    parser.add_argument("--recent-window", type=int, default=180, help="Avoid reusing a tile within this many placements when possible.")
    parser.add_argument("--tone-strength", type=float, default=0.34, help="Blend amount toward target cell brightness.")
    parser.add_argument("--fast-tone-bins", type=int, default=32, help="Precompute this many tone variants per tile for faster rendering.")
    parser.add_argument("--no-zoom", action="store_true", help="Disable occasional zoom-in transitions.")
    parser.add_argument("--zoom-period", type=float, default=22.0, help="Seconds between zoom transitions.")
    parser.add_argument("--zoom-duration", type=float, default=1.15, help="Seconds per zoom transition.")
    parser.add_argument("--zoom-max", type=float, default=2.35, help="Maximum zoom factor.")
    parser.add_argument("--download-official", action="store_true", help="Opt-in best-effort download of direct assets linked from official pages.")
    parser.add_argument("--official-url", action="append", default=[], help="Official .gov page to scan. Can be repeated.")
    parser.add_argument("--enable-ocr", choices=["auto", "always", "never"], default="auto", help="OCR images for text/PII filtering.")
    parser.add_argument("--strict-filter", action="store_true", help="Use stricter visual/text heuristics. Safer, but may reject more images.")
    parser.add_argument("--allow-text-tiles", action="store_true", help="Allow text/UI-heavy tiles, useful for personal screenshot mosaics.")
    parser.add_argument("--photo-only", action="store_true", help="Reject document scans and keep only images that look like real photos.")
    parser.add_argument("--photo-white-limit", type=float, default=0.38, help="White-background limit for --photo-only document rejection.")
    parser.add_argument("--min-images", type=int, default=200, help="Warn if fewer usable images are found.")
    parser.add_argument("--max-source-images", type=int, default=None, help="Limit source image scan for huge folders.")
    parser.add_argument("--frame-format", choices=["jpg", "png"], default="jpg", help="Rendered frame format.")
    parser.add_argument("--crf", type=int, default=18, help="ffmpeg x264 CRF quality.")
    parser.add_argument("--preset", default="slow", help="ffmpeg x264 preset.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable path. Default: ffmpeg on PATH.")
    return parser.parse_args()


def require_runtime_dependencies(args: argparse.Namespace) -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if Image is None or ImageOps is None:
        missing.append("Pillow")
    if args.download_official and (requests is None or BeautifulSoup is None):
        missing.extend(["requests", "beautifulsoup4"])
    if missing:
        unique = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            f"Missing runtime dependencies: {unique}. "
            "Run: python -m pip install -r requirements.txt"
        )


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def rel_to_cwd(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def safe_extract_zip(zip_path: Path, dest_root: Path) -> Path:
    stat = zip_path.stat()
    extract_dir = dest_root / sha1_text(f"{zip_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
    done_marker = extract_dir / ".complete"
    if done_marker.exists():
        return extract_dir

    ensure_dirs(extract_dir)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            member_name = member.filename.replace("\\", "/")
            if Path(member_name).suffix.lower() not in IMAGE_EXTS:
                continue
            target = (extract_dir / member_name).resolve()
            if not str(target).startswith(str(extract_dir.resolve())):
                print(f"Skipping unsafe zip member: {member.filename}")
                continue
            ensure_dirs(target.parent)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    done_marker.write_text("ok\n", encoding="utf-8")
    return extract_dir


def iter_candidate_images(images_dir: Path, cache_dir: Path) -> list[Path]:
    extract_root = cache_dir / "extracted"
    ensure_dirs(extract_root)

    roots = [images_dir]
    for archive in images_dir.rglob("*"):
        if archive.is_file() and archive.suffix.lower() in ARCHIVE_EXTS:
            try:
                roots.append(safe_extract_zip(archive, extract_root))
            except Exception as exc:
                print(f"Could not extract {archive}: {exc}")

    candidates: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                key = str(path.resolve()).lower()
                if key not in seen:
                    candidates.append(path)
                    seen.add(key)
    return sorted(candidates, key=lambda p: str(p).lower())


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def image_luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def detect_faces(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return 0
    cascade = cv2.CascadeClassifier(str(cascade_path))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(24, 24))
    return len(faces)


def skin_fraction(rgb: np.ndarray) -> float:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mask = (cr > 133) & (cr < 173) & (cb > 77) & (cb < 127) & (y > 45)
    return float(mask.mean())


def text_like_density(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 80, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
    joined = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 12 <= w <= 260 and 4 <= h <= 36 and w / max(h, 1) >= 2.5:
            text_area += w * h
    return text_area / float(512 * 512)


def grayscale_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    probs = hist[hist > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def border_fraction(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    band = max(8, min(h, w) // 18)
    border = np.concatenate(
        [
            mask[:band, :].ravel(),
            mask[-band:, :].ravel(),
            mask[:, :band].ravel(),
            mask[:, -band:].ravel(),
        ]
    )
    return float(border.mean()) if border.size else 0.0


def photo_only_reject_reason(image: Image.Image, strict: bool, white_limit: float) -> Optional[str]:
    preview = image.copy()
    preview.thumbnail((768, 768), Image.Resampling.LANCZOS)
    rgb = np.asarray(preview.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]

    white_mask = (gray > 232) & (sat < 38)
    very_white_mask = (gray > 246) & (sat < 28)
    ink_mask = (gray < 185) & (sat < 95)
    white_fraction = float(white_mask.mean())
    very_white_fraction = float(very_white_mask.mean())
    ink_fraction = float(ink_mask.mean())
    low_sat_fraction = float((sat < 34).mean())
    mean_sat = float(sat.mean())
    entropy = grayscale_entropy(gray)
    text_density = text_like_density(rgb)
    white_border = border_fraction(white_mask)

    # Scanned PDFs tend to be white, low-saturation sheets with sparse gray ink.
    if white_fraction > white_limit and low_sat_fraction > 0.62 and ink_fraction > 0.0025:
        return "document_scan_white_ink"
    if very_white_fraction > 0.25 and white_border > 0.54 and text_density > 0.002:
        return "document_scan_border"
    if white_fraction > 0.52 and mean_sat < 28 and entropy < 5.2:
        return "document_scan_low_texture"
    if strict and white_fraction > 0.30 and text_density > 0.006:
        return "document_scan_text"

    # Blank/near-blank extracted PDF image objects are not useful photo tiles.
    if entropy < 2.2 and (white_fraction > 0.70 or float((gray < 12).mean()) > 0.70):
        return "near_blank"

    return None


def run_ocr_if_available(image: Image.Image, mode: str) -> str:
    if mode == "never":
        return ""
    if pytesseract is None:
        if mode == "always":
            raise RuntimeError(
                "--enable-ocr always was requested, but pytesseract/Tesseract is not available. "
                "Install pytesseract plus the Tesseract binary, or use --enable-ocr auto/never."
            )
        return ""
    try:
        small = image.copy()
        small.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
        return pytesseract.image_to_string(small, config="--psm 6") or ""
    except Exception as exc:
        if mode == "always":
            raise RuntimeError(f"OCR failed: {exc}") from exc
        return ""


def has_pii_text(text: str) -> bool:
    if not text.strip():
        return False
    normalized = " ".join(text.upper().split())
    for pattern in PII_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return True
    if len(re.findall(r"\b\d{1,5}\s+[A-Z][A-Z0-9.'-]+\s+(STREET|ST|AVENUE|AVE|ROAD|RD|DRIVE|DR)\b", normalized)) > 0:
        return True
    return False


def reject_reason(
    image: Image.Image,
    strict: bool,
    ocr_mode: str,
    allow_text_tiles: bool,
    photo_only: bool,
    photo_white_limit: float,
) -> Optional[str]:
    preview = image.copy()
    preview.thumbnail((768, 768), Image.Resampling.LANCZOS)
    rgb = np.asarray(preview.convert("RGB"))

    if detect_faces(rgb) > 0:
        return "face_detected"

    skin_limit = 0.18 if strict else 0.28
    if skin_fraction(rgb) > skin_limit:
        return "skin_tone_fraction"

    if photo_only:
        reason = photo_only_reject_reason(image, strict, photo_white_limit)
        if reason:
            return reason

    density = text_like_density(rgb)
    text_limit = 0.030 if strict else 0.055
    if density > text_limit and not allow_text_tiles:
        return "text_like_density"

    text = "" if allow_text_tiles else run_ocr_if_available(image, ocr_mode)
    if text and has_pii_text(text):
        return "pii_text"

    if strict and len(text.strip()) > 120:
        return "readable_text"

    return None


def source_signature(path: Path) -> str:
    stat = path.stat()
    return sha1_text(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")


def load_json(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def atomic_write_json(path: Path, data) -> None:
    ensure_dirs(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def process_images(args: argparse.Namespace) -> list[TileRecord]:
    images_dir = Path(args.images)
    cache_dir = Path(args.cache)
    tile_dir = cache_dir / "tiles"
    reject_dir = cache_dir / "rejected"
    ensure_dirs(images_dir, cache_dir, tile_dir, reject_dir)

    if args.download_official:
        download_official_assets(images_dir, args.official_url or DEFAULT_OFFICIAL_URLS)

    candidates = iter_candidate_images(images_dir, cache_dir)
    if args.max_source_images is not None:
        candidates = candidates[: max(0, args.max_source_images)]
    index_path = cache_dir / "tile_index.json"
    old_index = load_json(index_path, {})
    records_by_sig = old_index.get("records", {})
    rejects_by_sig = old_index.get("rejects", {})
    records: list[TileRecord] = []
    reject_counts: Counter[str] = Counter()

    tile_size = args.tile_size
    if tile_size is None:
        out_w, out_h = args.resolution
        grid_w, grid_h = args.grid
        tile_size = max(math.ceil(out_w / grid_w), math.ceil(out_h / grid_h))

    for idx, path in enumerate(candidates, start=1):
        try:
            sig = source_signature(path)
        except OSError:
            continue

        cached = records_by_sig.get(sig)
        if cached and not args.force_tiles and Path(cached["tile_path"]).exists():
            if sig not in rejects_by_sig:
                records.append(TileRecord(**cached))
                continue

        cached_reject = rejects_by_sig.get(sig)
        if cached_reject and not args.force_tiles:
            reject_counts[cached_reject] += 1
            continue

        try:
            with Image.open(path) as src:
                image = ImageOps.exif_transpose(src).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            rejects_by_sig[sig] = "unreadable"
            records_by_sig.pop(sig, None)
            reject_counts["unreadable"] += 1
            print(f"[{idx}/{len(candidates)}] Skipping unreadable image {path}: {exc}")
            continue

        try:
            reason = reject_reason(
                image,
                strict=args.strict_filter,
                ocr_mode=args.enable_ocr,
                allow_text_tiles=args.allow_text_tiles,
                photo_only=args.photo_only,
                photo_white_limit=args.photo_white_limit,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            reason = f"filter_error:{type(exc).__name__}"

        if reason:
            rejects_by_sig[sig] = reason
            records_by_sig.pop(sig, None)
            reject_counts[reason] += 1
            continue

        square = center_crop_square(image)
        thumb = square.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        rgb = np.asarray(thumb, dtype=np.float32)
        lum = image_luminance(rgb)
        avg_lum = float(np.mean(lum))
        contrast = float(np.std(lum))
        bucket = int(np.clip((avg_lum / 256.0) * args.buckets, 0, args.buckets - 1))

        tile_name = f"{sig}.jpg"
        tile_path = tile_dir / tile_name
        thumb.save(tile_path, format="JPEG", quality=92, optimize=True)

        record = TileRecord(
            source=str(path.resolve()),
            tile_path=str(tile_path.resolve()),
            luminance=avg_lum,
            contrast=contrast,
            bucket=bucket,
            width=image.width,
            height=image.height,
        )
        records_by_sig[sig] = record.__dict__
        rejects_by_sig.pop(sig, None)
        records.append(record)

        if idx % 250 == 0:
            print(f"Processed {idx}/{len(candidates)} image candidates...")

    atomic_write_json(
        index_path,
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tile_size": tile_size,
            "bucket_count": args.buckets,
            "records": records_by_sig,
            "rejects": rejects_by_sig,
        },
    )

    print(f"Found {len(candidates)} image candidates.")
    print(f"Usable cached tiles: {len(records)}")
    if reject_counts:
        print("Rejected this run/cache:")
        for reason, count in reject_counts.most_common():
            print(f"  {reason}: {count}")
    return records


def print_bucket_report(records: list[TileRecord], bucket_count: int) -> None:
    counts = Counter(record.bucket for record in records)
    print("\nBrightness buckets:")
    for bucket in range(bucket_count):
        lo = round(bucket * 255 / bucket_count)
        hi = round(((bucket + 1) * 255 / bucket_count) - 1)
        print(f"  {bucket:02d} ({lo:03d}-{hi:03d}): {counts[bucket]:5d}")


def official_host_allowed(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return parsed.scheme in {"https", "http"} and host in ALLOWED_DOWNLOAD_HOSTS


def download_official_assets(images_dir: Path, urls: list[str]) -> None:
    if requests is None or BeautifulSoup is None:
        raise RuntimeError("Official downloader requires requests and beautifulsoup4 from requirements.txt.")

    download_dir = images_dir / "_official_downloads"
    ensure_dirs(download_dir)
    session = requests.Session()
    session.headers.update({"User-Agent": "official-evidence-mosaic-local-pipeline/1.0"})

    for page_url in urls:
        if not official_host_allowed(page_url):
            print(f"Skipping non-allowed official URL: {page_url}")
            continue
        print(f"Scanning official page: {page_url}")
        response = session.get(page_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links: set[str] = set()
        for tag in soup.find_all(["a", "img", "source"]):
            raw = tag.get("href") or tag.get("src")
            if not raw:
                continue
            linked = urljoin(page_url, raw)
            suffix = Path(urlparse(linked).path).suffix.lower()
            if suffix in IMAGE_EXTS or suffix in ARCHIVE_EXTS:
                if official_host_allowed(linked):
                    links.add(linked)

        print(f"  Found {len(links)} direct official image/archive links.")
        for linked in sorted(links):
            parsed = urlparse(linked)
            name = Path(parsed.path).name or sha1_text(linked)
            target = download_dir / name
            if target.exists() and target.stat().st_size > 0:
                continue
            print(f"  Downloading {linked}")
            with session.get(linked, stream=True, timeout=60) as asset_response:
                asset_response.raise_for_status()
                tmp = target.with_suffix(target.suffix + ".tmp")
                with tmp.open("wb") as fh:
                    for chunk in asset_response.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            fh.write(chunk)
                tmp.replace(target)


def load_tile_arrays(records: list[TileRecord], cell_w: int, cell_h: int) -> dict[int, list[dict]]:
    buckets: dict[int, list[dict]] = {}
    for i, record in enumerate(records):
        image = cv2.imread(record.tile_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[1] != cell_w or image.shape[0] != cell_h:
            image = cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        buckets.setdefault(record.bucket, []).append(
            {"id": i, "rgb": image.astype(np.float32), "luminance": record.luminance, "bucket": record.bucket}
        )
    return buckets


def precompute_tone_variants(bucket_map: dict[int, list[dict]], tone_bins: int, tone_strength: float) -> None:
    tone_bins = max(2, tone_bins)
    for tiles in bucket_map.values():
        for tile in tiles:
            variants = []
            current = float(tile["luminance"]) + 1e-6
            rgb = tile["rgb"]
            for tone_idx in range(tone_bins):
                target_lum = 255.0 * tone_idx / (tone_bins - 1)
                scale = np.clip(target_lum / current, 0.38, 2.85)
                adjusted = np.clip(rgb * scale, 0, 255)
                toned = adjusted * (1.0 - tone_strength) + target_lum * tone_strength
                variants.append(np.clip(toned, 0, 255).astype(np.uint8))
            tile["variants"] = variants


def nearest_bucket_candidates(bucket_map: dict[int, list[dict]], target_bucket: int, bucket_count: int) -> list[dict]:
    for radius in range(bucket_count):
        candidates: list[dict] = []
        for bucket in {target_bucket - radius, target_bucket + radius}:
            if 0 <= bucket < bucket_count:
                candidates.extend(bucket_map.get(bucket, []))
        if candidates:
            return candidates
    return []


def choose_tile(
    bucket_map: dict[int, list[dict]],
    target_lum: float,
    bucket_count: int,
    rng: random.Random,
    avoid: set[int],
    recent: deque[int],
    recent_set: set[int],
) -> dict:
    target_bucket = int(np.clip((target_lum / 256.0) * bucket_count, 0, bucket_count - 1))
    candidates = nearest_bucket_candidates(bucket_map, target_bucket, bucket_count)
    if not candidates:
        raise RuntimeError("No tile candidates are available.")

    if len(candidates) <= 10:
        for tile in candidates:
            if tile["id"] not in avoid and tile["id"] not in recent_set:
                return tile
        for tile in candidates:
            if tile["id"] not in avoid:
                return tile
        return rng.choice(candidates)

    for _ in range(12):
        tile = candidates[rng.randrange(len(candidates))]
        if tile["id"] not in avoid and tile["id"] not in recent_set:
            return tile
    for _ in range(8):
        tile = candidates[rng.randrange(len(candidates))]
        if tile["id"] not in avoid:
            return tile
    return rng.choice(candidates)


def mark_recent(tile_id: int, recent: deque[int], recent_set: set[int], limit: int) -> None:
    recent.append(tile_id)
    recent_set.add(tile_id)
    while len(recent) > limit:
        old = recent.popleft()
        if old not in recent:
            recent_set.discard(old)


def match_tile_luminance(tile_rgb: np.ndarray, target_lum: float, tone_strength: float) -> np.ndarray:
    current = float(np.mean(image_luminance(tile_rgb))) + 1e-6
    scale = np.clip(target_lum / current, 0.38, 2.85)
    adjusted = np.clip(tile_rgb * scale, 0, 255)
    toned = adjusted * (1.0 - tone_strength) + target_lum * tone_strength
    return np.clip(toned, 0, 255).astype(np.uint8)


def get_fast_toned_tile(tile: dict, target_lum: float, tone_bins: int) -> np.ndarray:
    variants = tile.get("variants")
    if not variants:
        return match_tile_luminance(tile["rgb"], target_lum, 0.34)
    idx = int(np.clip(round((target_lum / 255.0) * (tone_bins - 1)), 0, tone_bins - 1))
    return variants[idx]


def zoom_factor_for_frame(frame_index: int, fps: float, args: argparse.Namespace) -> float:
    if args.no_zoom or args.zoom_period <= 0 or args.zoom_duration <= 0:
        return 1.0
    t = frame_index / fps
    phase = t % args.zoom_period
    if phase > args.zoom_duration:
        return 1.0
    x = phase / args.zoom_duration
    pulse = math.sin(math.pi * x)
    return 1.0 + (args.zoom_max - 1.0) * pulse


def apply_zoom(rgb: np.ndarray, factor: float, frame_index: int) -> np.ndarray:
    if factor <= 1.001:
        return rgb
    h, w = rgb.shape[:2]
    crop_w = max(1, int(w / factor))
    crop_h = max(1, int(h / factor))
    centers = [
        (0.50, 0.50),
        (0.35, 0.42),
        (0.64, 0.58),
        (0.42, 0.66),
        (0.58, 0.34),
    ]
    cx_frac, cy_frac = centers[(frame_index // 200) % len(centers)]
    cx = int(w * cx_frac)
    cy = int(h * cy_frac)
    x0 = int(np.clip(cx - crop_w // 2, 0, w - crop_w))
    y0 = int(np.clip(cy - crop_h // 2, 0, h - crop_h))
    crop = rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)


def add_top_bar(rgb: np.ndarray, frame_index: int, total_frames: int, dataset_count: int, zoom_factor: float) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    bar_h = 44
    overlay = out.copy()
    overlay[:bar_h, :, :] = (16, 20, 22)
    out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
    cv2.line(out, (0, bar_h), (w, bar_h), (58, 68, 72), 1)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bgr, "OFFICIAL IMAGE REVIEW", (24, 29), font, 0.58, (228, 235, 232), 1, cv2.LINE_AA)
    frame_text = f"FRAME {frame_index + 1:06d}/{total_frames:06d}"
    data_text = f"DATASET {dataset_count:,} USABLE TILES"
    cv2.putText(bgr, frame_text, (w // 2 - 130, 29), font, 0.50, (198, 212, 208), 1, cv2.LINE_AA)
    cv2.putText(bgr, data_text, (w - 360, 29), font, 0.50, (198, 212, 208), 1, cv2.LINE_AA)
    if zoom_factor > 1.001:
        cv2.putText(bgr, "DETAIL REVIEW", (w - 550, 29), font, 0.50, (172, 210, 196), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_frames(args: argparse.Namespace, records: list[TileRecord]) -> tuple[float, int]:
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    out_w, out_h = args.resolution
    grid_w, grid_h = args.grid
    if out_w % grid_w != 0 or out_h % grid_h != 0:
        raise ValueError("Resolution must divide evenly by grid so cells remain clean square-ish tiles.")
    cell_w, cell_h = out_w // grid_w, out_h // grid_h
    if cell_w != cell_h:
        print(f"Warning: cell size is {cell_w}x{cell_h}; 16:9 grids such as 48x27 keep square cells.")

    frame_dir = Path(args.frames)
    ensure_dirs(frame_dir)
    bucket_map = load_tile_arrays(records, cell_w, cell_h)
    if not bucket_map:
        raise RuntimeError("No cached tile images could be loaded.")
    print(f"Precomputing {args.fast_tone_bins} tone variants per tile...")
    precompute_tone_variants(bucket_map, args.fast_tone_bins, args.tone_strength)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps or source_fps
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_duration = source_frames / source_fps if source_frames else 0
    start_frame = int(args.start * source_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    remaining_source_seconds = max(source_duration - args.start, 0) if source_duration else 0
    if args.full_video or not args.duration:
        render_seconds = remaining_source_seconds or 0
    elif remaining_source_seconds:
        render_seconds = min(args.duration, remaining_source_seconds)
    else:
        render_seconds = args.duration
    total_frames = int(round(render_seconds * fps))

    if total_frames <= 0:
        raise RuntimeError("No frames to render. Check --video, --start, and --duration.")

    recent: deque[int] = deque()
    recent_set: set[int] = set()
    prev_cell_ids = np.full((grid_h, grid_w), -1, dtype=np.int64)
    rng = random.Random(args.seed)

    print(f"Rendering {total_frames} frames at {fps:.3f} fps to {frame_dir.resolve()}")
    for out_index in range(total_frames):
        frame_path = frame_dir / f"frame_{out_index:06d}.{args.frame_format}"
        if frame_path.exists() and not args.force_frames:
            continue

        source_pos = start_frame + int(round(out_index * source_fps / fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_pos)
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"Stopped early at rendered frame {out_index}; source video ended.")
            total_frames = out_index
            break

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        mosaic = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        current_cell_ids = np.full((grid_h, grid_w), -1, dtype=np.int64)

        for y in range(grid_h):
            for x in range(grid_w):
                target_lum = float(small[y, x])
                avoid = set()
                if x > 0:
                    avoid.add(int(current_cell_ids[y, x - 1]))
                if y > 0:
                    avoid.add(int(current_cell_ids[y - 1, x]))
                avoid.add(int(prev_cell_ids[y, x]))
                avoid.discard(-1)

                tile = choose_tile(bucket_map, target_lum, args.buckets, rng, avoid, recent, recent_set)
                cell = get_fast_toned_tile(tile, target_lum, args.fast_tone_bins)
                y0, y1 = y * cell_h, (y + 1) * cell_h
                x0, x1 = x * cell_w, (x + 1) * cell_w
                mosaic[y0:y1, x0:x1] = cell
                current_cell_ids[y, x] = tile["id"]
                mark_recent(tile["id"], recent, recent_set, args.recent_window)

        factor = zoom_factor_for_frame(out_index, fps, args)
        mosaic = apply_zoom(mosaic, factor, out_index)
        mosaic = add_top_bar(mosaic, out_index, total_frames, len(records), factor)

        bgr_out = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
        if args.frame_format == "jpg":
            cv2.imwrite(str(frame_path), bgr_out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(str(frame_path), bgr_out, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

        prev_cell_ids = current_cell_ids
        if (out_index + 1) % 100 == 0:
            print(f"  Rendered {out_index + 1}/{total_frames} frames")

    cap.release()
    terminator = frame_dir / f"frame_{total_frames:06d}.{args.frame_format}"
    if terminator.exists():
        terminator.unlink()
    return fps, total_frames


def encode_video(args: argparse.Namespace, fps: float, total_frames: int) -> None:
    if total_frames <= 0:
        raise RuntimeError("No frames were rendered, so there is nothing to encode.")

    frame_dir = Path(args.frames)
    out_path = Path(args.out)
    ensure_dirs(out_path.parent)
    pattern = str((frame_dir / f"frame_%06d.{args.frame_format}").resolve())
    video_path = str(Path(args.video).resolve())

    cmd = [
        args.ffmpeg,
        "-y",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        pattern,
        "-i",
        video_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-shortest",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(out_path.resolve()),
    ]
    print("Encoding final video with ffmpeg...")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    if shutil.which(args.ffmpeg) is None and not Path(args.ffmpeg).exists():
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg on PATH or pass --ffmpeg C:\\path\\to\\ffmpeg.exe."
        )
    subprocess.run(cmd, check=True)
    print(f"Done: {out_path.resolve()}")


def main() -> int:
    args = parse_args()
    require_runtime_dependencies(args)
    if args.buckets < 2:
        raise ValueError("--buckets must be at least 2.")
    if not 0 <= args.tone_strength <= 0.85:
        raise ValueError("--tone-strength should be between 0 and 0.85.")

    records = process_images(args)
    print_bucket_report(records, args.buckets)
    if len(records) < args.min_images:
        print(f"\nWarning: only {len(records)} usable images found; aim for {args.min_images}+ for varied mosaics.")

    if args.dry_run:
        print("\nDry run complete. Add more official images/zips to source_images/ if buckets are sparse.")
        return 0

    if not records:
        raise RuntimeError("No usable tiles were found. Populate source_images/ with official images or zips first.")

    fps, total_frames = render_frames(args, records)
    encode_video(args, fps, total_frames)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
