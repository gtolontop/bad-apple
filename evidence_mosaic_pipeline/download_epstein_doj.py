#!/usr/bin/env python3
"""
Download official DOJ Epstein Library dataset PDFs and extract image candidates.

This script only crawls justice.gov URLs. It does not bypass age gates,
captchas, JavaScript search UIs, robots, authentication, or access controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


BASE = "https://www.justice.gov"
DISCLOSURES = "https://www.justice.gov/epstein/doj-disclosures"
DATASET_RE = re.compile(r"/epstein/doj-disclosures/data-set-(\d+)-files(?:\?page=\d+)?$")
PDF_RE = re.compile(r"\.pdf(?:$|\?)", re.IGNORECASE)


@dataclass(frozen=True)
class PdfLink:
    dataset: int
    label: str
    url: str


def parse_dataset_selector(value: str) -> set[int]:
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            selected.update(range(lo, hi + 1))
        else:
            selected.add(int(part))
    invalid = [item for item in selected if item < 1 or item > 12]
    if invalid:
        raise argparse.ArgumentTypeError("Datasets must be between 1 and 12.")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official DOJ Epstein Library dataset PDFs.")
    parser.add_argument("--datasets", type=parse_dataset_selector, default=parse_dataset_selector("1-12"))
    parser.add_argument("--download-dir", default="official_downloads/doj_pdfs")
    parser.add_argument("--images-dir", default="source_images/_pdf_images")
    parser.add_argument("--manifest", default="official_downloads/doj_manifest.json")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--download", action="store_true", help="Download PDFs after discovery.")
    parser.add_argument("--extract", action="store_true", help="Extract image candidates from downloaded PDFs.")
    parser.add_argument("--max-files", type=int, default=None, help="Debug/testing cap. Default: no cap.")
    parser.add_argument("--max-bytes", type=int, default=None, help="Stop before exceeding this many newly downloaded bytes.")
    parser.add_argument("--min-image-side", type=int, default=240)
    parser.add_argument("--min-image-pixels", type=int, default=90_000)
    parser.add_argument("--render-photo-pages", action="store_true", help="Also render low-text, image-heavy PDF pages.")
    parser.add_argument("--page-render-dpi", type=int, default=120)
    parser.add_argument("--crawl-pagination", action="store_true", help="Try DOJ ?page=N listing pages. These are often blocked by Akamai.")
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--user-agent", default="official-evidence-mosaic-local-pipeline/1.0")
    parser.add_argument(
        "--accept-age-gate",
        action="store_true",
        help="Set the DOJ age-verification cookie for this download session.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def allowed_justice_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"https", "http"} and parsed.netloc.lower() in {"www.justice.gov", "justice.gov"}


def get_soup(session: requests.Session, url: str, timeout: float) -> BeautifulSoup:
    if not allowed_justice_url(url):
        raise ValueError(f"Refusing non-justice.gov URL: {url}")
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def normalize_page_url(url: str) -> str:
    parsed = urlparse(url)
    query = parsed.query
    if query == "page=0":
        query = ""
    return parsed._replace(fragment="", query=query).geturl()


def discover_dataset_roots(session: requests.Session, datasets: set[int], timeout: float) -> list[str]:
    soup = get_soup(session, DISCLOSURES, timeout)
    roots: dict[int, str] = {}
    for link in soup.find_all("a"):
        href = link.get("href")
        if not href:
            continue
        url = urljoin(DISCLOSURES, href)
        path = urlparse(url).path
        match = re.search(r"/epstein/doj-disclosures/data-set-(\d+)-files$", path)
        if match:
            dataset_id = int(match.group(1))
            if dataset_id in datasets:
                roots[dataset_id] = url
    missing = sorted(datasets - set(roots))
    if missing:
        print(f"Warning: did not find root page for datasets: {missing}")
    return [roots[key] for key in sorted(roots)]


def discover_pdf_links(
    session: requests.Session,
    datasets: set[int],
    timeout: float,
    delay: float,
    crawl_pagination: bool,
) -> list[PdfLink]:
    queue = discover_dataset_roots(session, datasets, timeout)
    seen_pages: set[str] = set()
    links: dict[str, PdfLink] = {}

    while queue:
        page_url = normalize_page_url(queue.pop(0))
        if page_url in seen_pages:
            continue
        seen_pages.add(page_url)
        try:
            soup = get_soup(session, page_url, timeout)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            print(f"Warning: could not read listing page ({status}): {page_url}")
            continue
        print(f"Discovered page {len(seen_pages):04d}: {page_url}")

        current_dataset = None
        match_page = DATASET_RE.search(urlparse(page_url).path + (("?" + urlparse(page_url).query) if urlparse(page_url).query else ""))
        if match_page:
            current_dataset = int(match_page.group(1))

        for tag in soup.find_all("a"):
            href = tag.get("href")
            if not href:
                continue
            url = urljoin(page_url, href)
            if not allowed_justice_url(url):
                continue
            parsed = urlparse(url)
            text = " ".join(tag.get_text(" ", strip=True).split())

            dataset_match = DATASET_RE.search(parsed.path + (("?" + parsed.query) if parsed.query else ""))
            if dataset_match:
                dataset_id = int(dataset_match.group(1))
                normalized = normalize_page_url(url)
                if urlparse(normalized).query.startswith("page=") and not crawl_pagination:
                    continue
                if dataset_id in datasets and normalized not in seen_pages and normalized not in queue:
                    queue.append(normalized)

            if PDF_RE.search(parsed.path):
                dataset_id = current_dataset
                if dataset_id is None:
                    inferred = re.search(r"DataSet%20(\d+)|DataSet[ /_-]*(\d+)", url, flags=re.IGNORECASE)
                    dataset_id = int(next(group for group in inferred.groups() if group)) if inferred else 0
                if dataset_id in datasets:
                    label = text or Path(parsed.path).name
                    links[url] = PdfLink(dataset=dataset_id, label=label, url=url)

        time.sleep(delay)

    return sorted(links.values(), key=lambda item: (item.dataset, item.label, item.url))


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"pdfs": {}, "images": {}, "runs": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"pdfs": {}, "images": {}, "runs": []}


def save_manifest(path: Path, manifest: dict) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def pdf_target_path(download_dir: Path, link: PdfLink) -> Path:
    name = Path(urlparse(link.url).path).name or f"{sha1_text(link.url)}.pdf"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return download_dir / f"dataset_{link.dataset:02d}" / safe_name


def download_one(session: requests.Session, link: PdfLink, target: Path, timeout: float) -> tuple[str, int]:
    ensure_dir(target.parent)
    if target.exists() and target.stat().st_size > 0:
        return "cached", target.stat().st_size

    tmp = target.with_suffix(target.suffix + ".part")
    headers = {}
    existing = tmp.stat().st_size if tmp.exists() else 0
    mode = "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with session.get(link.url, stream=True, timeout=timeout, headers=headers) as response:
        if existing and response.status_code == 200:
            existing = 0
            mode = "wb"
        response.raise_for_status()
        with tmp.open(mode + "") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(target)
    return "downloaded", target.stat().st_size


def download_pdfs(
    session: requests.Session,
    links: list[PdfLink],
    download_dir: Path,
    manifest: dict,
    manifest_path: Path,
    timeout: float,
    delay: float,
    max_bytes: int | None,
) -> None:
    new_bytes = 0
    for idx, link in enumerate(links, start=1):
        target = pdf_target_path(download_dir, link)
        if max_bytes is not None and new_bytes >= max_bytes:
            print(f"Reached --max-bytes ({max_bytes}). Stopping downloads.")
            return
        try:
            status, size = download_one(session, link, target, timeout)
            if status == "downloaded":
                new_bytes += size
            manifest["pdfs"][link.url] = {
                "dataset": link.dataset,
                "label": link.label,
                "url": link.url,
                "path": str(target.resolve()),
                "size": size,
                "status": status,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            print(f"[{idx}/{len(links)}] {status}: dataset {link.dataset} {target.name} ({size:,} bytes)")
        except Exception as exc:
            manifest["pdfs"][link.url] = {
                "dataset": link.dataset,
                "label": link.label,
                "url": link.url,
                "path": str(target.resolve()),
                "status": "error",
                "error": repr(exc),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            print(f"[{idx}/{len(links)}] ERROR {link.url}: {exc}")
        if idx % 20 == 0:
            save_manifest(manifest_path, manifest)
        time.sleep(delay)
    save_manifest(manifest_path, manifest)


def iter_downloaded_pdfs(manifest: dict) -> Iterable[tuple[str, Path, int]]:
    for url, item in sorted(manifest.get("pdfs", {}).items()):
        path = Path(item.get("path", ""))
        if item.get("status") in {"downloaded", "cached"} and path.exists():
            yield url, path, int(item.get("dataset") or 0)


def reencode_image_bytes(data: bytes, target: Path, min_side: int, min_pixels: int) -> bool:
    try:
        from io import BytesIO

        with Image.open(BytesIO(data)) as src:
            image = src.convert("RGB")
            if min(image.size) < min_side or image.width * image.height < min_pixels:
                return False
            ensure_dir(target.parent)
            image.save(target, format="JPEG", quality=92, optimize=True)
            return True
    except (UnidentifiedImageError, OSError):
        return False


def page_is_visual(page, page_area: float) -> bool:
    text = page.get_text("text") or ""
    if len(text.strip()) > 180:
        return False
    drawings = page.get_drawings()
    if len(drawings) > 30:
        return False
    images = page.get_images(full=True)
    if not images:
        return False
    return True


def render_page(page, target: Path, dpi: int) -> bool:
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    ensure_dir(target.parent)
    pix.save(str(target))
    return target.exists() and target.stat().st_size > 0


def extract_images_from_pdfs(
    manifest: dict,
    manifest_path: Path,
    images_dir: Path,
    min_side: int,
    min_pixels: int,
    render_photo_pages: bool,
    page_render_dpi: int,
) -> None:
    if fitz is None:
        raise RuntimeError("PyMuPDF is missing. Run: python -m pip install -r requirements.txt")

    extracted = 0
    skipped_done = 0
    pdfs = list(iter_downloaded_pdfs(manifest))
    for pdf_index, (url, pdf_path, dataset) in enumerate(pdfs, start=1):
        pdf_key = sha1_text(str(pdf_path.resolve()))
        pdf_state = manifest.setdefault("extraction", {}).get(pdf_key, {})
        if pdf_state.get("complete") and pdf_state.get("mtime_ns") == pdf_path.stat().st_mtime_ns:
            skipped_done += 1
            continue

        print(f"[extract {pdf_index}/{len(pdfs)}] {pdf_path.name}")
        count_for_pdf = 0
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            manifest.setdefault("extraction", {})[pdf_key] = {"complete": False, "error": repr(exc)}
            continue

        with doc:
            for page_number in range(len(doc)):
                page = doc[page_number]
                for image_index, img in enumerate(page.get_images(full=True), start=1):
                    xref = img[0]
                    try:
                        info = doc.extract_image(xref)
                    except Exception:
                        continue
                    data = info.get("image")
                    if not data:
                        continue
                    image_key = sha1_text(f"{pdf_key}:{page_number}:{xref}:{len(data)}")
                    target = images_dir / f"dataset_{dataset:02d}" / f"{pdf_path.stem}_p{page_number + 1:04d}_x{xref}_{image_key[:10]}.jpg"
                    if target.exists():
                        continue
                    if reencode_image_bytes(data, target, min_side, min_pixels):
                        manifest.setdefault("images", {})[str(target.resolve())] = {
                            "source_pdf": str(pdf_path.resolve()),
                            "source_url": url,
                            "dataset": dataset,
                            "page": page_number + 1,
                            "kind": "embedded_image",
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        count_for_pdf += 1
                        extracted += 1

                if render_photo_pages and page_is_visual(page, float(page.rect.width * page.rect.height)):
                    image_key = sha1_text(f"{pdf_key}:render:{page_number}")
                    target = images_dir / f"dataset_{dataset:02d}" / f"{pdf_path.stem}_p{page_number + 1:04d}_render_{image_key[:10]}.png"
                    if not target.exists() and render_page(page, target, page_render_dpi):
                        manifest.setdefault("images", {})[str(target.resolve())] = {
                            "source_pdf": str(pdf_path.resolve()),
                            "source_url": url,
                            "dataset": dataset,
                            "page": page_number + 1,
                            "kind": "rendered_visual_page",
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        count_for_pdf += 1
                        extracted += 1

        manifest.setdefault("extraction", {})[pdf_key] = {
            "complete": True,
            "pdf": str(pdf_path.resolve()),
            "mtime_ns": pdf_path.stat().st_mtime_ns,
            "images": count_for_pdf,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if pdf_index % 10 == 0:
            save_manifest(manifest_path, manifest)

    save_manifest(manifest_path, manifest)
    print(f"Extraction complete. New images: {extracted}. PDFs already complete: {skipped_done}.")


def main() -> int:
    args = parse_args()
    if not args.discover_only and not args.download and not args.extract:
        args.download = True
        args.extract = True

    download_dir = Path(args.download_dir)
    images_dir = Path(args.images_dir)
    manifest_path = Path(args.manifest)
    ensure_dir(download_dir)
    ensure_dir(images_dir)
    ensure_dir(manifest_path.parent)

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})
    if args.accept_age_gate:
        session.cookies.set("justiceGovAgeVerified", "true", domain="justice.gov", path="/")
        session.cookies.set("justiceGovAgeVerified", "true", domain="www.justice.gov", path="/")
    manifest = load_manifest(manifest_path)
    manifest.setdefault("runs", []).append(
        {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "datasets": sorted(args.datasets),
            "download": args.download,
            "extract": args.extract,
        }
    )

    links = discover_pdf_links(session, args.datasets, args.timeout, args.request_delay, args.crawl_pagination)
    if args.max_files:
        links = links[: args.max_files]
    manifest["discovered"] = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasets": sorted(args.datasets),
        "pdf_count": len(links),
        "links": [link.__dict__ for link in links],
    }
    save_manifest(manifest_path, manifest)
    print(f"Total official DOJ PDF links discovered: {len(links):,}")

    if args.discover_only:
        return 0

    if args.download:
        download_pdfs(
            session=session,
            links=links,
            download_dir=download_dir,
            manifest=manifest,
            manifest_path=manifest_path,
            timeout=args.timeout,
            delay=args.request_delay,
            max_bytes=args.max_bytes,
        )

    if args.extract:
        extract_images_from_pdfs(
            manifest=manifest,
            manifest_path=manifest_path,
            images_dir=images_dir,
            min_side=args.min_image_side,
            min_pixels=args.min_image_pixels,
            render_photo_pages=args.render_photo_pages,
            page_render_dpi=args.page_render_dpi,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
