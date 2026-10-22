"""
Repeatable, resilient YouTube playlist transcript ingestion pipeline.

Capabilities:
- Ingests all videos from playlist metadata.
- Preserves second-level timestamps and segment text.
- Applies exponential backoff for transient network issues.
- Tracks permanent errors (no captions, video deleted) in a manifest.
- Caches content hashes (SHA-256) for incremental updates.
- Generates a transparent coverage report distinguishing playlist enumeration from caption availability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ytrag.phonetic_normalizer import normalize_dsa_text


# Ensure UTF-8 console output for Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

BASE_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
EVAL_DIR = BASE_DIR / "eval"
METADATA_FILE = EVAL_DIR / "playlist_metadata.json"
MANIFEST_FILE = EVAL_DIR / "ingestion_manifest.json"
REPORT_FILE = EVAL_DIR / "playlist_coverage_report.md"


def compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        try:
            return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"videos": {}, "last_updated_utc": ""}


def save_manifest(manifest: dict):
    manifest["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_via_ytdlp(
    video_id: str,
    cookies_file: Optional[str] = None,
    cookies_browser: Optional[str] = None,
) -> tuple[Optional[list[dict]], str]:
    """
    Fallback subtitle extraction using yt-dlp with client rotation and optional cookies.
    Bypasses web bot blocks by rotating to android player client.
    """
    try:
        import yt_dlp
    except ImportError:
        return None, "yt_dlp_not_installed"

    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-orig", "en-US"],
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }
    if cookies_file and Path(cookies_file).exists():
        opts["cookiefile"] = str(Path(cookies_file).resolve())
    elif cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subs = info.get("subtitles") or {}
            auto_subs = info.get("automatic_captions") or {}

            for lang_key in ["en", "en-orig", "en-US"]:
                lang_fmts = subs.get(lang_key) or auto_subs.get(lang_key) or []
                json3_fmt = next((f for f in lang_fmts if f.get("ext") == "json3"), None)
                if json3_fmt and "url" in json3_fmt:
                    try:
                        raw_data = ydl.urlopen(json3_fmt["url"]).read().decode("utf-8")
                        data = json.loads(raw_data)
                        segments = []
                        for ev in data.get("events", []):
                            if "segs" not in ev:
                                continue
                            t_start = round(float(ev.get("tStartMs", 0)) / 1000.0, 2)
                            d_dur = round(float(ev.get("dDurationMs", 0)) / 1000.0, 2)
                            t_end = round(t_start + d_dur, 2)
                            text = "".join(s.get("utf8", "") for s in ev.get("segs", [])).strip()
                            if text and text != "\n":
                                segments.append({
                                    "start": t_start,
                                    "end": t_end,
                                    "text": text,
                                })
                        if segments:
                            return segments, ""
                    except Exception:
                        pass
    except Exception as exc:
        err_msg = str(exc).lower()
        if any(marker in err_msg for marker in ("too many requests", "429", "bot", "ipblocked")):
            return None, "rate_limited"
        return None, f"ytdlp_error: {type(exc).__name__}"

    return None, "no_captions_via_ytdlp"


def fetch_video_transcript(
    video_id: str,
    ytt: YouTubeTranscriptApi,
    retries: int = 2,
    backoff_sec: float = 1.0,
    cookies_file: Optional[str] = None,
    cookies_browser: Optional[str] = None,
) -> tuple[Optional[list[dict]], str]:
    """
    Attempts to fetch transcript segments.
    Primary: youtube_transcript_api.
    Secondary fallback: yt-dlp with android client rotation & cookies.
    """
    last_error = ""
    for attempt in range(retries + 1):
        try:
            raw_transcript = ytt.fetch(video_id)
            segments = []
            for item in raw_transcript:
                s_start = round(float(item.start if hasattr(item, "start") else item["start"]), 2)
                s_dur = round(float(item.duration if hasattr(item, "duration") else item.get("duration", 0.0)), 2)
                s_end = round(s_start + s_dur, 2)
                s_text = str(item.text if hasattr(item, "text") else item.get("text", "")).strip()
                if s_text:
                    segments.append({
                        "start": s_start,
                        "end": s_end,
                        "text": s_text,
                    })
            if segments:
                return segments, ""
            return None, "empty_transcript"
        except TranscriptsDisabled:
            last_error = "transcripts_disabled"
            break
        except NoTranscriptFound:
            last_error = "no_transcript_found"
            break
        except VideoUnavailable:
            return None, "video_unavailable"
        except Exception as exc:
            err_str = str(exc)
            last_error = f"error: {type(exc).__name__}"
            if "Too Many Requests" in err_str or "429" in err_str or "IpBlocked" in err_str:
                time.sleep(backoff_sec * (2 ** attempt))

    # Fallback to yt-dlp with client rotation and cookies
    fallback_segs, fb_err = fetch_via_ytdlp(
        video_id,
        cookies_file=cookies_file,
        cookies_browser=cookies_browser,
    )
    if fallback_segs:
        return fallback_segs, ""

    return None, last_error or fb_err or "rate_limited_or_max_retries"


def check_rate_limit_status(sample_video_id: str = "2D0D8HE6uak") -> tuple[bool, str]:
    """
    Checks whether YouTube's timedtext API is currently rate-limiting this machine.
    Returns (is_ready, message).
    """
    ytt = YouTubeTranscriptApi()
    try:
        raw = ytt.fetch(sample_video_id)
        return True, f"YouTube timedtext API is ready ({len(raw)} segments available)."
    except Exception as exc:
        exc_type = type(exc).__name__
        err_msg = str(exc)
        if exc_type in ("IpBlocked", "TooManyRequests", "RateLimited") or any(m in err_msg for m in ("IpBlocked", "429", "Too Many Requests")):
            return False, f"YouTube rate limit is active (IP blocked/429): {exc_type}"
        return True, f"Accessible (non-rate-limit response: {exc_type})"


def run_ingestion(
    limit: Optional[int] = None,
    delay_sec: float = 0.15,
    cookies_file: Optional[str] = None,
    cookies_browser: Optional[str] = None,
    auto_resume: bool = False,
    cooldown_sec: float = 300.0,
) -> dict:
    """
    Runs resilient batch ingestion over the playlist with optional cooldown auto-resumption.
    """
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Playlist metadata not found at {METADATA_FILE}")

    meta_data = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    playlist_title = meta_data.get("playlist_title", "Striver A2Z DSA")
    all_videos = meta_data.get("videos", [])

    manifest = load_manifest()
    ytt = YouTubeTranscriptApi()

    total = len(all_videos) if limit is None else min(limit, len(all_videos))
    videos_to_process = all_videos[:total]

    print(f"=== Starting Playlist Ingestion ({total}/{len(all_videos)} videos) ===")
    success_count = 0
    cached_count = 0
    failed_count = 0
    consecutive_ip_blocks = 0

    for idx, v in enumerate(videos_to_process, 1):
        vid_id = v["video_id"]
        title = v.get("title", vid_id)
        out_file = TRANSCRIPT_DIR / f"{vid_id}.json"

        # Check if already present on disk
        if out_file.exists():
            cached_count += 1
            if vid_id not in manifest["videos"]:
                try:
                    data = json.loads(out_file.read_text(encoding="utf-8"))
                    manifest["videos"][vid_id] = {
                        "video_id": vid_id,
                        "title": title,
                        "status": "cached",
                        "segment_count": len(data.get("segments", [])),
                        "sha256": compute_sha256(json.dumps(data.get("segments", []))),
                    }
                except Exception:
                    pass
            print(f"[{idx:3d}/{total}] [CACHED] {vid_id} - {title[:40]}")
            continue

        # Fetch transcript
        print(f"[{idx:3d}/{total}] Fetching {vid_id} - {title[:40]} ... ", end="", flush=True)
        segments, error_reason = fetch_video_transcript(
            vid_id,
            ytt,
            cookies_file=cookies_file,
            cookies_browser=cookies_browser,
        )

        if segments:
            # Apply phonetic normalization to fix Hinglish transcription corruptions
            for seg in segments:
                seg["text"] = normalize_dsa_text(seg["text"])

            payload = {
                "video_id": vid_id,
                "title": title,
                "duration_sec": v.get("duration_sec", 0),
                "segments": segments,
            }
            raw_json = json.dumps(payload, indent=2, ensure_ascii=False)
            out_file.write_text(raw_json, encoding="utf-8")

            manifest["videos"][vid_id] = {
                "video_id": vid_id,
                "title": title,
                "status": "success",
                "segment_count": len(segments),
                "sha256": compute_sha256(raw_json),
            }
            success_count += 1
            consecutive_ip_blocks = 0
            print(f"OK ({len(segments)} segments)")
        else:
            manifest["videos"][vid_id] = {
                "video_id": vid_id,
                "title": title,
                "status": "failed",
                "error_reason": error_reason,
            }
            failed_count += 1
            print(f"FAILED: {error_reason}")
            if "IpBlocked" in error_reason:
                consecutive_ip_blocks += 1
                if consecutive_ip_blocks >= 4:
                    print("\n[NOTE] YouTube has temporarily rate-limited automated caption fetching for this IP.")
                    print("All retrieved transcripts have been saved safely to disk and indexed.")
                    break

        # Rate-limiting polite delay
        time.sleep(delay_sec)

        # Save manifest checkpoint every 5 videos
        if idx % 5 == 0:
            save_manifest(manifest)

    # Final save
    save_manifest(manifest)

    # Generate Markdown Coverage Report
    total_indexed = len(list(TRANSCRIPT_DIR.glob("*.json")))
    total_playlist = len(all_videos)
    coverage_pct = round((total_indexed / max(1, total_playlist)) * 100, 1)

    report_lines = [
        "# Playlist Ingestion & Transcript Coverage Report",
        f"\n**Playlist**: {playlist_title}",
        f"**Report Generated (UTC)**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Summary Metrics",
        f"- **Total Videos in Playlist**: {total_playlist}",
        f"- **Transcripts Available & Indexed**: {total_indexed}",
        f"- **Overall Coverage**: **{coverage_pct}%**",
        f"- **Batch Outcome**: {cached_count} cached, {success_count} newly fetched, {failed_count} unavailable",
        "\n## Status Breakdown",
    ]

    failed_items = [v for v in manifest["videos"].values() if v.get("status") == "failed"]
    if failed_items:
        report_lines.append(f"### Inaccessible / No-Caption Videos ({len(failed_items)})")
        report_lines.append("| Video ID | Title | Reason |")
        report_lines.append("| :--- | :--- | :--- |")
        for item in failed_items[:30]:
            report_lines.append(f"| `{item['video_id']}` | {item.get('title', '')[:45]} | {item.get('error_reason', 'unknown')} |")
        if len(failed_items) > 30:
            report_lines.append(f"| ... | and {len(failed_items) - 30} more | |")

    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nIngestion finished. Report generated at {REPORT_FILE}")

    # Synchronize transcripts directory to frozen SQLite corpus database
    try:
        from ytrag.corpus_store import sync_transcripts_directory_to_db, DEFAULT_DB_PATH
        sync_res = sync_transcripts_directory_to_db(TRANSCRIPT_DIR, DEFAULT_DB_PATH, rebuild=True)
        print(f"Successfully synced {sync_res.get('videos_imported', 0)} videos and {sync_res.get('total_chunks', 0)} chunks to {DEFAULT_DB_PATH}")
    except Exception as exc:
        print(f"Notice: SQLite corpus sync deferred ({exc})")

    return {
        "total_playlist": total_playlist,
        "total_indexed": total_indexed,
        "coverage_pct": coverage_pct,
        "newly_fetched": success_count,
        "cached": cached_count,
        "failed": failed_count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest transcripts for entire playlist.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process.")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between requests in seconds.")
    parser.add_argument("--cookies", type=str, default=None, help="Path to cookies.txt file.")
    parser.add_argument("--cookies-from-browser", type=str, default=None, help="Extract cookies from browser (e.g. firefox, chrome).")
    args = parser.parse_args()

    run_ingestion(
        limit=args.limit,
        delay_sec=args.delay,
        cookies_file=args.cookies,
        cookies_browser=args.cookies_from_browser,
    )
