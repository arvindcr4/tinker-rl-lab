#!/usr/bin/env python3
import html
import json
import re
import time
from pathlib import Path

import requests


VIDEO_LIST = Path("/tmp/claude-1001/-home-claude-tinker-rl-lab/30780cda-8d14-4b43-8179-c46d6b7ba4dc/scratchpad/berkeley_videos.json")
OUT_ROOT = Path("/home/claude/tinker-rl-lab-minimax/berkeley_courses")
MANIFEST = OUT_ROOT / "TRANSCRIPTS_MANIFEST.md"
FIRECRAWL_URL = "https://api.firecrawl.dev/v2/scrape"
FIRECRAWL_RETRIES = 2
FIRECRAWL_BACKOFF_SECONDS = 10
VIDEO_SLEEP_SECONDS = 3


def load_firecrawl_key() -> str:
    config = json.loads((Path.home() / ".claude.json").read_text())
    key = (
        config.get("mcpServers", {})
        .get("firecrawl", {})
        .get("env", {})
        .get("FIRECRAWL_API_KEY")
    )
    if key and not looks_placeholder(key):
        return key.strip()

    cli_credentials = Path.home() / ".config/firecrawl-cli/credentials.json"
    if cli_credentials.exists():
        cli_config = json.loads(cli_credentials.read_text())
        cli_key = cli_config.get("apiKey")
        if cli_key and not looks_placeholder(cli_key):
            return cli_key.strip()

    if not key:
        raise RuntimeError("Firecrawl API key missing at mcpServers.firecrawl.env.FIRECRAWL_API_KEY")
    raise RuntimeError("Firecrawl API key at mcpServers.firecrawl.env.FIRECRAWL_API_KEY looks like a placeholder")
    return key


def looks_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in ("your_", "placeholder", "<", "todo"))


def firecrawl_scrape_raw_html(url: str, api_key: str, actions: list[dict] | None = None) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"url": url, "formats": ["rawHtml"], "maxAge": 0}
    if actions:
        payload["actions"] = actions
    last_error = None

    for attempt in range(FIRECRAWL_RETRIES + 1):
        try:
            response = requests.post(FIRECRAWL_URL, headers=headers, json=payload, timeout=90)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            body = response.json()
            data = body.get("data") if isinstance(body, dict) else None
            raw_html = data.get("rawHtml") if isinstance(data, dict) else None
            if not raw_html:
                raise RuntimeError(f"missing rawHtml in Firecrawl response: {str(body)[:300]}")
            return raw_html
        except Exception as exc:
            last_error = exc
            if attempt < FIRECRAWL_RETRIES:
                time.sleep(FIRECRAWL_BACKOFF_SECONDS)

    raise RuntimeError(f"Firecrawl scrape failed for {url}: {last_error}")


def find_balanced_json(text: str, start: int) -> str:
    open_at = text.find("{", start)
    if open_at == -1:
        raise ValueError("opening brace not found")

    depth = 0
    in_string = False
    escape = False
    for idx in range(open_at, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_at : idx + 1]

    raise ValueError("balanced JSON object not found")


def extract_player_response(raw_html: str) -> dict:
    markers = ("ytInitialPlayerResponse =", "ytInitialPlayerResponse=", "var ytInitialPlayerResponse =")
    for marker in markers:
        marker_at = raw_html.find(marker)
        if marker_at == -1:
            continue
        try:
            return json.loads(find_balanced_json(raw_html, marker_at))
        except Exception:
            pass

    # Some render paths escape the player response into HTML/JS strings.
    unescaped = html.unescape(raw_html)
    for marker in markers:
        marker_at = unescaped.find(marker)
        if marker_at == -1:
            continue
        try:
            return json.loads(find_balanced_json(unescaped, marker_at))
        except Exception:
            pass

    raise RuntimeError("ytInitialPlayerResponse not found")


def extract_title(raw_html: str, player_response: dict) -> str:
    title = (
        player_response.get("videoDetails", {}).get("title")
        if isinstance(player_response, dict)
        else None
    )
    if title:
        return collapse_space(str(title))

    match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    if match:
        title = html.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))
        title = re.sub(r"\s+-\s+YouTube\s*$", "", title, flags=re.IGNORECASE)
        return collapse_space(title)

    return "untitled"


def caption_tracks(player_response: dict) -> list[dict]:
    return (
        player_response.get("captions", {})
        .get("playerCaptionsTracklistRenderer", {})
        .get("captionTracks", [])
    )


def select_english_track(tracks: list[dict]) -> dict:
    if not tracks:
        raise RuntimeError("no captionTracks found")

    def is_english(track: dict) -> bool:
        language = str(track.get("languageCode", "")).lower()
        vss_id = str(track.get("vssId", "")).lower()
        name = track.get("name", {})
        if isinstance(name, dict):
            name_text = " ".join(run.get("text", "") for run in name.get("runs", []))
            name_text = name_text or name.get("simpleText", "")
        else:
            name_text = str(name)
        name_text = name_text.lower()
        return language == "en" or vss_id.startswith(".en") or "english" in name_text

    exact = [track for track in tracks if str(track.get("languageCode", "")).lower() == "en"]
    if exact:
        return exact[0]

    asr_english = [
        track
        for track in tracks
        if str(track.get("kind", "")).lower() == "asr" and is_english(track)
    ]
    if asr_english:
        return asr_english[0]

    english = [track for track in tracks if is_english(track)]
    if english:
        return english[0]

    raise RuntimeError("English caption track not found")


def with_json3_format(url: str) -> str:
    clean_url = html.unescape(url)
    separator = "&" if "?" in clean_url else "?"
    return f"{clean_url}{separator}fmt=json3"


def fetch_timedtext_json3(url: str, api_key: str) -> dict:
    direct_error = None
    try:
        response = requests.get(url, timeout=45)
        if response.status_code == 200 and response.text.strip():
            return parse_jsonish(response.text)
        direct_error = f"direct GET HTTP {response.status_code}"
    except Exception as exc:
        direct_error = f"direct GET failed: {exc}"

    raw_html = firecrawl_scrape_raw_html(url, api_key)
    try:
        return parse_jsonish(raw_html)
    except Exception as exc:
        raise RuntimeError(f"timedtext parse failed after {direct_error}; Firecrawl parse error: {exc}")


def fetch_glasp_reader_transcript(video_id: str, api_key: str) -> tuple[str, str]:
    url = f"https://glasp.co/reader?url=https://www.youtube.com/watch?v={video_id}"
    raw_html = firecrawl_scrape_raw_html(url, api_key)
    title = extract_glasp_title(raw_html) or "untitled"
    entries = extract_glasp_transcripts(raw_html)
    transcript = transcript_entries_to_text(entries)
    return title, transcript


def fetch_savesubs_transcript(video_id: str, api_key: str) -> tuple[str, str]:
    process_url = f"https://savesubs.com/process?url=https://www.youtube.com/watch?v={video_id}"
    raw_html = firecrawl_scrape_raw_html(
        process_url,
        api_key,
        actions=[{"type": "wait", "milliseconds": 8000}],
    )
    title = extract_title(raw_html, {}) or "untitled"
    vtt_url = extract_savesubs_download_url(raw_html, "vtt")
    response = requests.get(
        vtt_url,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0", "Referer": process_url},
    )
    if response.status_code >= 400 or not response.text.strip():
        raise RuntimeError(f"SaveSubs VTT download failed with HTTP {response.status_code}")
    transcript = vtt_to_transcript(response.text)
    return title, transcript


def extract_savesubs_download_url(raw_html: str, extension: str) -> str:
    for match in re.finditer(r'href="(https://savesubs\.com/save/[^"]+)"', raw_html):
        url = html.unescape(match.group(1))
        if f"ext={extension}" in url:
            return url
    raise RuntimeError(f"SaveSubs {extension.upper()} download link not found")


def vtt_to_transcript(vtt_text: str) -> str:
    entries = []
    current_start = None
    current_lines = []

    for raw_line in vtt_text.splitlines():
        line = raw_line.strip()
        if not line or line == "WEBVTT" or line.startswith(("NOTE", "Kind:", "Language:")):
            continue
        if "-->" in line:
            if current_start is not None and current_lines:
                entries.append({"start": current_start, "text": " ".join(current_lines)})
            start_text = line.split("-->", 1)[0].strip()
            current_start = parse_vtt_timestamp(start_text)
            current_lines = []
            continue
        if current_start is not None and not line.isdigit():
            clean_line = re.sub(r"<[^>]+>", "", html.unescape(line))
            if clean_line:
                current_lines.append(clean_line)

    if current_start is not None and current_lines:
        entries.append({"start": current_start, "text": " ".join(current_lines)})
    if not entries:
        raise RuntimeError("SaveSubs VTT contained no cues")
    return transcript_entries_to_text(entries)


def parse_vtt_timestamp(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(parts[0])


def extract_glasp_title(raw_html: str) -> str:
    marker = '\\"title\\":\\"'
    marker_at = raw_html.find(marker)
    if marker_at == -1:
        match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return collapse_space(html.unescape(re.sub(r"<[^>]+>", " ", match.group(1))))
        return ""

    value_start = marker_at + len(marker)
    value_end = find_escaped_string_end(raw_html, value_start)
    escaped = raw_html[value_start:value_end]
    return collapse_space(json.loads(f'"{escaped}"'))


def extract_glasp_transcripts(raw_html: str) -> list[dict]:
    marker = '\\"transcripts\\":['
    marker_at = raw_html.find(marker)
    if marker_at == -1:
        raise RuntimeError("Glasp reader transcript array not found")

    array_start = raw_html.find("[", marker_at)
    array_end = find_escaped_json_array_end(raw_html, array_start)
    escaped_array = raw_html[array_start : array_end + 1]
    json_array = escaped_array.encode("utf-8").decode("unicode_escape")
    entries = json.loads(json_array)
    if not entries:
        raise RuntimeError("Glasp reader transcript array is empty")
    return entries


def find_escaped_string_end(text: str, start: int) -> int:
    idx = start
    while idx < len(text):
        if text[idx : idx + 2] == '\\"':
            backslashes = 0
            cursor = idx - 1
            while cursor >= start and text[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                return idx
            idx += 2
            continue
        idx += 1
    raise ValueError("escaped string end not found")


def find_escaped_json_array_end(text: str, start: int) -> int:
    depth = 0
    in_string = False
    idx = start
    while idx < len(text):
        if in_string:
            if text[idx : idx + 2] == '\\"':
                backslashes = 0
                cursor = idx - 1
                while cursor >= start and text[cursor] == "\\":
                    backslashes += 1
                    cursor -= 1
                if backslashes % 2 == 0:
                    in_string = False
                idx += 2
                continue
            idx += 1
            continue

        if text[idx : idx + 2] == '\\"':
            in_string = True
            idx += 2
            continue
        if text[idx] == "[":
            depth += 1
        elif text[idx] == "]":
            depth -= 1
            if depth == 0:
                return idx
        idx += 1
    raise ValueError("escaped JSON array end not found")


def transcript_entries_to_text(entries: list[dict]) -> str:
    lines = []
    buffer = []
    next_boundary_seconds = 0

    for entry in entries:
        start_seconds = float(entry.get("start", 0) or 0)
        while start_seconds >= next_boundary_seconds:
            if buffer:
                lines.append(collapse_space(" ".join(buffer)))
                buffer = []
            lines.append(format_timestamp(int(next_boundary_seconds * 1000)))
            next_boundary_seconds += 60

        text = collapse_space(str(entry.get("text", "")).replace("\\n", " "))
        if text:
            buffer.append(text)

    if buffer:
        lines.append(collapse_space(" ".join(buffer)))

    transcript = "\n".join(line for line in lines if line.strip()).strip() + "\n"
    if not transcript.strip():
        raise RuntimeError("empty transcript after parsing Glasp entries")
    return transcript


def parse_jsonish(text: str) -> dict:
    candidates = [text, html.unescape(text)]
    tagless = re.sub(r"<[^>]+>", "", text)
    candidates.extend([tagless, html.unescape(tagless)])
    stripped = text.strip()
    if "{" in stripped and "}" in stripped:
        candidates.append(stripped[stripped.find("{") : stripped.rfind("}") + 1])

    last_error = None
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            last_error = exc
    raise ValueError(f"could not parse JSON: {last_error}")


def format_timestamp(ms: int) -> str:
    total_seconds = max(0, ms // 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"


def json3_to_transcript(data: dict) -> str:
    lines = []
    buffer = []
    next_boundary_ms = 0

    for event in data.get("events", []):
        if "segs" not in event:
            continue
        start_ms = int(event.get("tStartMs", 0) or 0)
        while start_ms >= next_boundary_ms:
            if buffer:
                lines.append(collapse_space(" ".join(buffer)))
                buffer = []
            lines.append(format_timestamp(next_boundary_ms))
            next_boundary_ms += 60_000

        text = "".join(seg.get("utf8", "") for seg in event.get("segs", []))
        text = collapse_space(text)
        if text:
            buffer.append(text)

    if buffer:
        lines.append(collapse_space(" ".join(buffer)))

    transcript = "\n".join(line for line in lines if line.strip()).strip() + "\n"
    if not transcript.strip():
        raise RuntimeError("empty transcript after parsing json3 events")
    return transcript


def collapse_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str, max_chars: int = 60) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    slug = re.sub(r"-+", "-", slug)
    if len(slug) > max_chars:
        slug = slug[:max_chars].rstrip("-")
    return slug or "untitled"


def word_count(transcript: str) -> int:
    return len(re.findall(r"\b[\w']+\b", transcript))


def manifest_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def process_video(video: dict, api_key: str) -> dict:
    video_id = video["video_id"]
    course = video["course"]
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    raw_html = firecrawl_scrape_raw_html(youtube_url, api_key)
    player_response = extract_player_response(raw_html)
    title = extract_title(raw_html, player_response)
    track = select_english_track(caption_tracks(player_response))
    base_url = track.get("baseUrl")
    if not base_url:
        raise RuntimeError("selected caption track missing baseUrl")

    timedtext_url = with_json3_format(base_url)
    try:
        transcript_json = fetch_timedtext_json3(timedtext_url, api_key)
        transcript = json3_to_transcript(transcript_json)
    except Exception:
        try:
            glasp_title, transcript = fetch_glasp_reader_transcript(video_id, api_key)
            if glasp_title and glasp_title != "untitled":
                title = glasp_title
        except Exception:
            savesubs_title, transcript = fetch_savesubs_transcript(video_id, api_key)
            if savesubs_title and savesubs_title != "untitled":
                title = savesubs_title

    out_dir = OUT_ROOT / course / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}__{slugify(title)}.txt"
    out_path.write_text(transcript)

    return {
        "course": course,
        "video_id": video_id,
        "title": title,
        "status": "ok",
        "file_path": str(out_path),
        "word_count": word_count(transcript),
        "error": "",
    }


def write_manifest(results: list[dict]) -> None:
    ok_count = sum(1 for result in results if result["status"] == "ok")
    failed_count = len(results) - ok_count
    total_words = sum(result.get("word_count", 0) for result in results)

    lines = [
        "# Berkeley Course Transcripts",
        "",
        "| course | video_id | title | status | file path | word count |",
        "|---|---|---|---|---|---:|",
    ]
    for result in results:
        title = result["title"] or result.get("error", "")
        lines.append(
            "| {course} | {video_id} | {title} | {status} | {file_path} | {word_count} |".format(
                course=manifest_escape(result["course"]),
                video_id=manifest_escape(result["video_id"]),
                title=manifest_escape(title),
                status=manifest_escape(result["status"]),
                file_path=manifest_escape(result["file_path"]),
                word_count=result.get("word_count", 0),
            )
        )

    lines.extend(
        [
            "",
            f"Totals: {ok_count} ok, {failed_count} failed, {total_words} words.",
            "",
        ]
    )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(lines))


def main() -> int:
    api_key = load_firecrawl_key()
    videos = json.loads(VIDEO_LIST.read_text())
    results = []

    for idx, video in enumerate(videos, start=1):
        video_id = video.get("video_id", "")
        print(f"[{idx}/{len(videos)}] processing {video.get('course')} {video_id}", flush=True)
        try:
            result = process_video(video, api_key)
            print(f"  ok: {result['title']} ({result['word_count']} words)", flush=True)
        except Exception as exc:
            result = {
                "course": video.get("course", ""),
                "video_id": video_id,
                "title": "",
                "status": "failed",
                "file_path": "",
                "word_count": 0,
                "error": str(exc),
            }
            print(f"  failed: {video_id}: {exc}", flush=True)
        results.append(result)
        write_manifest(results)
        if idx < len(videos):
            time.sleep(VIDEO_SLEEP_SECONDS)

    ok_count = sum(1 for result in results if result["status"] == "ok")
    failed_count = len(results) - ok_count
    print(f"Final summary: {ok_count} ok, {failed_count} failed", flush=True)
    return 0 if ok_count >= 25 else 1


if __name__ == "__main__":
    raise SystemExit(main())
