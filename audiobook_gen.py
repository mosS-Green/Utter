"""
Audiobook Generator — Converts TXT/EPUB files to chapter-wise audiobooks
using Gemini 2.5 Flash Native Audio Dialog via the Live API.

Usage:
    python audiobook_gen.py <file_path>

Dependencies:
    pip install google-genai ebooklib beautifulsoup4 pydub
    (Requires ffmpeg on PATH for pydub WAV→MP3 conversion)

Environment:
    GEMINI_API_KEY — your Gemini API key
"""

import asyncio
import io
import os
import re
import sys
import struct
import wave
import argparse
import textwrap
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("ERROR: Install the Gemini SDK first →  pip install google-genai")


# ─── File Parsers ────────────────────────────────────────────────────────────

def parse_txt(file_path: str) -> list[dict]:
    """
    Parse a .txt file into chapters.
    Heuristic: lines matching 'Chapter <N>' (case-insensitive) are split points.
    Falls back to splitting by double-newline if no chapter headings found.
    """
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")

    # Try splitting on "Chapter N" / "CHAPTER N" patterns
    chapter_pattern = re.compile(
        r"^(chapter\s+\w+[:\-—.]*\s*.*)", re.IGNORECASE | re.MULTILINE
    )
    splits = list(chapter_pattern.finditer(text))

    if len(splits) >= 2:
        chapters = []
        for i, match in enumerate(splits):
            start = match.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            title = match.group(1).strip()
            body = text[start:end].strip()
            chapters.append({"title": title, "body": body})
        return chapters

    # Fallback — treat entire file as one chapter, or split by large gaps
    paragraphs = re.split(r"\n{3,}", text)
    if len(paragraphs) <= 1:
        return [{"title": "Full Text", "body": text.strip()}]

    # Group into ~3000-word chunks as "chapters"
    chapters = []
    buf, word_count, idx = [], 0, 1
    for para in paragraphs:
        words = len(para.split())
        buf.append(para.strip())
        word_count += words
        if word_count >= 3000:
            chapters.append({
                "title": f"Section {idx}",
                "body": "\n\n".join(buf)
            })
            buf, word_count = [], 0
            idx += 1
    if buf:
        chapters.append({
            "title": f"Section {idx}",
            "body": "\n\n".join(buf)
        })
    return chapters


def parse_epub(file_path: str) -> list[dict]:
    """Parse an EPUB file into chapters using ebooklib + BeautifulSoup."""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        sys.exit(
            "ERROR: Install epub dependencies →  pip install ebooklib beautifulsoup4"
        )

    book = epub.read_epub(file_path, options={"ignore_ncx": True})
    chapters = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_body_content(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        if len(text.split()) < 20:  # skip TOC / copyright pages
            continue

        # Try to extract a title from the first heading
        heading = soup.find(re.compile(r"^h[1-3]$"))
        title = heading.get_text(strip=True) if heading else item.get_name()
        chapters.append({"title": title, "body": text})

    return chapters


def load_chapters(file_path: str) -> list[dict]:
    """Dispatch to the right parser based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".txt":
        return parse_txt(file_path)
    elif ext == ".epub":
        return parse_epub(file_path)
    elif ext in (".md", ".markdown"):
        return parse_txt(file_path)  # markdown → same heuristic
    else:
        # Attempt plain-text fallback
        print(f"⚠  Unknown extension '{ext}', attempting plain-text parse…")
        return parse_txt(file_path)


# ─── Text Chunking ───────────────────────────────────────────────────────────

def chunk_text(text: str, max_tokens: int = 4000) -> list[str]:
    """
    Split text into chunks small enough to fit the Live API context
    without getting cut off. Rough heuristic: 1 token ≈ 4 chars.
    We aim for ~4000 tokens ≈ 16,000 chars per chunk, split on paragraph
    boundaries to keep narration natural.
    """
    max_chars = max_tokens * 4
    paragraphs = re.split(r"\n{2,}", text)
    chunks, current = [], ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = ""
        current += para.strip() + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


# ─── Audio Collection via Live API ───────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a world-class audiobook narrator. Your task is to read the
    provided text aloud with expressive, engaging narration that matches
    the tone, mood, and genre of the content.

    Guidelines:
    • Use appropriate pacing — slow for dramatic moments, faster for action.
    • Vary pitch and inflection to differentiate characters in dialogue.
    • Add natural pauses at paragraph breaks and chapter headings.
    • Convey emotion: tension, joy, sadness, humor — whatever the text demands.
    • Do NOT add commentary, explanations, or ad-libs. Read the text faithfully.
    • If the text contains a chapter heading, read it clearly, then pause briefly.
    • Pronounce words carefully and naturally.

    You will receive the text in segments. Read each segment as a continuation
    of the same narration. Maintain consistent voice and tone throughout.
""")


async def generate_audio_for_chunk(
    client: genai.Client,
    chunk: str,
    chunk_idx: int,
    total_chunks: int,
    voice: str = "Kore",
) -> bytes:
    """
    Send a text chunk to the Live API and collect all returned PCM audio bytes.
    Returns raw PCM 24kHz 16-bit mono audio data.
    """
    continuation_note = ""
    if chunk_idx > 0:
        continuation_note = (
            "\n[Continue narrating seamlessly from where you left off. "
            "Do not repeat any previously read text.]\n\n"
        )

    prompt = (
        f"{continuation_note}"
        f"[Segment {chunk_idx + 1} of {total_chunks}]\n\n"
        f"{chunk}"
    )

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(
            parts=[types.Part(text=SYSTEM_PROMPT)]
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice
                )
            )
        ),
        output_audio_transcription={},
    )

    audio_buffer = bytearray()
    transcript_parts = []

    async with client.aio.live.connect(
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        config=config,
    ) as session:
        # Send text as client content
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": prompt}]},
            turn_complete=True,
        )

        # Collect audio responses
        async for response in session.receive():
            content = response.server_content
            if content is None:
                continue

            if content.model_turn:
                for part in content.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        audio_buffer.extend(part.inline_data.data)

            if content.output_transcription and content.output_transcription.text:
                transcript_parts.append(content.output_transcription.text)

            if content.turn_complete:
                break

    transcript = "".join(transcript_parts)
    if transcript:
        # Show a preview of what was narrated
        preview = transcript[:120].replace("\n", " ")
        print(f"    📝 Transcript preview: \"{preview}…\"")

    return bytes(audio_buffer)


# ─── PCM → WAV / MP3 Conversion ─────────────────────────────────────────────

def generate_silence(duration_sec: float = 0.5, sample_rate: int = 24000) -> bytes:
    """Generate silent PCM data (16-bit mono) for the given duration."""
    num_samples = int(sample_rate * duration_sec)
    return b'\x00\x00' * num_samples  # 2 bytes per sample (16-bit)


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
    """Convert raw PCM 16-bit mono data to a WAV byte buffer."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def save_audio(pcm_data: bytes, output_path: str, fmt: str = "mp3"):
    """Save PCM data as WAV or MP3."""
    wav_bytes = pcm_to_wav(pcm_data)

    if fmt == "wav":
        Path(output_path).write_bytes(wav_bytes)
        return

    # Convert to MP3 via pydub (requires ffmpeg)
    try:
        from pydub import AudioSegment
        audio_seg = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        audio_seg.export(output_path, format="mp3", bitrate="192k")
    except ImportError:
        # Fallback: save as WAV
        fallback = output_path.rsplit(".", 1)[0] + ".wav"
        Path(fallback).write_bytes(wav_bytes)
        print(f"  ⚠  pydub not installed. Saved as WAV instead: {fallback}")
    except Exception as e:
        fallback = output_path.rsplit(".", 1)[0] + ".wav"
        Path(fallback).write_bytes(wav_bytes)
        print(f"  ⚠  MP3 conversion failed ({e}). Saved as WAV: {fallback}")


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def display_chapters(chapters: list[dict]):
    """Print a nice chapter listing."""
    print("\n╔══════════════════════════════════════════════════╗")
    print("║            📖  AUDIOBOOK GENERATOR  📖            ║")
    print("╚══════════════════════════════════════════════════╝\n")
    print(f"  Found {len(chapters)} chapter(s):\n")
    for i, ch in enumerate(chapters, 1):
        word_count = len(ch["body"].split())
        title = ch["title"][:50]
        print(f"  {i:>3}.  {title:<50}  ({word_count:,} words)")
    print()


async def process_chapter(
    client: genai.Client,
    chapter: dict,
    chapter_num: int,
    output_dir: str,
    voice: str,
    fmt: str,
):
    """Generate audio for a single chapter."""
    title = chapter["title"]
    body = chapter["body"]
    safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)[:60]

    print(f"\n  🎙  Generating audio for Chapter {chapter_num}: {title}")
    print(f"  {'─' * 55}")

    chunks = chunk_text(body)
    print(f"  📄 Split into {len(chunks)} segment(s)")

    # All chunk audio is merged into one contiguous buffer → one file per chapter
    all_audio = bytearray()
    inter_segment_silence = generate_silence(0.5)  # 0.5s pause between segments

    for i, chunk in enumerate(chunks):
        print(f"  ▶  Processing segment {i + 1}/{len(chunks)}…", end="", flush=True)
        try:
            pcm = await generate_audio_for_chunk(
                client, chunk, i, len(chunks), voice
            )
            if all_audio and pcm:
                all_audio.extend(inter_segment_silence)  # smooth gap between segments
            all_audio.extend(pcm)
            duration_sec = len(pcm) / (24000 * 2)  # 24kHz, 16-bit
            print(f"  ✅ ({duration_sec:.1f}s of audio)")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print(f"      Retrying in 5 seconds…")
            await asyncio.sleep(5)
            try:
                pcm = await generate_audio_for_chunk(
                    client, chunk, i, len(chunks), voice
                )
                if all_audio and pcm:
                    all_audio.extend(inter_segment_silence)
                all_audio.extend(pcm)
                duration_sec = len(pcm) / (24000 * 2)
                print(f"      ✅ Retry succeeded ({duration_sec:.1f}s)")
            except Exception as e2:
                print(f"      ❌ Retry failed: {e2}. Skipping segment.")

    if not all_audio:
        print(f"  ⚠  No audio generated for chapter {chapter_num}.")
        return

    # Final merged audio for the entire chapter
    total_duration = len(all_audio) / (24000 * 2)
    filename = f"Ch{chapter_num:02d}_{safe_title}.{fmt}"
    output_path = os.path.join(output_dir, filename)

    print(f"\n  🔗 Merging {len(chunks)} segment(s) into single file…")
    save_audio(bytes(all_audio), output_path, fmt)

    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)
    print(f"\n  💾 Saved: {filename}")
    print(f"  ⏱  Duration: {minutes}m {seconds}s")


async def main():
    parser = argparse.ArgumentParser(
        description="📖 Convert text files to audiobooks using Gemini Native Audio"
    )
    parser.add_argument("file", help="Path to TXT, EPUB, or MD file")
    parser.add_argument(
        "--voice", default="Kore",
        help="Voice name (default: Kore). Options: Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr"
    )
    parser.add_argument(
        "--format", default="mp3", choices=["mp3", "wav"],
        help="Output format (default: mp3)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: <filename>_audiobook/)"
    )
    args = parser.parse_args()

    # ── Validate ──
    if not os.path.isfile(args.file):
        sys.exit(f"ERROR: File not found: {args.file}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: Set GEMINI_API_KEY environment variable.")

    # ── Parse chapters ──
    print(f"\n  📂 Loading: {args.file}")
    chapters = load_chapters(args.file)
    if not chapters:
        sys.exit("ERROR: No chapters found in the file.")

    display_chapters(chapters)

    # ── Ask for chapter selection ──
    while True:
        try:
            user_input = input(
                f"  Enter chapter number to convert (1-{len(chapters)}), "
                f"'all' for everything, or 'q' to quit: "
            ).strip().lower()

            if user_input == "q":
                print("  👋 Goodbye!")
                return

            if user_input == "all":
                selected = list(range(len(chapters)))
                break

            # Support ranges like "1-5" and comma-separated "1,3,5"
            selected = []
            for part in user_input.split(","):
                part = part.strip()
                if "-" in part:
                    start, end = part.split("-", 1)
                    selected.extend(range(int(start) - 1, int(end)))
                else:
                    selected.append(int(part) - 1)

            # Validate
            if all(0 <= s < len(chapters) for s in selected):
                break
            else:
                print(f"  ⚠  Invalid selection. Choose between 1 and {len(chapters)}.")
        except (ValueError, IndexError):
            print(f"  ⚠  Invalid input. Enter a number, range (1-5), or 'all'.")

    # ── Setup output ──
    stem = Path(args.file).stem
    output_dir = args.output or f"{stem}_audiobook"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  📁 Output directory: {output_dir}")

    # ── Generate ──
    client = genai.Client(api_key=api_key)

    for idx in selected:
        chapter = chapters[idx]
        await process_chapter(
            client, chapter, idx + 1, output_dir, args.voice, args.format
        )

    print("\n╔══════════════════════════════════════════════════╗")
    print("║              ✅  GENERATION COMPLETE              ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  📁 Files saved to: {output_dir}\n")


if __name__ == "__main__":
    asyncio.run(main())
