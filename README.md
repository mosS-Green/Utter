# Utter

Utter is a python tool that converts text and ebook files into audiobooks with natural narration. It utilizes the Gemini 2.5 Flash Native Audio Dialog model via the Live API to provide expressive, human-like voice acting that adapts to the tone and mood of the book chapter by chapter.

## Key Features

- Intelligent Parsing: Automatically identifies chapters and sections in TXT, MD, and EPUB files.
- Advanced Narration: Uses the native audio capabilities of Gemini to ensure proper tone, pacing, and emotional depth.
- Merged Output: Fragments of long chapters are automatically narrated in sequence and merged into a single high-quality MP3 file per chapter.
- Chapter Selection: Allows converting a single chapter, a specific range, or the entire book at once.
- Versatile Voices: Supports multiple built-in voices including Kore, Puck, Charon, and others.

## Prerequisites

- Python 3.10 or higher.
- A Gemini API Key from Google AI Studio.
- ffmpeg installed and available on your system path.

## Installation

Install the required dependencies using pip:

```bash
pip install google-genai ebooklib beautifulsoup4 pydub python-dotenv audioop-lts
```

## Setup

1. Create a file named .env in the root directory.
2. Add your Gemini API Key:
   ```text
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Usage

Run the script and provide the path to your source file:

```bash
python audiobook_gen.py book_title.epub
```

Follow the interactive prompts to select which chapters you wish to convert.

## Configuration

The script supports additional flags:

- --voice: Select a specific voice (e.g., --voice Aoede).
- --format: Choose output format (mp3 or wav).
- --output: Specify a custom output directory.
