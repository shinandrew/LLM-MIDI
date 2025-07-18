# Symbolic Music Generation using LLMs

This repository accompanies the paper "[Large Language Models' Internal Perception of Symbolic Music](https://arxiv.org/abs/2507.12808)". It contains code to generate symbolic (MIDI) music data using a GPT-based language model conditioned on musical genre and style.

## Overview

The script `generate.py` generates 4-track MIDI files with the following characteristics:
- **Tracks**: `melody`, `chords`, `bass`, `rhythm`
- **Format**: JSON-based event representation, then rendered to `.mid` format
- **Conditioning**: Music is generated based on 13 genres (from [TOP-MAGD](https://www.ifs.tuwien.ac.at/mir/msd/TopMAGD.html)) and 25 styles (from [MASD](https://repositori-api.upf.edu/api/core/bitstreams/4aef9065-3cec-4356-8d6f-3c0825e1a45b/content))
- **Output**: Each genre-style combination produces a folder with MIDI files

## Usage

1. Set your OpenAI API key in `generate.py`:
   ```python
   client = OpenAI(api_key="{YOUR_OPENAI_API_KEY}")
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python3 generate.py
   ```

By default, it generates one MIDI file per genre-style combination and stores them under midi_data/.

## Notes

- MIDI structure:
  - **Melody/Chords/Bass**: played on General MIDI instruments
  - **Rhythm**: mapped to drum notes (e.g., kick, snare, hi-hat)
- Failed generations (e.g. invalid JSON output from the LLM) are skipped after retries.

## Citation

If you use this code, please cite our paper:

**"Large Language Models' Internal Perception of Symbolic Music"**, Andrew Shin and Kunitake Kaneko, arxiv 2025.


