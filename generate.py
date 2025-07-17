import os
import random
import json
import time
from mido import MidiFile, MidiTrack, Message, MetaMessage
from openai import OpenAI, OpenAIError
from tqdm import tqdm

# Define genres and styles
TOP_MAGD_GENRES = [
    "Pop", "Rock", "Jazz", "Classical", "Electronic", "Hip-Hop", "Country",
    "Blues", "Folk", "Reggae", "Metal", "Punk", "Ambient"
]
MASD_STYLES = [
    "Baroque", "Romantic", "Modernist", "Minimalist", "Funky", "Groovy", "Swing",
    "Bossa Nova", "Waltz", "March", "Ballad", "Upbeat", "Chill", "Dark", "Bright",
    "Melodic", "Rhythmic", "Syncopated", "Bluesy", "Folky", "Reggaeton", "Grunge",
    "Industrial", "Trance", "Drone"
]

# Initialize OpenAI client
client = OpenAI(api_key="{YOUR_OPENAI_API_KEY}")


def generate_music_sequence(prompt, song_index, max_retries=5):
    """
    Uses GPT model to generate a 4-track MIDI description, returns None if all retries fail.
    """
    moods = ["happy", "sad", "energetic", "calm", "mysterious"]
    mood = random.choice(moods)
    
    instruction = (
        f"Generate a 4-track MIDI description for an 8-bar '{prompt}' with a {mood} mood. "
        "Include exactly these tracks: 'melody', 'chords', 'bass', 'rhythm'. "
        "Each track is a list of (pitch, duration, velocity, start_time) tuples: "
        "pitch (0-127), duration (240=8th, 480=quarter, 960=half), velocity (0-127), "
        "start_time (0-7680 ticks for 8 bars at 120 BPM). Try diverse duration for each pitch."
        "Melody: ~16 events, cohesive sequence. Chords: ~8 events, harmonic support. "
        "Bass: ~8 events, low-end depth. Rhythm: ~16 events, drum pitches (35=kick, 38=snare, 42=hi-hat). "
        "Output ONLY a valid JSON string like "
        "{'melody': [(60, 480, 80, 0), ...], 'chords': [...], 'bass': [...], 'rhythm': [...]} "
        "with no additional text, no explanations, no formatting outside the JSON. "
        "Ensure the JSON is complete and parseable."
    )
    
    temperature = 0.6 + (song_index % 10) * 0.04
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a music generator. Output only a valid JSON string with 4-track MIDI data, no extra text."},
                    {"role": "user", "content": instruction}
                ],
                max_tokens=1200,
                temperature=temperature,
                top_p=0.9
            )
            
            generated_text = response.choices[0].message.content.strip()
            tracks = json.loads(generated_text)
            with open('sample.json','w') as f:
                json.dump(tracks,f)
            
            # Validate required tracks
            if not all(k in tracks for k in ["melody", "chords", "bass", "rhythm"]):
                raise ValueError("Missing required tracks")
            
            # Clamp and validate values
            for track_name in tracks:
                if not isinstance(tracks[track_name], list):
                    raise ValueError(f"Track '{track_name}' is not a list")
                tracks[track_name] = [
                    (max(0, min(127, int(p))), max(240, min(960, int(d))), max(0, min(127, int(v))), max(0, min(7680, int(t))))
                    for p, d, v, t in tracks[track_name]
                ]
                if track_name == "rhythm":
                    tracks[track_name] = [(max(35, min(50, p)), d, v, t) for p, d, v, t in tracks[track_name]]
            
            return tracks
        
        except (json.JSONDecodeError, ValueError, OpenAIError) as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for '{prompt}' (song {song_index + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"Skipping '{prompt}' (song {song_index + 1}) after {max_retries} failed attempts.")
                return None  # Skip to next prompt

def create_midi_file(tracks, filename, tempo=120):
    """Creates a 4-track MIDI file from a tracks dict."""
    mid = MidiFile()
    
    meta_track = MidiTrack()
    mid.tracks.append(meta_track)
    meta_track.append(MetaMessage("set_tempo", tempo=int(60000000 / tempo)))
    meta_track.append(MetaMessage("time_signature", numerator=4, denominator=4))
    meta_track.append(MetaMessage("end_of_track", time=0))
    
    instruments = {
        "melody": 0,   # Piano
        "chords": 0,   # Piano
        "bass": 33,    # Electric Bass
        "rhythm": None # Drums on channel 9
    }
    
    for track_name, events in tracks.items():
        track = MidiTrack()
        mid.tracks.append(track)
        
        if track_name != "rhythm" and instruments[track_name] is not None:
            track.append(Message("program_change", program=instruments[track_name], time=0))
        
        channel = 9 if track_name == "rhythm" else 0
        events = sorted(events, key=lambda x: x[3])
        current_time = 0
        
        for pitch, duration, velocity, start_time in events:
            delta_time = start_time - current_time
            if delta_time < 0:
                delta_time = 0
            track.append(Message("note_on", note=pitch, velocity=velocity, time=delta_time, channel=channel))
            track.append(Message("note_off", note=pitch, velocity=0, time=duration, channel=channel))
            current_time = start_time + duration
    
        track.append(MetaMessage("end_of_track", time=0))
    
    mid.save(filename)

def generate_dataset(output_dir="midi_data", songs_per_combo=50):
    """Generates 50 diverse MIDI files per genre-style combination with progress tracking."""
    os.makedirs(output_dir, exist_ok=True)
    
    total_combinations = len(TOP_MAGD_GENRES) * len(MASD_STYLES)
    skipped_songs = 0
    
    with tqdm(total=total_combinations, desc="Genre-Style Combinations") as pbar_outer:
        for genre in TOP_MAGD_GENRES:
            for style in MASD_STYLES:
                class_dir = os.path.join(output_dir, f"{genre.lower()}_{style.lower()}")
                os.makedirs(class_dir, exist_ok=True)
                
                prompt = f"{genre} song in {style} style"
                with tqdm(total=songs_per_combo, desc=f"{prompt}", leave=False) as pbar_inner:
                    for i in range(songs_per_combo):
                        tracks = generate_music_sequence(prompt, i)
                        if tracks is not None:  # Only save if generation succeeded
                            filename = os.path.join(class_dir, f"{i+1:03d}.mid")
                            create_midi_file(tracks, filename)
                        else:
                            skipped_songs += 1
                        pbar_inner.update(1)
                pbar_outer.update(1)
    
    total_songs = total_combinations * songs_per_combo
    print(f"Generated {total_songs - skipped_songs} MIDI files, skipped {skipped_songs} due to generation failures.")
    print(f"Files saved in {total_combinations} subdirectories under '{output_dir}'.")

if __name__ == "__main__":
    generate_dataset(songs_per_combo=1)
