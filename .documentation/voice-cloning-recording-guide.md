# Voice Cloning Recording Guide

A guide for planning and running a fresh voice-actor recording session intended for voice cloning with this project (GPT-SoVITS via Audio-Trainer).

## Context

This project targets **GPT-SoVITS** (see `README.md` and `core/constants.py`). That framework only needs **5–30 minutes** of clean speech to clone a voice well — but *what* those minutes contain matters as much as *how many* there are. A well-designed ~30-minute script beats 2 hours of random chatter for prosody, phonetic coverage, and expressive range.

This document covers:
1. Core principles for designing a training-corpus script
2. A concrete script structure with examples
3. Recording conditions for the actor
4. How to hand the recordings off to the existing preprocessing pipeline

---

## 1. Core principles for training-corpus script design

A good voice-cloning training script hits four targets simultaneously:

| Target | Why it matters for the clone |
|---|---|
| **Phonetic coverage** | Every phoneme (and common phoneme combinations) in the target language must appear, or the model interpolates them poorly. |
| **Prosodic variety** | Declaratives, questions, exclamations, lists, pauses — without these, the clone sounds flat or monotone. |
| **Emotional/stylistic match to end use** | GPT-SoVITS learns the *style* it hears. If all training is deadpan narration, the clone will sound deadpan even when asked to sound excited. |
| **Consistency** | Same mic, same room, same distance, same vocal energy *within a style block*. Consistency > variety for identity. |

### Critical design decision: one style or many?

For a **videogame character voice**, decide upfront:

- **Option A — Single consistent persona.** The actor stays "in character" the entire session. Best for a main character with one dominant tone (e.g., gruff narrator, calm mentor). Produces the strongest, most coherent clone.
- **Option B — Multiple emotion blocks.** Record separate ~5-minute blocks in distinct emotional registers (neutral / angry / scared / excited). This gives more range but risks an averaged-out identity unless you train separate models per emotion or use emotion tagging at inference.

**Recommendation:** For a first project, go with **Option A**. Have the actor pick the character's *default* voice and stay there for the entire training corpus. You can always record an emotion expansion pack later.

---

## 2. Recommended script structure (~30 minutes total)

Break the session into **six blocks**. Aim for ~5 minutes of speech per block. Tell the actor to leave 1–2 seconds of silence between each line (this helps the auto-slicer in the pipeline).

### Block 1 — Phonetically balanced sentences (~8 min, ~40–60 sentences)

Purpose: cover every English phoneme and common phoneme combinations.

Draw from public-domain corpora designed for exactly this:

- **Harvard Sentences** (IEEE Recommended Practice for Speech Quality Measurements) — 720 sentences, each phonetically balanced. Use 30–50 of them.
- **CMU Arctic prompts** — ~1150 sentences, phonetically diverse, public domain.
- **The Rainbow Passage** — classic ~330-word paragraph with near-complete English phoneme coverage.
- **The Grandfather Passage** — speech pathology standard, short and phonetically rich.
- **"The Comma Gets a Cure"** — designed to cover all English lexical sets; excellent for accent capture.

Example Harvard-style lines:
- *"The birch canoe slid on the smooth planks."*
- *"Glue the sheet to the dark blue background."*
- *"It's easy to tell the depth of a well."*
- *"The juice of lemons makes fine punch."*

### Block 2 — Prosodic variety (~5 min)

Purpose: teach the model question intonation, emphasis, list cadence, exclamation.

Include a mix of:
- **Questions (rising intonation):** *"Are you sure this is the right way?"* / *"What do you think happens next?"* / *"Did anyone see where she went?"*
- **Yes/no vs wh-questions:** both intonation patterns
- **Exclamations:** *"That's incredible!"* / *"Watch out!"* / *"I can't believe it."*
- **Lists / enumerations:** *"We'll need rope, water, a map, and a compass."*
- **Conditional/complex sentences:** *"If the door is locked, try the window around the back."*
- **Short interjections:** *"Right."* / *"Of course."* / *"Maybe."* / *"Not now."*

Aim for ~5–15 of each type.

### Block 3 — Hard cases for TTS (~4 min)

Purpose: these are where synthetic voices typically fail. Including them in training dramatically improves quality.

- **Numbers:** *"There were forty-seven survivors on deck number three."* / *"The temperature dropped to minus nineteen degrees."*
- **Dates and times:** *"On July fourth, nineteen seventy-six, at a quarter past noon..."*
- **Proper nouns / names:** *"Doctor Eleanor Whitaker spoke with Commander Okafor."*
- **Acronyms and initialisms:** *"The N.A.S.A. report confirmed the F.B.I.'s findings."* (read as letters)
- **Spelling things out:** *"My name is spelled K-A-T-H-R-Y-N."*
- **Foreign loanwords / place names** relevant to your production (customize)
- **Hyphenates and compounds:** *"It was a well-lit, fast-moving, state-of-the-art facility."*
- **Contractions and reductions:** *"I'd've told you if I'd known."*

### Block 4 — Sentence-length variety (~4 min)

Purpose: teach the model to handle both short barks and long sweeping lines. Keep individual lines under ~15 seconds (the pipeline caps clips at 15 s — see `core/constants.py`).

- **Short (1–3 words):** *"Ready."* / *"Hold position."* / *"It's over."*
- **Medium (one clause):** *"We should move before the storm hits."*
- **Long (two–three clauses):** *"After the gate closed behind us, we realized there was no way back, and the only path led deeper into the ruins."*

Roughly 40% short, 40% medium, 20% long.

### Block 5 — In-character / project-relevant lines (~5 min)

Purpose: this block is what makes it a *videogame/production voice* instead of an audiobook voice. Include the *style* of line the character will actually need to deliver at runtime.

Have the actor record generic-but-in-character lines covering the project's tonal range:
- **Combat barks:** *"Cover me!"* / *"They're flanking us!"* / *"Got one!"*
- **Exploration reactions:** *"What is this place..."* / *"I've never seen anything like it."*
- **Dialogue openers:** *"Listen — I need to tell you something."*
- **Confirmations and refusals:** *"Understood."* / *"Not a chance."*
- **Character-specific vocabulary:** any proper nouns, fictional place names, fantasy terms, or technical jargon from your story world — **include them here**, because the model has never heard them and needs examples to learn their pronunciation.

Write these to match your actual script's tone. If the character is laconic, the lines should be laconic. If expansive and theatrical, match that.

### Block 6 — Held-out evaluation lines (~2–4 min) — **do not include in training**

Purpose: ground truth for measuring clone quality. Write 10–20 lines that the actor records but you *set aside and do not train on*. After training, generate the same lines synthetically and compare A/B. This tells you whether the clone actually sounds like the actor or just sounds like "a voice."

Pick lines that are (a) representative of final production use, (b) contain at least one hard case (number, name, emotion), and (c) not in any of the other blocks.

---

## 3. Recording conditions (tell the actor these)

These are as important as the script itself. A perfect script recorded on a bad setup will produce a mediocre clone.

### Environment
- Quiet room, minimal reverb (closet, bedroom with soft furniture, booth — not kitchen, not bathroom, not echoey hallway)
- HVAC off during recording
- Phone on airplane mode, laptop fans away from mic
- No background hum, no ticking clocks, no computer fan noise

### Mic and technique
- Same microphone for the entire session (do not switch mics between blocks)
- Pop filter or keep mic slightly off-axis to avoid plosives
- Consistent 15–25 cm distance from mouth; don't drift closer/farther
- No mic handling (mounted on stand or boom, not held)
- Avoid clothing rustle, jewelry, chair creaks

### Performance
- Warm up voice before recording (5 min of humming, lip trills, reading aloud)
- Stay consistently in the chosen character voice — do not drop character between lines
- Keep energy level consistent across blocks (don't let Block 4 be tired-sounding)
- Pause 1–2 seconds between lines (helps the auto-slicer; makes editing easy)
- If a line is flubbed, pause, then re-read the whole line — don't try to splice
- Take a 5–10 minute break every 20 minutes to prevent vocal fatigue

### Format (if recording software is configurable)
- 48 kHz or 44.1 kHz (will be downsampled to 32 kHz by the pipeline)
- Mono (or stereo — pipeline converts to mono)
- WAV, 16-bit or 24-bit (not MP3, not AAC — lossless only)
- Peak level around -6 dB to -3 dB; **never clip**

### Session budget
- Plan for **45–90 minutes of actual studio time** to get 30 minutes of usable audio. Retakes, breaks, and fluffs eat into it fast.

---

## 4. Handoff to the preprocessing pipeline

Everything downstream of recording is already built in this project:

1. **Drop the raw WAVs into `data/raw/`** (see `README.md` quick start)
2. **Run `make preprocess`** — this normalizes to 32 kHz mono, denoises via Demucs, slices on silence (2–15 s clips, -40 dB threshold — see `configs/preprocessing.yaml`), and transcribes with Whisper large-v3
3. **Spot-check the output** in `data/processed/<speaker>/wavs/` and review `transcripts.list` for Whisper errors on any character-specific vocabulary (Block 5 content — correct these manually)
4. **Set aside the Block 6 held-out lines** — do not let them end up in the training folder
5. **Run `make train`** (or `make train-quick` first for a pipeline smoke test)
6. **Evaluate** by generating your Block 6 lines synthetically and comparing to the real recordings

### Relevant files in this repo

- `README.md` — pipeline commands
- `core/constants.py` — target audio specs (32 kHz / mono / 16-bit) and clip duration bounds (2–15 s)
- `configs/preprocessing.yaml` — tunable preprocessing parameters
- `preprocessing/pipeline.py` — the orchestrator that will consume these recordings

---

## Verification — how to know the recording session succeeded

1. You have ~30 min of audio across 6 blocks, with Block 6 quarantined separately
2. Waveforms show no clipping (peaks below 0 dB), floor noise well below speech level
3. After `make preprocess`, you get 150–400 clips of 2–15 s in the training folder
4. `transcripts.list` matches the audio (spot-check 10 random entries, especially Block 5 project-specific terms)
5. `make train-quick` completes without data errors
6. After full training, synthesized Block 6 lines sound recognizably like the actor — not just "a voice"

---

## Open considerations before committing to a specific script

- **Character tone** — gruff? young? theatrical? neutral narrator? This drives Block 5 heavily.
- **Project vocabulary** — any proper nouns, fictional terms, technical jargon to bake into Blocks 3 and 5?
- **Single character vs multiple characters from the same actor** — if the actor is voicing more than one character, each needs its own separate training corpus and its own cloned model.
