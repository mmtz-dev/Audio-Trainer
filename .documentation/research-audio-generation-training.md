# Training an Audio Generation Model to Produce Specific Audio

## Research Document

---

## Table of Contents

1. [Overview of Audio Generation Architectures](#1-overview-of-audio-generation-architectures)
2. [Audio Representation & Preprocessing](#2-audio-representation--preprocessing)
3. [Datasets & Data Preparation](#3-datasets--data-preparation)
4. [Model Architectures in Detail](#4-model-architectures-in-detail)
5. [Training Procedure](#5-training-procedure)
6. [Fine-Tuning & Conditioning for Specific Audio](#6-fine-tuning--conditioning-for-specific-audio)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Open-Source Tools & Frameworks](#8-open-source-tools--frameworks)
9. [Practical Step-by-Step Guide](#9-practical-step-by-step-guide)
10. [References](#10-references)

---

## 1. Overview of Audio Generation Architectures

Modern audio generation spans several architectural families, each with distinct trade-offs in quality, speed, controllability, and training cost.

### 1.1 Autoregressive Transformer Models

Models like **Jukebox** (OpenAI, 2020), **AudioLM** (Google, 2022), and **MusicGen** (Meta, 2023) generate audio token-by-token in sequence.

| Aspect | Detail |
|--------|--------|
| **Mechanism** | Predict the next audio token given all previous tokens |
| **Strengths** | High coherence over long sequences; natural modeling of temporal structure |
| **Weaknesses** | Slow inference (sequential generation); can accumulate errors |
| **Best for** | Music generation, speech continuation, long-form audio |

### 1.2 Diffusion Models

**AudioLDM** (2023), **Stable Audio** (Stability AI, 2023), and **Riffusion** (2022) apply denoising diffusion to audio spectrograms or latent spaces.

| Aspect | Detail |
|--------|--------|
| **Mechanism** | Iteratively denoise from Gaussian noise toward target distribution |
| **Strengths** | High-quality output; strong conditioning support (text, embeddings); parallelizable |
| **Weaknesses** | Requires many denoising steps (slow without distillation); memory intensive |
| **Best for** | Text-to-audio, sound effects, music generation with text prompts |

### 1.3 Neural Codec Language Models

**SoundStorm** (Google, 2023), **VALL-E** (Microsoft, 2023), and **MusicGen** combine neural audio codecs with language model decoding.

| Aspect | Detail |
|--------|--------|
| **Mechanism** | Encode audio into discrete tokens via a neural codec (e.g., EnCodec), then model token sequences |
| **Strengths** | Compact representation; fast parallel decoding possible (SoundStorm); high fidelity |
| **Weaknesses** | Quality bounded by codec; requires two-stage training (codec + LM) |
| **Best for** | Speech synthesis, music generation, zero-shot voice cloning |

### 1.4 Flow Matching Models

**VoiceBox** (Meta, 2023), **Audiobox** (Meta, 2024), and **E2 TTS** use continuous normalizing flows.

| Aspect | Detail |
|--------|--------|
| **Mechanism** | Learn a vector field that transports noise to data distribution via ODE solving |
| **Strengths** | Faster than diffusion (fewer steps); strong in-context learning; flexible conditioning |
| **Weaknesses** | Relatively new; fewer pretrained models available |
| **Best for** | Speech synthesis, audio editing/infilling, style transfer |

### 1.5 GANs (Generative Adversarial Networks)

**HiFi-GAN**, **WaveGAN**, **MelGAN** — mostly used as vocoders (spectrogram-to-waveform) rather than end-to-end generators.

| Aspect | Detail |
|--------|--------|
| **Mechanism** | Generator-discriminator adversarial training |
| **Strengths** | Extremely fast inference; high-quality waveform synthesis |
| **Weaknesses** | Training instability; mode collapse; less suited for diverse generation |
| **Best for** | Vocoding (mel-to-waveform), real-time synthesis |

### 1.6 VAE-Based Models

**AudioVAE**, **RAVE** (2021), **SampleRNN** variants use variational autoencoders for learning latent audio spaces.

| Aspect | Detail |
|--------|--------|
| **Mechanism** | Encode audio into continuous latent space, decode back |
| **Strengths** | Smooth latent space for interpolation; fast inference |
| **Weaknesses** | Often blurry/less detailed output compared to diffusion or AR |
| **Best for** | Latent space exploration, real-time performance, audio morphing |

### Architecture Comparison Summary

| Architecture | Quality | Speed | Controllability | Training Cost |
|-------------|---------|-------|----------------|---------------|
| Autoregressive | High | Slow | Medium | High |
| Diffusion | Very High | Medium | High | High |
| Codec LM | High | Fast* | High | Medium |
| Flow Matching | High | Fast | High | Medium |
| GAN | High (vocoding) | Very Fast | Low | Medium |
| VAE | Medium | Fast | Medium | Low |

*SoundStorm-style parallel decoding

---

## 2. Audio Representation & Preprocessing

How audio is represented determines model architecture choices, training efficiency, and output quality.

### 2.1 Raw Waveforms

- **Format**: 1D floating-point signal, typically at 16kHz (speech), 22.05kHz, or 44.1kHz (music)
- **Pros**: No information loss; end-to-end training possible
- **Cons**: Extremely long sequences (1 second at 44.1kHz = 44,100 samples); hard to model directly
- **Used by**: WaveNet, SampleRNN, WaveGAN

### 2.2 Mel Spectrograms

The most common intermediate representation for audio ML:

```
Parameters:
  - Sample rate: 16000-44100 Hz
  - FFT size (n_fft): 1024-2048
  - Hop length: 256-512 samples
  - Number of mel bins: 80-128
  - Frequency range: 20 Hz to Nyquist
```

- **Format**: 2D time-frequency matrix (mel_bins x time_frames)
- **Pros**: Compact; perceptually motivated; compatible with image-based architectures (U-Net, ViT)
- **Cons**: Phase information lost; requires a vocoder (HiFi-GAN, BigVGAN) to reconstruct waveform
- **Used by**: AudioLDM, Riffusion, Tacotron, most TTS systems

### 2.3 Neural Audio Codecs

The breakthrough representation enabling codec language models:

#### EnCodec (Meta, 2022)
- Encodes audio at 24kHz into discrete tokens using **Residual Vector Quantization (RVQ)**
- Produces 1-8 codebook streams at 75 tokens/second per stream
- At 6kbps: 1 second = 75 tokens across ~4 codebooks = 300 tokens (vs. 24,000 raw samples)
- Trained with reconstruction loss + adversarial loss + perceptual loss

#### SoundStream (Google, 2021)
- Similar RVQ architecture to EnCodec
- 12 codebook levels, 50 tokens/second per level
- Backbone: SEANet encoder-decoder with quantization bottleneck

#### DAC — Descript Audio Codec (2023)
- Improved codec with better quality at low bitrates
- 44.1kHz support, 9 codebook levels
- Factorized and improved codebook utilization

**RVQ Explained**: Residual Vector Quantization encodes audio hierarchically. The first codebook captures coarse structure (pitch, rhythm); subsequent codebooks capture finer detail (timbre, texture). This enables coarse-to-fine generation strategies.

### 2.4 CLAP Embeddings

**CLAP** (Contrastive Language-Audio Pretraining) maps both text and audio into a shared embedding space, analogous to CLIP for images.

- Used for conditioning: provide text prompts that steer generation
- Used for evaluation: measure how well generated audio matches text descriptions
- Models: LAION-CLAP, Microsoft CLAP
- Embedding dimension: typically 512

### 2.5 Preprocessing Pipeline

```
Raw Audio (.wav/.mp3/.flac)
    |
    v
Resample to target rate (e.g., 24kHz)
    |
    v
Normalize amplitude (peak or RMS normalization)
    |
    v
Trim silence (leading/trailing, with threshold ~-40dB)
    |
    v
Segment into fixed-length chunks (e.g., 10s, 30s)
    |
    v
Choose representation:
    ├── Mel Spectrogram (for diffusion/U-Net models)
    ├── Neural Codec Tokens (for codec LMs)
    └── Raw Waveform (for end-to-end models)
    |
    v
Pair with metadata/labels/text descriptions
```

---

## 3. Datasets & Data Preparation

### 3.1 Public Datasets

#### Speech
| Dataset | Size | Description |
|---------|------|-------------|
| LibriSpeech | 1,000 hrs | English read speech from audiobooks |
| LibriLight | 60,000 hrs | Unlabeled English speech |
| Common Voice | 20,000+ hrs | Multilingual crowdsourced speech |
| VCTK | 44 hrs | 110 English speakers, clean studio recordings |
| GigaSpeech | 10,000 hrs | Multi-domain English speech |

#### Music
| Dataset | Size | Description |
|---------|------|-------------|
| MusicCaps | 5.5k clips | 10s clips with text descriptions (from Google) |
| FMA (Free Music Archive) | 106 days | 106,574 tracks, Creative Commons |
| MagnaTagATune | 200 hrs | 25,863 clips with tags |
| MTG-Jamendo | 55,000 tracks | CC-licensed music with tags |
| MoisesDB | 240 tracks | Multi-track stems for source separation |

#### Sound Effects & Environmental Audio
| Dataset | Size | Description |
|---------|------|-------------|
| AudioSet | 5,800 hrs | 2M+ YouTube clips, 632 audio event classes |
| ESC-50 | 2,000 clips | 50 environmental sound classes |
| FSD50K | 108 hrs | Freesound clips with AudioSet labels |
| AudioCaps | 46k clips | AudioSet subset with human-written captions |
| WavCaps | 400k clips | Large-scale weakly-labeled audio with captions |

### 3.2 Curating a Custom Dataset

For training a model to produce a **specific** type of audio, data curation is critical:

1. **Define the target domain**: Be precise (e.g., "fingerstyle acoustic guitar in open tunings" not just "guitar music")

2. **Collection strategies**:
   - Record original audio in controlled conditions
   - License audio from stock libraries (Freesound, Splice, etc.)
   - Use existing datasets filtered to your domain
   - Synthetic augmentation of a small seed dataset

3. **Quality filtering**:
   - Remove clips with clipping, excessive noise, or artifacts
   - Filter by SNR (Signal-to-Noise Ratio) — aim for >20dB
   - Use DNSMOS or PESQ scores for automated quality assessment
   - Manual listening pass for small datasets

4. **Labeling & annotation**:
   - Text captions describing each clip (critical for text-conditioned models)
   - Tags/attributes (instrument, tempo, key, mood, etc.)
   - Use audio tagging models (PANNs, BEATs) for automated labeling
   - Use LLMs to generate captions from structured metadata

5. **Segmentation**: Slice long recordings into consistent-length chunks (5-30 seconds typical)

6. **Dataset size guidelines**:
   - Fine-tuning a pretrained model: 1-10 hours can be sufficient
   - Training from scratch (small model): 100+ hours
   - Training from scratch (large model): 1,000+ hours
   - More data with less curation often beats less data perfectly curated

### 3.3 Data Augmentation

- **Pitch shifting**: shift +/- 1-2 semitones
- **Time stretching**: 0.9x-1.1x speed without pitch change
- **Additive noise**: low-level background noise injection
- **RIR (Room Impulse Response) convolution**: simulate different acoustic environments
- **Random EQ**: slight frequency boosts/cuts
- **Random gain**: +/- 3-6 dB
- **Chunk shuffling**: (for non-temporal tasks) randomly crop different segments
- **SpecAugment**: mask random time/frequency bands in spectrograms

---

## 4. Model Architectures in Detail

### 4.1 Diffusion-Based: AudioLDM / AudioLDM 2

**Paper**: AudioLDM (Liu et al., 2023); AudioLDM 2 (Liu et al., 2024)

**Architecture**:
```
Text Prompt
    |
    v
CLAP Text Encoder --> text embedding (512-d)
    |
    v
Latent Diffusion Model (U-Net in VAE latent space)
    |  - Input: noisy latent z_t + conditioning
    |  - Cross-attention on text embeddings
    |  - T diffusion steps (e.g., 200 training, 25-50 inference with DDIM)
    v
VAE Decoder --> mel spectrogram
    |
    v
HiFi-GAN Vocoder --> waveform
```

**Key details**:
- VAE compresses mel spectrograms by 4-8x in time dimension
- U-Net has ~400M parameters (base), ~700M (large)
- Classifier-free guidance scale typically 2.0-5.0
- AudioLDM 2 adds a GPT-2 "AudioMAE" bridge for multi-modal conditioning

### 4.2 Diffusion-Based: Stable Audio / Stable Audio Open

**Paper**: Evans et al. (2024), Stability AI

**Architecture**:
- Diffusion Transformer (DiT) instead of U-Net — uses transformer blocks with adaptive layer norm
- Operates on latent representations from a custom VAE
- Conditioning: T5 text encoder + timing embeddings (start_time, total_duration)
- Supports variable-length generation up to 95 seconds (Stable Audio 2.0)

**Key innovation**: Timing conditioning allows the model to understand and control temporal structure, generating audio of specified duration with proper intro/outro structure.

### 4.3 Codec Language Models: MusicGen

**Paper**: Copet et al. (2023), Meta

**Architecture**:
```
Text Prompt
    |
    v
T5 Text Encoder --> conditioning embeddings
    |
    v
Transformer Decoder (autoregressive)
    |  - 32 layers, 2048-d hidden, 32 heads (large)
    |  - Cross-attention on text embeddings
    |  - Generates EnCodec tokens
    |  - Codebook interleaving pattern for parallel codebook generation
    v
EnCodec Tokens (4 codebooks x 50 tokens/sec)
    |
    v
EnCodec Decoder --> waveform (32kHz)
```

**Model sizes**: Small (300M), Medium (1.5B), Large (3.3B)

**Codebook patterns** (key innovation):
- **Flat**: generate all codebooks sequentially (slow, highest quality)
- **Parallel**: generate all codebooks at once (fast, lower quality)
- **Delay**: each codebook is offset by 1 step — good speed/quality trade-off

**Conditioning**: text (T5), melody (chromagram extraction + conditioning), audio prompting (prefix with audio tokens)

### 4.4 Codec Language Models: SoundStorm

**Paper**: Borsos et al. (2023), Google

**Architecture**:
- Non-autoregressive, parallel generation using **MaskGIT**-style iterative decoding
- Conditioned on semantic tokens from a SoundStream-based AudioLM pipeline
- Generates all acoustic tokens in parallel, refining over ~16 iterations
- Each iteration: predict masked tokens, keep most confident, re-mask the rest

**Speed**: 100x faster than autoregressive AudioLM while maintaining quality

### 4.5 Flow Matching: VoiceBox / Audiobox

**Paper**: Le et al. (2023), Meta

**Architecture**:
- Continuous normalizing flow (CNF) trained with **flow matching** objective
- Backbone: Transformer encoder (24 layers, 1024-d for VoiceBox)
- Operates on log-mel filterbank features
- **In-context learning**: provide surrounding audio context, model fills in masked region

**Capabilities**: TTS, noise removal, content editing, style transfer, cross-lingual synthesis

**Audiobox** extends VoiceBox with:
- Joint speech + sound generation
- Natural language description-based control
- Unified model for voice and sound effects

### 4.6 Transformer-Based: Jukebox

**Paper**: Dhariwal et al. (2020), OpenAI

**Architecture**:
- VQ-VAE with 3 hierarchical levels of codes
- Separate autoregressive transformers (Sparse Transformers) at each level
- Top-level: coarse structure; mid/bottom: increasing detail
- Conditioned on artist, genre, and lyrics

**Scale**: 5B parameters, trained on 1.2M songs

**Limitations**: Very slow generation (~9 hours for 1 minute at full quality); dated architecture by current standards; but demonstrated that raw audio music generation at scale is feasible.

---

## 5. Training Procedure

### 5.1 Common Hyperparameters

#### Diffusion Models (AudioLDM-style)
```yaml
optimizer: AdamW
learning_rate: 1e-4 to 3e-5 (cosine decay)
weight_decay: 0.01
batch_size: 16-64 per GPU (effective 256+ with gradient accumulation)
ema_decay: 0.9999
diffusion_steps: 1000 (training), 25-200 (inference via DDIM/DPM-Solver)
loss: MSE on predicted noise (epsilon-prediction) or v-prediction
gradient_clipping: 1.0
warmup_steps: 1000-5000
```

#### Codec Language Models (MusicGen-style)
```yaml
optimizer: AdamW
learning_rate: 1e-4 (with warmup + cosine decay)
weight_decay: 0.1
batch_size: 64-192 (effective, across GPUs)
loss: cross-entropy on codec tokens
label_smoothing: 0.0-0.1
dropout: 0.0-0.1
warmup_steps: 4000
total_steps: 500k-1M
gradient_clipping: 1.0
```

#### Flow Matching (VoiceBox-style)
```yaml
optimizer: AdamW
learning_rate: 1e-4
batch_size: 307,200 frames per batch
loss: conditional flow matching (L2 on velocity field)
training_duration: 500k-1M steps
noise_schedule: optimal transport (OT) path
```

### 5.2 Loss Functions

| Model Type | Primary Loss | Auxiliary Losses |
|-----------|-------------|-----------------|
| Diffusion | MSE (noise prediction or v-prediction) | — |
| Codec LM | Cross-entropy (token prediction) | — |
| Flow Matching | L2 (velocity field) | — |
| GAN Vocoder | Multi-scale discriminator loss | Feature matching loss, mel reconstruction loss |
| VAE | Reconstruction (L1/L2 on mel) | KL divergence, adversarial loss, perceptual (STFT) loss |
| Neural Codec (EnCodec) | Reconstruction (time + frequency domain) | Commitment loss, adversarial loss, perceptual loss |

### 5.3 Hardware Requirements

| Task | Minimum | Recommended | Typical Duration |
|------|---------|-------------|-----------------|
| Fine-tune MusicGen-small (300M) | 1x A100 40GB | 1-2x A100 80GB | 2-12 hours |
| Fine-tune MusicGen-large (3.3B) | 4x A100 80GB | 8x A100 80GB | 12-72 hours |
| Fine-tune AudioLDM | 1x A100 40GB | 2-4x A100 80GB | 4-24 hours |
| Train AudioLDM from scratch | 8x A100 80GB | 16-32x A100 80GB | 1-2 weeks |
| Train MusicGen from scratch | 32x A100 80GB | 64x A100 80GB | 1-3 weeks |
| Train neural codec (EnCodec) | 8x A100 80GB | 8-16x A100 80GB | 3-7 days |

**Consumer GPU estimates (fine-tuning only)**:
- RTX 3090/4090 (24GB): Fine-tune small models with gradient checkpointing and reduced batch size
- 2x RTX 4090: Fine-tune medium models
- Not practical: Training from scratch for any serious model

### 5.4 Distributed Training Strategies

- **DDP (Distributed Data Parallel)**: Standard for multi-GPU; replicate model on each GPU
- **FSDP (Fully Sharded Data Parallel)**: Shard model parameters across GPUs; essential for >1B parameter models
- **DeepSpeed ZeRO Stage 2/3**: Alternative to FSDP; good for very large models
- **Mixed precision**: BF16 or FP16 with loss scaling; ~2x speedup and ~half memory
- **Gradient checkpointing**: Trade compute for memory; enables larger batch sizes
- **Flash Attention**: Faster and more memory-efficient attention computation

---

## 6. Fine-Tuning & Conditioning for Specific Audio

This is the core section for producing **specific** audio from a trained model.

### 6.1 Full Fine-Tuning

The most straightforward approach: take a pretrained model and continue training on your target dataset.

```python
# Pseudocode for fine-tuning MusicGen
from audiocraft.models import MusicGen

model = MusicGen.get_pretrained("facebook/musicgen-medium")

# Prepare dataset of (audio, text_description) pairs
# Continue training with lower learning rate
# lr = 1e-5 to 5e-5 (10-100x lower than pretraining)
# Train for 1k-10k steps depending on dataset size
```

**Tips**:
- Use 10-100x lower learning rate than pretraining
- Monitor for overfitting closely (small datasets overfit fast)
- Keep validation set to track quality
- Freeze early layers, fine-tune later layers if overfitting occurs

### 6.2 LoRA / Adapter-Based Fine-Tuning

**LoRA (Low-Rank Adaptation)** adds small trainable matrices to frozen model weights, dramatically reducing memory and training time.

```
Original weight matrix: W (d x d)
LoRA adaptation: W + alpha * (B @ A)
  where A is (d x r), B is (r x d), r << d (e.g., r=4, 8, 16)
  
Trainable parameters: 2 * d * r (vs d * d for full fine-tuning)
Typical reduction: 90-99% fewer trainable parameters
```

**Application to audio models**:
- Apply LoRA to attention layers (Q, K, V, O projections)
- Rank r=4-16 is typically sufficient for style adaptation
- Can train on a single consumer GPU (RTX 3090/4090)
- Multiple LoRA adapters can be merged or swapped for different styles

### 6.3 Classifier-Free Guidance (CFG)

A critical technique for steering generation quality and adherence to conditions.

**During training**: Randomly drop conditioning (replace with null/empty) with probability p=0.1-0.2

**During inference**:
```
output = uncond_output + guidance_scale * (cond_output - uncond_output)
```

- `guidance_scale = 1.0`: no guidance (just conditional)
- `guidance_scale = 3.0-5.0`: typical for good text adherence
- `guidance_scale > 7.0`: strong adherence but potential artifacts/oversaturation

This works for both diffusion and codec LM architectures.

### 6.4 Text-Conditioned Generation with CLAP

**CLAP** (Contrastive Language-Audio Pretraining) enables text-based control:

1. Train or fine-tune CLAP on your domain-specific (audio, text) pairs
2. Use CLAP text embeddings as conditioning for your generation model
3. At inference: encode text prompt with CLAP, generate audio conditioned on embedding

**Advanced technique — CLAP-guided optimization**:
```python
# Generate audio, compute CLAP similarity to target text, backpropagate
for step in range(optimization_steps):
    audio = model.generate(latent)
    audio_embedding = clap.encode_audio(audio)
    text_embedding = clap.encode_text(target_prompt)
    loss = -cosine_similarity(audio_embedding, text_embedding)
    loss.backward()
    optimizer.step()
```

### 6.5 Audio Prompting / Style Transfer

Many modern models support **audio prompting** — providing a reference audio clip that the model should match in style:

**MusicGen melody conditioning**:
- Extract chromagram from reference audio
- Condition generation on chromagram + text
- Output matches melodic structure of reference with style from text

**Audio prefix prompting** (codec LMs):
- Encode reference audio to codec tokens
- Prefix the generation with these tokens
- Model continues in the same style/timbre

**VALL-E style zero-shot cloning**:
- Provide 3-second audio prompt
- Model generates new content in the same voice/style
- Works via in-context learning — the acoustic tokens from the prompt teach the model the target characteristics

### 6.6 Textual Inversion / Concept Learning

Borrowed from image generation (Textual Inversion, DreamBooth):

1. Learn a new "token" embedding that represents your target audio concept
2. Train only the embedding while keeping the model frozen
3. Use the learned token in text prompts: "A song in the style of [V]"

This approach is emerging in audio but less mature than in image generation.

### 6.7 Inference-Time Optimization

Instead of fine-tuning the model, optimize the **input** at inference time:

- **Latent optimization**: Start from random latent, optimize toward target using a perceptual loss or CLAP score
- **Guided diffusion**: Modify the denoising process with gradients from a classifier or CLAP model
- **Iterative refinement**: Generate, evaluate, adjust prompt/seed, regenerate

---

## 7. Evaluation Metrics

### 7.1 Objective Metrics

| Metric | What it measures | How to compute | Good values |
|--------|-----------------|----------------|-------------|
| **FAD** (Frechet Audio Distance) | Distribution-level quality | Compare VGGish/PANN embeddings of generated vs. real audio | < 2.0 (good), < 1.0 (excellent) |
| **FD** (Frechet Distance on CLAP) | Semantic quality | Compare CLAP embeddings | Lower is better |
| **KL Divergence** | Label distribution match | Compare predicted label distributions | < 1.5 (good) |
| **CLAP Score** | Text-audio alignment | Cosine similarity of CLAP embeddings | > 0.3 (good) |
| **IS** (Inception Score) | Quality + diversity | Entropy of classifier predictions | Higher is better |
| **SI-SNR** | Signal reconstruction quality | Scale-invariant signal-to-noise ratio | > 10 dB (good) |

### 7.2 Subjective Metrics

| Metric | What it measures | Method |
|--------|-----------------|--------|
| **MOS** (Mean Opinion Score) | Overall perceived quality | Human listeners rate 1-5 |
| **MUSHRA** | Comparative quality | Rate multiple systems against reference |
| **A/B preference** | Pairwise comparison | Listeners choose preferred output |

### 7.3 Domain-Specific Evaluation

For **specific audio** generation, also consider:
- **Style accuracy**: Does generated audio match the target style? (human eval or classifier)
- **Temporal coherence**: Is the audio consistent over time? (spectral continuity metrics)
- **Diversity**: Are outputs varied or collapsed to one mode? (pairwise distance in embedding space)
- **Controllability**: Does changing the condition meaningfully change the output?

---

## 8. Open-Source Tools & Frameworks

### 8.1 Full Frameworks

| Tool | Maintainer | Models | Link |
|------|-----------|--------|------|
| **AudioCraft** | Meta | MusicGen, AudioGen, EnCodec, Multi Band Diffusion | github.com/facebookresearch/audiocraft |
| **Stable Audio Tools** | Stability AI | Stable Audio Open | github.com/Stability-AI/stable-audio-tools |
| **RAVE** | ACIDS/IRCAM | RAVE (real-time audio VAE) | github.com/acids-ircam/RAVE |
| **Bark** | Suno | Bark (TTS + music + sound effects) | github.com/suno-ai/bark |
| **TorToise TTS** | neonbjb | TorToise (high-quality TTS) | github.com/neonbjb/tortoise-tts |
| **Amphion** | Open-source | Multiple audio generation models | github.com/open-mmlab/Amphion |
| **ESPnet** | CMU/Johns Hopkins | TTS, speech enhancement, separation | github.com/espnet/espnet |

### 8.2 Audio Processing Libraries

| Library | Purpose |
|---------|---------|
| **torchaudio** | PyTorch audio loading, transforms, augmentation |
| **librosa** | Audio analysis (mel spectrograms, feature extraction) |
| **soundfile** | Read/write audio files |
| **audiomentations** | Audio data augmentation |
| **pedalboard** | Audio effects (Spotify) |
| **einops** | Tensor manipulation (essential for model code) |

### 8.3 Pretrained Models (Hugging Face)

| Model | Parameters | Modality | HF ID |
|-------|-----------|----------|-------|
| MusicGen Small | 300M | Music | facebook/musicgen-small |
| MusicGen Medium | 1.5B | Music | facebook/musicgen-medium |
| MusicGen Large | 3.3B | Music | facebook/musicgen-large |
| AudioLDM 2 | 712M | General audio | cvssp/audioldm2 |
| AudioLDM 2 Music | 712M | Music | cvssp/audioldm2-music |
| Stable Audio Open | 1.2B | Music + SFX | stabilityai/stable-audio-open-1.0 |
| Bark | 800M | Speech/Music/SFX | suno/bark |
| EnCodec | 15M | Audio codec | facebook/encodec_24khz |
| CLAP | 155M | Audio-text embeddings | laion/larger_clap_music |

### 8.4 Training Infrastructure

| Tool | Purpose |
|------|---------|
| **PyTorch Lightning / Fabric** | Training loop management, multi-GPU |
| **Hugging Face Accelerate** | Distributed training abstraction |
| **DeepSpeed** | Large model training optimization |
| **Weights & Biases** | Experiment tracking |
| **Hydra** | Configuration management |

---

## 9. Practical Step-by-Step Guide

A concrete plan for training/fine-tuning a model to generate a specific type of audio.

### Step 1: Define Your Target

Clearly define what "specific audio" means for your use case:
- What genre/type? (e.g., "lo-fi hip hop beats", "industrial machinery sounds", "bird songs")
- What duration? (5s, 30s, 3min?)
- What conditioning? (text prompts, audio references, unconditional?)
- What quality bar? (proof-of-concept vs. production)

### Step 2: Choose Your Architecture

**Recommended starting points by use case**:

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Music generation (text-conditioned) | MusicGen (fine-tune) | Best quality/ecosystem balance |
| Sound effects (text-conditioned) | AudioLDM 2 or Stable Audio Open | Strong text-audio alignment |
| Voice/speech | VoiceBox-style or VALL-E | In-context learning, zero-shot |
| Real-time / interactive | RAVE | Low latency VAE |
| General purpose | Stable Audio Open | Good all-rounder, open weights |

### Step 3: Prepare Your Dataset

```bash
# Example dataset structure
dataset/
├── audio/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
├── metadata.json        # or metadata.csv
│   # {"sample_001": {"caption": "bright acoustic guitar arpeggio in C major",
│   #                  "tags": ["guitar", "acoustic", "arpeggio", "major"]}}
└── splits/
    ├── train.txt        # list of training sample IDs
    └── valid.txt        # list of validation sample IDs
```

**Minimum viable dataset sizes**:
- Fine-tuning with LoRA: 30 minutes - 2 hours
- Full fine-tuning: 2-10 hours
- Training from scratch: 100+ hours

### Step 4: Set Up Environment

```bash
# Core dependencies
pip install torch torchaudio
pip install audiocraft           # for MusicGen
pip install stable-audio-tools   # for Stable Audio
pip install librosa soundfile    # audio processing
pip install wandb                # experiment tracking
pip install einops flash-attn    # efficient training

# For LoRA fine-tuning
pip install peft                 # Hugging Face PEFT library
```

### Step 5: Fine-Tune

**Option A: MusicGen fine-tuning with AudioCraft**

```python
# 1. Load pretrained model
from audiocraft.models import MusicGen
model = MusicGen.get_pretrained("facebook/musicgen-medium")

# 2. Prepare training data as (audio_tensor, text_description) pairs
# 3. Set up training loop with:
#    - lr: 1e-5
#    - optimizer: AdamW
#    - batch_size: 4-8 per GPU
#    - gradient accumulation: 4-8 steps
#    - max_steps: 2000-10000
#    - eval every: 500 steps
#    - save checkpoints every: 1000 steps
```

**Option B: Stable Audio Open fine-tuning**

```python
# Uses the stable-audio-tools training pipeline
# Configure via YAML:
# model_config:
#   model_type: "diffusion_cond"
#   ...
# training:
#   learning_rate: 1e-5
#   batch_size: 4
#   ...
# dataset:
#   path: "./dataset"
#   ...
```

**Option C: LoRA fine-tuning (memory-efficient)**

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                    # rank
    lora_alpha=16,          # scaling
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(base_model, lora_config)
# Trainable params: ~0.5% of total
# Can fine-tune on single 24GB GPU
```

### Step 6: Evaluate

```python
# 1. Generate a batch of samples with various prompts
samples = model.generate(prompts, duration=10.0, cfg_scale=3.0)

# 2. Compute FAD against held-out real audio
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance(model_name="vggish")
score = fad.score(generated_dir, reference_dir)

# 3. Compute CLAP score for text-audio alignment
from laion_clap import CLAP_Module
clap = CLAP_Module(enable_fusion=False)
clap.load_ckpt()
similarity = clap.compute_similarity(audio_embeddings, text_embeddings)

# 4. Listen to samples manually — automated metrics don't catch everything
```

### Step 7: Iterate

- If **quality is low**: More data, longer training, larger model
- If **overfitting**: Reduce lr, add dropout/augmentation, use LoRA instead of full fine-tune
- If **style doesn't match**: Better text descriptions, more domain-specific data, higher CFG scale
- If **diversity is low**: Lower CFG scale, temperature sampling, more diverse training data
- If **artifacts present**: Check for data quality issues, reduce learning rate, use EMA

---

## 10. References

### Key Papers

1. **AudioLM**: Borsos et al., "AudioLM: a Language Modeling Approach to Audio Generation" (2022)
2. **MusicLM**: Agostinelli et al., "MusicLM: Generating Music From Text" (2023)
3. **MusicGen**: Copet et al., "Simple and Controllable Music Generation" (2023)
4. **AudioLDM**: Liu et al., "AudioLDM: Text-to-Audio Generation with Latent Diffusion Models" (2023)
5. **AudioLDM 2**: Liu et al., "AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining" (2024)
6. **Stable Audio**: Evans et al., "Fast Timing-Conditioned Latent Audio Diffusion" (2024)
7. **SoundStorm**: Borsos et al., "SoundStorm: Efficient Parallel Audio Generation" (2023)
8. **VoiceBox**: Le et al., "VoiceBox: Text-Guided Multilingual Universal Speech Generation at Scale" (2023)
9. **Audiobox**: Vyas et al., "Audiobox: Unified Audio Generation with Natural Language Prompts" (2024)
10. **VALL-E**: Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (2023)
11. **EnCodec**: Defossez et al., "High Fidelity Neural Audio Compression" (2022)
12. **SoundStream**: Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec" (2021)
13. **CLAP**: Wu et al., "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation" (2023)
14. **Jukebox**: Dhariwal et al., "Jukebox: A Generative Model for Music" (2020)
15. **HiFi-GAN**: Kong et al., "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (2020)
16. **RAVE**: Caillon & Esling, "RAVE: A variational autoencoder for fast and high-quality neural audio synthesis" (2021)
17. **DAC**: Kumar et al., "High-Fidelity Audio Compression with Improved RVQGAN" (2023)
18. **Classifier-Free Guidance**: Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022)
19. **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
20. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling" (2023)

### Repositories

- AudioCraft: `github.com/facebookresearch/audiocraft`
- Stable Audio Tools: `github.com/Stability-AI/stable-audio-tools`
- AudioLDM: `github.com/haoheliu/AudioLDM`
- RAVE: `github.com/acids-ircam/RAVE`
- Bark: `github.com/suno-ai/bark`
- Amphion: `github.com/open-mmlab/Amphion`
- LAION-CLAP: `github.com/LAION-AI/CLAP`
- Frechet Audio Distance: `github.com/google-research/google-research/tree/master/frechet_audio_distance`

---

*Document created: 2026-03-30*
*Project: Audio-Trainer*
