# ==============================================================================
# Implementation of Advanced Statistical GOP Calculation Methods
# ==============================================================================
# This script provides Python implementations for the five advanced statistical
# methods discussed for calculating Goodness of Pronunciation (GOP) scores.
# Each function is self-contained and includes detailed comments.
#
# To run this script, you need the following libraries:
# pip install numpy scipy scikit-learn statsmodels
# ==============================================================================

import numpy as np
from scipy.stats import skew, kurtosis, entropy
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.stattools import acf
from typing import List, Dict, Any, Optional, Tuple

# --- 1. Dummy Data Generation ---
# Let's create some sample data to simulate the output of a neural network
# for a single phoneme segment.

# Assume the phoneme segment spans 15 time frames.
T = 15 
# Assume the model's vocabulary has 50 phonemes.
D = 50 

# Let's say the target phoneme has an ID of 7.
TARGET_PHONEME_ID = 7

# Generate a sequence of logits for our target phoneme.
# This simulates the model's confidence in the target phoneme over time.
# A good pronunciation might have high, stable logits.
# A poor one might have low, fluctuating logits.
np.random.seed(42)
logit_sequence = np.random.normal(loc=2.5, scale=1.5, size=T)

# Generate a full logit matrix for the entire vocabulary over the segment.
full_logit_matrix = np.random.normal(loc=0, scale=2, size=(T, D))
# Artificially boost the logits for our target phoneme to make it more realistic.
full_logit_matrix[:, TARGET_PHONEME_ID] = logit_sequence

# Calculate posterior probabilities using softmax for entropy-based methods.
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

posterior_sequence = softmax(full_logit_matrix)

print("--- Setup Complete ---")
print(f"Generated dummy logit sequence for a phoneme with {T} frames.")
print(f"Target Phoneme ID: {TARGET_PHONEME_ID}")
print("-" * 25)


# ==============================================================================
# Non-statistical utility functions needed for experiments
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    """Set common RNG seeds for reproducibility (NumPy and, if available, PyTorch).

    This has no side effects unless explicitly called.
    """
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
    except Exception:
        # Torch is optional here; ignore if unavailable
        pass


def print_system_info() -> "object":
    """Print basic system information and return start timestamp object.

    Returns the datetime object so callers can log durations if needed.
    """
    from datetime import datetime
    import os
    import socket

    start_time = datetime.now()
    print(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Full hostname: {socket.gethostname()}")
    print(f"FQDN: {socket.getfqdn()}")
    try:
        machine_architecture = os.uname().machine  # type: ignore[attr-defined]
        print(f"Machine Architecture: {machine_architecture}")
    except Exception:
        pass
    return start_time


def initialize_model(prep_path: str, cache_dir: Optional[str], device: Optional["object"] = None) -> Tuple["object", "object", "object"]:
    """Initialize Wav2Vec2 feature extractor, tokenizer, and model.

    Imports are local to keep this module lightweight when utilities are unused.
    """
    from transformers import (
        Wav2Vec2FeatureExtractor,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2ForCTC,
    )
    import torch  # type: ignore

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(prep_path, cache_dir=cache_dir)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path, cache_dir=cache_dir)
    model = Wav2Vec2ForCTC.from_pretrained(prep_path, cache_dir=cache_dir)
    model.to(device)
    return feature_extractor, tokenizer, model


def load_audio(audio_input: "object", samplerate: int) -> np.ndarray:
    """Load audio from path or array-like and ensure mono at target samplerate.

    Supports:
    - str path via soundfile (preferred) or torchaudio fallback
    - dict with key 'array' (HF datasets style)
    - numpy array (assumed already at correct samplerate)
    """
    if isinstance(audio_input, str):
        try:
            import soundfile as sf  # type: ignore
            data, sr = sf.read(audio_input)
            if getattr(data, "ndim", 1) > 1:
                data = data.mean(axis=1)
            if sr != samplerate:
                import librosa  # type: ignore
                data = librosa.resample(data, orig_sr=sr, target_sr=samplerate)
            return data.astype(np.float32)
        except Exception:
            try:
                import torchaudio  # type: ignore
                import torch  # type: ignore
                t, sr = torchaudio.load(audio_input)
                t = t.mean(dim=0, keepdim=False) if t.ndim > 1 else t
                if sr != samplerate:
                    t = torchaudio.functional.resample(t.unsqueeze(0), sr, samplerate).squeeze(0)
                return t.numpy().astype(np.float32)
            except Exception as e:
                raise RuntimeError(f"Failed to load audio '{audio_input}': {e}")

    if isinstance(audio_input, dict) and "array" in audio_input:
        arr = np.asarray(audio_input["array"], dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return arr

    arr = np.asarray(audio_input, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr


def tokenize_phonemes(tokenizer: "object", phoneme_sequence: List[str]) -> List[np.ndarray]:
    """Tokenize a list of phoneme strings into ID arrays, removing unknowns."""
    vocab = tokenizer.get_vocab()
    unk_id = vocab.get("[UNK]", -1)
    tokenized: List[np.ndarray] = []
    for p in phoneme_sequence:
        ids = tokenizer(p)["input_ids"]
        arr = np.array(ids, dtype=np.int32)
        if unk_id != -1:
            arr = arr[arr != unk_id]
        tokenized.append(arr)
    return tokenized


def ctc_align_phonemes(
    audio: np.ndarray,
    phoneme_sequence: List[str],
    feature_extractor: "object",
    tokenizer: "object",
    model: "object",
    samplerate: int,
    device: Optional["object"] = None,
) -> List[Dict[str, Any]]:
    """Align phonemes to CTC frames and return timing and frame spans.

    Returns a list of dicts containing:
    - phoneme, start_time, end_time, conf
    - start_frame, end_frame, index_duration

    This function deliberately avoids computing higher-level statistical metrics;
    it focuses on alignment and raw spans so downstream experiments can compute
    their own features.
    """
    import torch  # type: ignore
    import ctc_segmentation  # type: ignore

    assert audio.ndim == 1, "Audio must be mono (1D)."

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = feature_extractor(audio, return_tensors="pt", sampling_rate=samplerate, padding="longest")
    inputs.input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(inputs.input_values).logits.detach().cpu()[0]
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()

    num_frames = probs.shape[0]
    audio_duration = audio.shape[0] / float(samplerate)
    index_duration = audio_duration / float(num_frames)

    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]

    tokenized_phonemes = tokenize_phonemes(tokenizer, phoneme_sequence)
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = index_duration

    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokenized_phonemes)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs, ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(
        config, utt_begin_indices, char_probs, timings, phoneme_sequence
    )

    results: List[Dict[str, Any]] = []
    for p, s in zip(phoneme_sequence, segments):
        start_time = float(s[0])
        end_time = float(s[1])
        conf = float(s[2])
        start_frame = int(max(0, start_time / index_duration))
        end_frame = int(min(num_frames - 1, end_time / index_duration))
        results.append({
            "phoneme": p,
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "conf": round(conf, 3),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "index_duration": index_duration,
        })

    return results


def extract_phoneme_accuracies(example: Dict[str, Any]) -> Dict[str, Any]:
    """Safely extract human phoneme accuracies from dataset example.

    If unavailable, returns a list of Nones matching the phoneme count.
    """
    if "words" in example and example["words"] is not None:
        return {
            "phoneme_accuracies": [acc for word in example["words"] for acc in word.get("phones-accuracy", [])]
        }
    phoneme_count = len(example.get("cmu_ipa_phonetic_transcription", []))
    return {"phoneme_accuracies": [None] * phoneme_count}


# ==============================================================================
# Method 1: Moment-Generating Functions (Skewness & Kurtosis)
# ==============================================================================

def calculate_gop_moments(logits: np.ndarray) -> dict:
    """
    Calculates GOP scores based on higher-order moments of the logit distribution.
    
    Concept:
    Instead of just using variance (2nd moment) like GOP_VarLogit, this method
    also computes Skewness (3rd moment) and Kurtosis (4th moment).
    - Skewness measures the asymmetry of the distribution. A positive skew
      might indicate confidence that builds up and then drops sharply.
    - Kurtosis measures the "tailedness" or "peakiness" of the distribution.
      A high kurtosis means the confidence is highly concentrated around a
      central value, with occasional extreme outliers.
      
    These provide a much richer "shape" profile of the model's confidence
    dynamics than variance alone.
    
    Args:
        logits (np.ndarray): A 1D array of logit values for the target phoneme
                             over its aligned time frames.
                             
    Returns:
        dict: A dictionary containing the skewness and kurtosis.
    """
    if len(logits) < 3: # Skew and Kurtosis are not well-defined for very few points
        return {"skewness": 0.0, "kurtosis": 0.0}
        
    gop_skewness = skew(logits)
    gop_kurtosis = kurtosis(logits) # Fisher's definition (normal=0.0)
    
    return {"skewness": gop_skewness, "kurtosis": gop_kurtosis}

# --- Example for Method 1 ---
gop_moment_scores = calculate_gop_moments(logit_sequence)
print("Method 1: Moment-Based GOP")
print(f"  - Skewness: {gop_moment_scores['skewness']:.4f}")
print(f"  - Kurtosis: {gop_moment_scores['kurtosis']:.4f}")
print("-" * 25)


# ==============================================================================
# Method 2: Information Theory (Entropy & KL Divergence)
# ==============================================================================

def calculate_gop_entropy(posteriors: np.ndarray) -> float:
    """
    Calculates the average Shannon Entropy of the posterior distributions.
    
    Concept:
    This method replaces GOP_DNN by looking at the entire probability distribution
    at each time step, not just the probability of the target phoneme.
    Entropy measures the "uncertainty" or "confusion" of the model.
    A high entropy means the probability is spread out over many phonemes,
    indicating the model is very unsure, which is a strong sign of a poor
    or ambiguous pronunciation. A low entropy means the probability is
    concentrated on one or a few phonemes, indicating confidence.
    
    Args:
        posteriors (np.ndarray): A 2D array of posterior probabilities (output of softmax)
                                with shape (T, D), where T is time frames and D is vocab size.
                                
    Returns:
        float: The mean entropy over all time frames. A higher value suggests
               poorer pronunciation quality due to higher model uncertainty.
    """
    frame_entropies = entropy(posteriors, axis=1)
    return np.mean(frame_entropies)

def calculate_gop_kl_divergence(posteriors: np.ndarray, target_id: int) -> float:
    """
    Calculates the average KL Divergence from an ideal "perfect confidence" distribution.
    
    Concept:
    This method measures how far the model's actual posterior distribution is from an
    "ideal" one. The ideal distribution is a one-hot vector where the target
    phoneme has 100% probability and all others have 0%. The KL Divergence
    quantifies this "distance". A larger divergence means the model's output is
    very different from the ideal, confident prediction, suggesting a mispronunciation.
    
    Args:
        posteriors (np.ndarray): A 2D array of posteriors with shape (T, D).
        target_id (int): The vocabulary index of the target phoneme.
        
    Returns:
        float: The mean KL Divergence. Higher values indicate poorer quality.
    """
    # Create the ideal one-hot distribution
    ideal_distribution = np.zeros(posteriors.shape[1])
    ideal_distribution[target_id] = 1.0
    
    # scipy.stats.entropy calculates KL divergence when two distributions are provided
    kl_divergences = [entropy(p, ideal_distribution) for p in posteriors]
    
    return np.mean(kl_divergences)
    
# --- Example for Method 2 ---
gop_entropy_score = calculate_gop_entropy(posterior_sequence)
gop_kl_score = calculate_gop_kl_divergence(posterior_sequence, TARGET_PHONEME_ID)
print("Method 2: Information Theory GOP")
print(f"  - Mean Entropy: {gop_entropy_score:.4f}")
print(f"  - Mean KL Divergence: {gop_kl_score:.4f}")
print("-" * 25)


# ==============================================================================
# Method 3: Gaussian Mixture Model (GMM) Fitting
# ==============================================================================

def calculate_gop_gmm(logits: np.ndarray, n_components: int = 2) -> dict:
    """
    Fits a Gaussian Mixture Model to the logit sequence and extracts its parameters.
    
    Concept:
    This method assumes that the logit distribution is not simple and unimodal, but
    could be a mixture of multiple underlying states (e.g., onset, steady-state, offset
    of a phoneme). A GMM can model these multi-modal distributions. The parameters
    of the fitted GMM (means, variances, weights of the components) can serve as
    sophisticated features for GOP. For example, a large distance between component
    means could indicate a dramatic shift in confidence during the phonemc.
    
    Args:
        logits (np.ndarray): A 1D array of logit values.
        n_components (int): The number of Gaussian components to fit.
        
    Returns:
        dict: A dictionary containing the means, variances, and weights of the
              fitted GMM components.
    """
    if len(logits) < n_components:
        return {}
        
    # Reshape data for sklearn: (n_samples, n_features)
    reshaped_logits = logits.reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(reshaped_logits)
    
    # Sort components by their means for consistent ordering
    sorted_indices = np.argsort(gmm.means_.flatten())
    
    return {
        "means": gmm.means_.flatten()[sorted_indices],
        "variances": gmm.covariances_.flatten()[sorted_indices],
        "weights": gmm.weights_.flatten()[sorted_indices]
    }
    
# --- Example for Method 3 ---
gop_gmm_features = calculate_gop_gmm(logit_sequence, n_components=2)
print("Method 3: GMM-Based GOP")
if gop_gmm_features:
    print(f"  - Component Means: {np.round(gop_gmm_features['means'], 4)}")
    print(f"  - Component Variances: {np.round(gop_gmm_features['variances'], 4)}")
    print(f"  - Component Weights: {np.round(gop_gmm_features['weights'], 4)}")
else:
    print("  - Not enough data points to fit GMM.")
print("-" * 25)


# ==============================================================================
# Method 4: Time Series Analysis (Autocorrelation)
# ==============================================================================

def calculate_gop_autocorrelation(logits: np.ndarray, lag: int = 1) -> float:
    """
    Calculates the autocorrelation of the logit sequence at a specified lag.
    
    Concept:
    This method treats the logit sequence as a time series, preserving the temporal order
    which methods like variance discard. Autocorrelation at lag 1 measures how strongly
    a logit value at time `t` is related to the value at `t-1`. A high autocorrelation
    implies a smooth, slowly changing confidence level, typical of a well-articulated,
    stable phoneme. A low or negative autocorrelation implies rapid, jittery fluctuations,
    which could be a sign of a pronunciation issue or instability.
    
    Args:
        logits (np.ndarray): A 1D array of logit values.
        lag (int): The time lag to compute autocorrelation for. Lag 1 is most common.
        
    Returns:
        float: The autocorrelation coefficient at the specified lag.
    """
    if len(logits) < 2:
        return 0.0
        
    # nlags specifies the maximum lag to compute. We only need up to the desired lag.
    # fft=False is more stable for short sequences.
    autocorr_values = acf(logits, nlags=lag, fft=False)
    
    return autocorr_values[lag]

# --- Example for Method 4 ---
gop_acf_score = calculate_gop_autocorrelation(logit_sequence, lag=1)
print("Method 4: Time Series GOP")
print(f"  - Autocorrelation at lag 1: {gop_acf_score:.4f}")
print("-" * 25)


# ==============================================================================
# Method 5: Extreme Value Theory (Top-k Mean)
# ==============================================================================

def calculate_gop_evt(logits: np.ndarray, k: int = 3) -> float:
    """
    Calculates the mean of the top-k highest logit values.
    
    Concept:
    This is a more robust alternative to GOP_MaxLogit. Instead of relying on a single
    maximum value, which can be noisy or an outlier, this method averages over the
    top `k` extreme values. This provides a more stable estimate of the model's
    "peak confidence". It answers the question: "When the model is most confident,
    what is its typical confidence level?" rather than "What was the absolute highest
    peak it ever reached?".
    
    Args:
        logits (np.ndarray): A 1D array of logit values.
        k (int): The number of top values to average.
        
    Returns:
        float: The mean of the top-k logit values.
    """
    # Ensure k is not larger than the number of available logits
    k = min(k, len(logits))
    if k == 0:
        return 0.0
        
    # Sort logits in descending order and take the top k
    top_k_logits = np.sort(logits)[-k:]
    
    return np.mean(top_k_logits)

# --- Example for Method 5 ---
gop_evt_score = calculate_gop_evt(logit_sequence, k=3)
print("Method 5: Extreme Value Theory GOP")
print(f"  - Mean of Top-3 Logits: {gop_evt_score:.4f}")
print("-" * 25)


# ==============================================================================
# Experiment workflow (align + compute features + CSV)
# ==============================================================================

def run_experiment(
    prep_path: str,
    cache_dir: Optional[str],
    data_path: str,
    csv_output: str,
    samplerate: int = 16000,
    subset: Optional[int] = None,
    alpha: float = 0.7,
) -> None:
    import csv
    import math
    import os
    try:
        import torch  # type: ignore
        from datasets import load_from_disk  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Missing dependencies for experiment: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_system_info()
    print(f"Using device: {device}")
    feat_extractor, tokenizer, model = initialize_model(prep_path, cache_dir, device)

    print(f"Loading dataset from: {data_path}")
    data = load_from_disk(data_path)
    data = data.map(extract_phoneme_accuracies)
    if subset is not None:
        data = data.select(range(min(subset, len(data))))

    os.makedirs(os.path.dirname(csv_output) or ".", exist_ok=True)
    vocab = tokenizer.get_vocab()

    with open(csv_output, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "uttid", "actual_phoneme", "mispronounced_phoneme",
            "start_time", "end_time", "confidence",
            "prosetrior_probability", "max_logit", "mean_logit_margin", "logit_variance", "combined_score",
            "evt_k3", "skewness", "kurtosis", "autocorr_lag1",
            "entropy_mean", "kl_to_onehot",
            "gmm_means_0", "gmm_means_1", "gmm_vars_0", "gmm_vars_1", "gmm_weights_0", "gmm_weights_1",
            "phoneme_accuracy", "mispronounced",
        ]
        writer.writerow(header)

        for idx, example in enumerate(data):
            uttid = example.get("uttid", idx)
            actual_phonemes = example["cmu_ipa_phonetic_transcription"]
            mispronounced_phonemes = example.get("cmu_ipa_mispronunciation_transcription", [None] * len(actual_phonemes))
            audio = load_audio(example["audio"], samplerate)
            human_accuracies = example.get("phoneme_accuracies", [None] * len(actual_phonemes))

            spans = ctc_align_phonemes(audio, actual_phonemes, feat_extractor, tokenizer, model, samplerate, device)

            with torch.no_grad():
                inputs = feat_extractor(audio, return_tensors="pt", sampling_rate=samplerate, padding="longest")
                inputs.input_values = inputs.input_values.to(device)
                logits = model(inputs.input_values).logits.detach().cpu()[0]
                probs = torch.nn.functional.softmax(logits, dim=-1).numpy()

            num_frames = probs.shape[0]
            vocab = tokenizer.get_vocab()

            for i, (actual, mispronounced) in enumerate(zip(actual_phonemes, mispronounced_phonemes)):
                if i >= len(spans):
                    continue
                s = spans[i]
                start_frame, end_frame = s["start_frame"], s["end_frame"]
                if start_frame > end_frame or not (0 <= start_frame < num_frames):
                    continue
                end_frame = min(end_frame, num_frames - 1)

                target_id = vocab.get(actual)
                import numpy as _np
                metrics = {
                    "prosetrior_probability": _np.nan,
                    "max_logit": _np.nan,
                    "mean_logit_margin": _np.nan,
                    "logit_variance": _np.nan,
                    "combined_score": _np.nan,
                    "evt_k3": _np.nan,
                    "skewness": _np.nan,
                    "kurtosis": _np.nan,
                    "autocorr_lag1": _np.nan,
                    "entropy_mean": _np.nan,
                    "kl_to_onehot": _np.nan,
                    "gmm_means_0": _np.nan,
                    "gmm_means_1": _np.nan,
                    "gmm_vars_0": _np.nan,
                    "gmm_vars_1": _np.nan,
                    "gmm_weights_0": _np.nan,
                    "gmm_weights_1": _np.nan,
                }

                if target_id is not None:
                    seg_logits = logits[start_frame:end_frame + 1]
                    target_logits = seg_logits[:, target_id].numpy()
                    other_logits = seg_logits.clone()
                    other_logits[:, target_id] = -float("inf")
                    max_comp = other_logits.max(dim=-1).values.numpy()

                    seg_probs = probs[start_frame:end_frame + 1]
                    target_probs = seg_probs[:, target_id]

                    metrics["prosetrior_probability"] = float(np.mean(target_probs))
                    metrics["max_logit"] = float(np.max(target_logits))
                    metrics["logit_variance"] = float(np.var(target_logits))
                    metrics["mean_logit_margin"] = float(np.mean(target_logits - max_comp))
                    # Compute combined_score consistent with official implementation
                    gop_score = -float(np.log(metrics["prosetrior_probability"] + 1e-15))
                    metrics["combined_score"] = float(alpha * metrics["mean_logit_margin"] - (1.0 - alpha) * gop_score)

                    metrics["evt_k3"] = float(calculate_gop_evt(target_logits, k=3))
                    m = calculate_gop_moments(target_logits)
                    metrics["skewness"] = float(m.get("skewness", np.nan))
                    metrics["kurtosis"] = float(m.get("kurtosis", np.nan))
                    metrics["autocorr_lag1"] = float(calculate_gop_autocorrelation(target_logits, lag=1))
                    metrics["entropy_mean"] = float(calculate_gop_entropy(seg_probs))
                    try:
                        metrics["kl_to_onehot"] = float(calculate_gop_kl_divergence(seg_probs, target_id))
                    except Exception:
                        pass
                    try:
                        gmm = calculate_gop_gmm(target_logits, n_components=2)
                        if gmm:
                            means = np.asarray(gmm.get("means", [np.nan, np.nan]))
                            vars_ = np.asarray(gmm.get("variances", [np.nan, np.nan]))
                            weights = np.asarray(gmm.get("weights", [np.nan, np.nan]))
                            if means.size >= 2:
                                metrics["gmm_means_0"], metrics["gmm_means_1"] = float(means[0]), float(means[1])
                            if vars_.size >= 2:
                                metrics["gmm_vars_0"], metrics["gmm_vars_1"] = float(vars_[0]), float(vars_[1])
                            if weights.size >= 2:
                                metrics["gmm_weights_0"], metrics["gmm_weights_1"] = float(weights[0]), float(weights[1])
                    except Exception:
                        pass

                human_acc = human_accuracies[i] if i < len(human_accuracies) else None
                writer.writerow([
                    uttid,
                    actual,
                    mispronounced,
                    s["start_time"],
                    s["end_time"],
                    s["conf"],
                    metrics["prosetrior_probability"], metrics["max_logit"], metrics["mean_logit_margin"], metrics["logit_variance"], metrics["combined_score"],
                    metrics["evt_k3"], metrics["skewness"], metrics["kurtosis"], metrics["autocorr_lag1"],
                    metrics["entropy_mean"], metrics["kl_to_onehot"],
                    metrics["gmm_means_0"], metrics["gmm_means_1"], metrics["gmm_vars_0"], metrics["gmm_vars_1"], metrics["gmm_weights_0"], metrics["gmm_weights_1"],
                    human_acc, actual != mispronounced
                ])
            f.flush()
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} examples...")

    print(f"CSV saved to {csv_output}")


if __name__ == "__main__":
    import argparse
    set_seed(42)
    parser = argparse.ArgumentParser(description="Run myImplementation SpeechOcean experiments")
    parser.add_argument("--prep_path", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    parser.add_argument("--cache_dir", default="../official/cache_dir")
    parser.add_argument("--data_path", required=False, help="Path to HF dataset on disk (load_from_disk)")
    parser.add_argument("--csv_output", default="output/myimpl_speechocean_metrics.csv")
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--subset", type=int, default=None, help="Optional number of examples to process")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for combined_score: alpha * mean_logit_margin - (1-alpha) * (-log p)")
    args = parser.parse_args()

    if not args.data_path:
        print("Please provide --data_path pointing to a local HF dataset (load_from_disk).")
    else:
        run_experiment(
            prep_path=args.prep_path,
            cache_dir=args.cache_dir,
            data_path=args.data_path,
            csv_output=args.csv_output,
            samplerate=args.samplerate,
            subset=args.subset,
            alpha=args.alpha,
        )
