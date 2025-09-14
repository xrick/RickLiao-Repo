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