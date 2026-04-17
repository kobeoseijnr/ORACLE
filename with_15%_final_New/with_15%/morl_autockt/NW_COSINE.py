"""
NW (Normalized Weighted Sum) and Cosine Similarity Scalarization Functions
for Multi-Objective Reinforcement Learning (MORL).

This module provides both NumPy and PyTorch implementations of:
  - Cosine similarity scalarization
  - Normalized weighted sum (NW) scalarization
  - Dispatcher functions that select between the two methods

Used by: mo_agent.py, mo_evaluator.py, main.py
"""

import numpy as np

try:
    import torch
except ImportError:
    torch = None


# =============================================================================
# NumPy versions (for reward shaping, evaluation, and post-hoc analysis)
# =============================================================================

def cosine_similarity_scalarization(reward_vector, preference):
    """
    Scalarize multi-objective reward using cosine similarity (as per MORL paper).
    This preserves the direction of preference rather than just magnitude.
    
    Args:
        reward_vector: numpy array of objective values
        preference: numpy array of preference weights (normalized)
    
    Returns:
        Scalarized reward value
    """
    # Normalize vectors
    reward_norm = np.linalg.norm(reward_vector)
    pref_norm = np.linalg.norm(preference)
    
    if reward_norm == 0 or pref_norm == 0:
        return 0.0
    
    # Cosine similarity: dot product of normalized vectors
    cosine_sim = np.dot(reward_vector, preference) / (reward_norm * pref_norm)
    
    # Scale by magnitude of reward to preserve scale information
    # This combines direction (cosine similarity) with magnitude
    return cosine_sim * reward_norm


def normalized_weighted_sum_scalarization(reward_vector, preference):
    """
    Scalarize multi-objective reward using normalized weighted sum (NW).
    Preference is normalized to sum to 1, then dot-producted with reward.
    
    Args:
        reward_vector: numpy array of objective values
        preference: numpy array of preference weights
    
    Returns:
        Scalarized reward value
    """
    reward_vector = np.array(reward_vector, dtype=np.float32)
    preference = np.array(preference, dtype=np.float32)
    pref_sum = float(np.sum(preference))
    if pref_sum != 0.0:
        preference = preference / pref_sum
    return float(np.dot(reward_vector, preference))


def scalarize_reward(reward_vector, preference, method="cosine"):
    """
    Dispatcher: scalarize multi-objective reward using the specified method.
    
    Args:
        reward_vector: numpy array of objective values
        preference: numpy array of preference weights
        method: 'cosine' or 'nw'
    
    Returns:
        Scalarized reward value
    """
    if method == "nw":
        return normalized_weighted_sum_scalarization(reward_vector, preference)
    else:
        return cosine_similarity_scalarization(reward_vector, preference)


# =============================================================================
# PyTorch versions (for Q-value scalarization in DQN action selection & update)
# =============================================================================

def cosine_similarity_scalarization_torch(q_values, preference):
    """
    Scalarize multi-objective Q-values using cosine similarity (PyTorch version).
    
    Args:
        q_values: torch.Tensor of shape (batch, actions, objectives)
        preference: torch.Tensor of shape (batch, objectives)
    
    Returns:
        Scalarized Q-values: torch.Tensor of shape (batch, actions)
    """
    # Normalize Q-values and preferences
    q_norm = torch.norm(q_values, dim=2, keepdim=True)  # (batch, actions, 1)
    pref_norm = torch.norm(preference, dim=1, keepdim=True).unsqueeze(1)  # (batch, 1, 1)
    
    # Avoid division by zero
    q_norm = torch.clamp(q_norm, min=1e-8)
    pref_norm = torch.clamp(pref_norm, min=1e-8)
    
    # Normalize
    q_normalized = q_values / q_norm  # (batch, actions, objectives)
    pref_normalized = preference.unsqueeze(1) / pref_norm  # (batch, 1, objectives)
    
    # Cosine similarity: dot product of normalized vectors
    cosine_sim = torch.sum(q_normalized * pref_normalized, dim=2)  # (batch, actions)
    
    # Scale by magnitude of Q-values to preserve scale information
    q_magnitude = q_norm.squeeze(2)  # (batch, actions)
    scalarized_q = cosine_sim * q_magnitude
    
    return scalarized_q


def normalized_weighted_sum_scalarization_torch(q_values, preference):
    """
    Scalarize multi-objective Q-values using normalized weighted sum (PyTorch version).
    
    Args:
        q_values: torch.Tensor of shape (batch, actions, objectives)
        preference: torch.Tensor of shape (batch, objectives)
    
    Returns:
        Scalarized Q-values: torch.Tensor of shape (batch, actions)
    """
    # Normalize preference to sum to 1 per batch element
    pref_sum = preference.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (batch, 1)
    pref_normalized = preference / pref_sum  # (batch, objectives)
    
    # Weighted sum: dot product along objectives dimension
    # q_values: (batch, actions, objectives), pref: (batch, 1, objectives)
    scalarized_q = torch.sum(q_values * pref_normalized.unsqueeze(1), dim=2)  # (batch, actions)
    
    return scalarized_q


def scalarize_q_torch(q_values, preference, method="cosine"):
    """
    Dispatcher: scalarize multi-objective Q-values using the specified method (PyTorch).
    
    Args:
        q_values: torch.Tensor of shape (batch, actions, objectives)
        preference: torch.Tensor of shape (batch, objectives)
        method: 'cosine' or 'nw'
    
    Returns:
        Scalarized Q-values: torch.Tensor of shape (batch, actions)
    """
    if method == "nw":
        return normalized_weighted_sum_scalarization_torch(q_values, preference)
    else:
        return cosine_similarity_scalarization_torch(q_values, preference)
