"""
Train MORL Agent for Experiment 5 and Save Model
This script trains a MORL agent using the AutoCkt dataset and saves the model for later evaluation.
"""
import os
import sys
import pickle
import numpy as np
import json
import torch
import gym
from pathlib import Path
from datetime import datetime

# Add methodology to path
BASE_DIR = Path(__file__).parent
METHODOLOGY_DIR = BASE_DIR / "methodology"
sys.path.insert(0, str(METHODOLOGY_DIR))

print("="*80)
print("TRAINING MORL AGENT FOR EXPERIMENT 5")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Enable NGSpice simulation environment
import os
os.environ['AUTOCKT_USE_SURROGATE'] = 'true'

# Configuration
config = {
    'dataset_path': BASE_DIR / "dataset" / "ngspice_specs_gen_two_stage_opamp",
    'yaml_config': BASE_DIR / "dataset" / "yaml_config" / "two_stage_opamp.yaml",
    'results_dir': BASE_DIR / "results",
    'models_dir': BASE_DIR / "results" / "models",
    'num_training_specs': 50,  # Train on 50 specs from 1000
    'num_preferences': 10,
    'max_steps': 30,
    'training_episodes': 100,  # Episodes per spec
    'save_frequency': 10  # Save model every N episodes
}

# Create directories
config['results_dir'].mkdir(parents=True, exist_ok=True)
config['models_dir'].mkdir(parents=True, exist_ok=True)

# Import MORL components
try:
    from autockt.envs.autockt_mo_env import AutoCktMOEnv
    from autockt.agents.mo_agent import MO_DQN_Agent, MO_PPO_Agent
    from autockt.utils.mo_utils import generate_preference_vectors
    print("MORL components imported successfully\n")
except ImportError as e:
    print(f"ERROR: Could not import MORL components: {e}")
    print("Please ensure methodology code is in place.")
    sys.exit(1)

def train_morl_agent(config):
    """
    Train MORL agent on AutoCkt dataset and save model.
    
    Returns:
        Path to saved model
    """
    print("="*80)
    print("STEP 1: LOADING DATASET")
    print("="*80)
    
    # Load dataset
    if not config['dataset_path'].exists():
        raise FileNotFoundError(f"Dataset not found: {config['dataset_path']}")
    
    with open(config['dataset_path'], 'rb') as f:
        specs = pickle.load(f)
    
    num_specs = len(list(specs.values())[0])
    print(f"Loaded {num_specs} target specifications from dataset")
    print(f"Dataset path: {config['dataset_path']}\n")
    
    # Select training specs (first 50)
    training_specs = list(range(min(config['num_training_specs'], num_specs)))
    print(f"Training on {len(training_specs)} specifications\n")
    
    print("="*80)
    print("STEP 2: INITIALIZING ENVIRONMENT AND AGENT")
    print("="*80)
    
    # Initialize environment
    try:
        env = AutoCktMOEnv(generalize=True, num_valid=num_specs, run_valid=True)
        print("Environment initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize environment: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Verify environment can reset
    try:
        test_state = env.reset()
        print(f"Environment reset successful. Initial state shape: {test_state.shape if hasattr(test_state, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"ERROR: Environment reset failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Get dimensions
    state_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 20
    
    # Handle Tuple action space
    if isinstance(env.action_space, gym.spaces.Tuple):
        # For Tuple space, action_dim is the number of discrete choices per parameter
        # Each parameter has 3 choices: [-1, 0, 2] -> indices [0, 1, 2]
        action_dim = env.action_space.spaces[0].n  # Should be 3
        num_params = len(env.action_space.spaces)
        print(f"Action space: Tuple of {num_params} Discrete({action_dim}) spaces")
    else:
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else 7
        num_params = 1
    
    reward_dim = 4
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Reward dimension: {reward_dim}\n")
    
    # Initialize agent
    try:
        agent = MO_DQN_Agent(state_dim, action_dim, reward_dim)
        print("Agent initialized: MO_DQN_Agent")
        
        # Verify agent has required methods
        assert hasattr(agent, 'select_action'), "Agent missing select_action method"
        assert hasattr(agent, 'set_preference'), "Agent missing set_preference method"
        assert hasattr(agent, 'memory'), "Agent missing memory buffer"
        print("Agent methods verified\n")
    except Exception as e:
        print(f"ERROR: Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("="*80)
    print("STEP 3: TRAINING AGENT")
    print("="*80)
    print(f"Training episodes per spec: {config['training_episodes']}")
    print(f"Max steps per episode: {config['max_steps']}")
    print(f"Total training episodes: {len(training_specs) * config['training_episodes']}\n")
    
    # Generate preference vectors for training
    # Use 'focused' method which guarantees valid vectors, or 'random' as fallback
    try:
        preferences = generate_preference_vectors(4, method='focused', num_vectors=config['num_preferences'])
        print(f"Generated {len(preferences)} preference vectors using 'focused' method")
    except Exception as e:
        print(f"Warning: 'focused' method failed: {e}")
        preferences = []
    
    if len(preferences) == 0:
        try:
            # Fallback to random if focused doesn't work
            preferences = generate_preference_vectors(4, method='random', num_vectors=config['num_preferences'])
            print(f"Generated {len(preferences)} preference vectors using 'random' method")
        except Exception as e:
            print(f"Warning: 'random' method failed: {e}")
            preferences = []
    
    if len(preferences) == 0:
        # Last resort: use equal weights
        preferences = [np.array([0.25, 0.25, 0.25, 0.25])]
        print("Using default equal-weight preference vector")
    
    # Limit to requested number
    if len(preferences) > config['num_preferences']:
        preferences = preferences[:config['num_preferences']]
    
    print(f"Using {len(preferences)} preference vectors for training\n")
    
    training_history = {
        'episodes': [],
        'rewards': [],
        'specs_trained': training_specs,
        'num_preferences': len(preferences)
    }
    
    episode_count = 0
    
    # Training loop
    print("Starting training loop...\n")
    
    for spec_idx, spec_num in enumerate(training_specs):
        print(f"Training on spec {spec_idx+1}/{len(training_specs)} (spec #{spec_num})...")
        
        try:
            env.base_env.obj_idx = spec_num
            state = env.reset()
            target_spec = env.base_env.specs_ideal.copy()
            print(f"  Spec target loaded: {target_spec}")
        except Exception as e:
            print(f"  ERROR: Failed to initialize spec {spec_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Train with different preferences
        if len(preferences) == 0:
            print(f"  ERROR: No preference vectors available! Skipping spec {spec_num}")
            continue
            
        for pref_idx, preference in enumerate(preferences):
            try:
                agent.set_preference(preference)
                print(f"  Set preference {pref_idx+1}/{len(preferences)}: {preference}")
            except Exception as e:
                print(f"  ERROR: Failed to set preference {pref_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            episodes_per_pref = max(1, config['training_episodes'] // len(preferences))
            print(f"  Training with preference {pref_idx+1}/{len(preferences)} ({episodes_per_pref} episodes)")
            
            for episode in range(episodes_per_pref):
                try:
                    state = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    
                    for step in range(config['max_steps']):
                        try:
                            # Select action
                            epsilon = max(0.1, 1.0 - episode_count / 1000)  # Decay epsilon
                            try:
                                action = agent.select_action(state, preference, epsilon=epsilon)
                                # Convert action to tuple if action space is Tuple
                                if isinstance(env.action_space, gym.spaces.Tuple):
                                    if isinstance(action, (int, np.integer)):
                                        # Single integer: apply same action to all parameters
                                        # Clip to valid range [0, 2] for action_meaning indices
                                        action_val = int(np.clip(action, 0, 2))
                                        action = tuple([action_val] * len(env.action_space.spaces))
                                    elif isinstance(action, (list, np.ndarray)):
                                        # List/array: convert to tuple, pad/truncate as needed
                                        action_list = [int(np.clip(a, 0, 2)) for a in action[:len(env.action_space.spaces)]]
                                        # Pad if too short
                                        while len(action_list) < len(env.action_space.spaces):
                                            action_list.append(0)
                                        action = tuple(action_list)
                                    elif not isinstance(action, tuple):
                                        # Fallback: convert to tuple
                                        action = tuple([int(np.clip(action, 0, 2))] * len(env.action_space.spaces))
                            except Exception as e:
                                print(f"    Error selecting action at step {step}: {e}")
                                import traceback
                                traceback.print_exc()
                                # Random action as fallback
                                if isinstance(env.action_space, gym.spaces.Tuple):
                                    action = tuple([np.random.randint(0, 3) for _ in range(len(env.action_space.spaces))])
                                else:
                                    action = env.action_space.sample()
                            
                            # Take step (returns 5 values: obs, reward, done, truncated, info)
                            try:
                                step_result = env.step(action)
                                if len(step_result) == 5:
                                    next_state, reward_vec, done, truncated, info = step_result
                                    done = done or truncated  # Combine done and truncated
                                elif len(step_result) == 4:
                                    next_state, reward_vec, done, info = step_result
                                else:
                                    raise ValueError(f"Unexpected step return: {len(step_result)} values")
                            except Exception as e:
                                print(f"    ERROR in env.step() at step {step}: {e}")
                                import traceback
                                traceback.print_exc()
                                break
                            
                            # Store experience (if agent has replay buffer)
                            if hasattr(agent, 'memory'):
                                try:
                                    if hasattr(agent.memory, 'push'):
                                        agent.memory.push(state, action, reward_vec, next_state, done, preference)
                                    elif hasattr(agent.memory, 'add'):
                                        agent.memory.add(state, action, reward_vec, next_state, done, preference)
                                except Exception as e:
                                    if episode_count % 100 == 0:
                                        print(f"    Warning: Failed to store experience: {e}")
                            
                            # Update agent (if enough experiences)
                            if hasattr(agent, 'memory') and len(agent.memory) > 32:
                                if episode_count % 4 == 0:  # Update every 4 steps
                                    try:
                                        if hasattr(agent, 'update'):
                                            agent.update(batch_size=32)
                                        elif hasattr(agent, 'update_from_memory'):
                                            agent.update_from_memory()
                                    except Exception as e:
                                        if episode_count % 100 == 0:  # Only print occasionally
                                            print(f"    Warning: Update failed: {e}")
                            
                            state = next_state
                            episode_reward += np.sum(reward_vec) if isinstance(reward_vec, (list, np.ndarray)) else reward_vec
                            episode_steps += 1
                            
                            if done:
                                break
                        except Exception as e:
                            print(f"    Error in episode {episode}, step {step}: {e}")
                            import traceback
                            traceback.print_exc()
                            break
                    
                    episode_count += 1
                    training_history['episodes'].append(episode_count)
                    training_history['rewards'].append(float(episode_reward))
                    
                    if episode_count % 10 == 0:
                        print(f"    Episode {episode_count}, Steps: {episode_steps}, Reward: {episode_reward:.2f}")
                    elif episode_count == 1:
                        print(f"    Episode {episode_count}, Steps: {episode_steps}, Reward: {episode_reward:.2f} (First episode completed!)")
                except Exception as e:
                    print(f"    ERROR in episode {episode}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Still increment episode count to avoid infinite loop
                    episode_count += 1
                    continue
                
                # Save model periodically
                if episode_count % config['save_frequency'] == 0:
                    try:
                        model_path = config['models_dir'] / f"model_checkpoint_ep{episode_count}.pth"
                        save_model(agent, model_path, config, episode_count)
                    except Exception as e:
                        print(f"    Warning: Failed to save checkpoint: {e}")
        
        print(f"  Completed spec {spec_num} ({episode_count} total episodes)\n")
    
    print("="*80)
    print("STEP 4: SAVING FINAL MODEL")
    print("="*80)
    
    # Save final model
    final_model_path = config['models_dir'] / "trained_morl_model_final.pth"
    save_model(agent, final_model_path, config, episode_count)
    
    # Save training history
    history_path = config['results_dir'] / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Total training episodes: {episode_count}\n")
    
    return final_model_path

def save_model(agent, model_path, config, episode_count):
    """Save agent model and configuration"""
    save_dict = {
        'agent_state_dict': agent.q_network.state_dict() if hasattr(agent, 'q_network') else None,
        'preference_net_state_dict': agent.preference_net.state_dict() if hasattr(agent, 'preference_net') else None,
        'optimizer_state_dict': agent.optimizer.state_dict() if hasattr(agent, 'optimizer') else None,
        'config': config,
        'episode_count': episode_count,
        'state_dim': agent.state_dim,
        'action_dim': agent.action_dim,
        'reward_dim': agent.reward_dim
    }
    
    torch.save(save_dict, model_path)
    print(f"  Model checkpoint saved: {model_path}")

def load_model(model_path, state_dim, action_dim, reward_dim):
    """Load agent model from file"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    agent = MO_DQN_Agent(state_dim, action_dim, reward_dim)
    
    if checkpoint['agent_state_dict']:
        agent.q_network.load_state_dict(checkpoint['agent_state_dict'])
    if checkpoint['preference_net_state_dict']:
        agent.preference_net.load_state_dict(checkpoint['preference_net_state_dict'])
    if checkpoint['optimizer_state_dict']:
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return agent

if __name__ == "__main__":
    try:
        model_path = train_morl_agent(config)
        
        print("="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Model saved to: {model_path}")
        print(f"Next step: Run evaluation with: python evaluate_with_saved_model.py")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()

