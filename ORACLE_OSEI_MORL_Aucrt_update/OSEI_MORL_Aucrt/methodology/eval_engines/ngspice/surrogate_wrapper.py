"""
Surrogate wrapper that approximates circuit performance without NGSpice.
Use this for testing PD-MORL methodology when NGSpice is not available.
"""
import numpy as np
from collections import OrderedDict

class SurrogateTwoStageClass:
    """
    Surrogate model that approximates TwoStageClass without NGSpice.
    Uses simplified circuit equations based on design principles.
    """
    
    def __init__(self, yaml_path=None, num_process=1, path=None, root_dir=None):
        """Initialize surrogate - ignores most args for compatibility."""
        pass
    
    def create_design_and_simulate(self, state_dict):
        """
        Evaluate circuit performance using surrogate model.
        
        Args:
            state_dict: Dictionary with parameters (mp1, mn1, mp3, mn3, mn4, mn5, cc)
            
        Returns:
            (state, specs_dict, info) where info=0 means success
        """
        # Extract parameters
        mp1 = state_dict.get('mp1', 50)
        mn1 = state_dict.get('mn1', 50)
        mp3 = state_dict.get('mp3', 50)
        mn3 = state_dict.get('mn3', 50)
        mn4 = state_dict.get('mn4', 50)
        mn5 = state_dict.get('mn5', 50)
        cc = state_dict.get('cc', 3e-12)
        
        # Simplified circuit model (approximate relationships)
        # These are rough approximations - for real results, use NGSpice
        
        # Gain ~ sqrt(mp1 * mn1) / sqrt(mn3) * gm_ratio
        gain_linear = np.sqrt(mp1 * mn1) / (np.sqrt(mn3) + 1) * 10
        gain_db = 20 * np.log10(gain_linear + 1) + 50
        gain_db = np.clip(gain_db, 40, 120)
        
        # UGBW ~ gm / (2*pi*Cc) where gm ~ sqrt(mp1)
        # Simplified: higher mp1 -> higher bandwidth, higher cc -> lower bandwidth
        gm_approx = np.sqrt(mp1) * 1e-3  # Approximate transconductance
        ugbw_hz = gm_approx / (2 * np.pi * cc + 1e-15)
        ugbw_hz = np.clip(ugbw_hz, 1e5, 50e6)
        
        # Phase margin: better with higher cc, worse at higher frequencies
        # Simplified model: PM ~ 60 - 10*log10(ugbw/1MHz) + compensation
        phm_deg = 60 - 10 * np.log10(max(ugbw_hz / 1e6, 0.1)) + 5 * np.log10(max(cc * 1e12, 0.1))
        phm_deg = np.clip(phm_deg, 30, 90)
        
        # Bias current ~ (mn4 + mn5) * I_unit (simplified)
        ibias_amp = (mn4 + mn5) * 1e-6
        ibias_amp = np.clip(ibias_amp, 1e-5, 1e-2)
        
        # Return in same format as TwoStageClass
        specs = OrderedDict([
            ('ugbw', ugbw_hz),
            ('gain', gain_db),
            ('phm', phm_deg),
            ('ibias', ibias_amp)
        ])
        
        return state_dict, specs, 0  # info=0 means success

