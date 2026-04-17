"""Quick diagnostic: test if NGSpice + environment works."""
import sys, os
sys.path.insert(0, 'methodology')
sys.path.insert(0, os.path.join('methodology', 'autockt'))

# Step 1: Check ngspice binary
from eval_engines.ngspice.ngspice_wrapper import NgSpiceWrapper
print("Step 1: NgSpiceWrapper imported OK")

# Step 2: Try to create the environment
from methodology.autockt.envs.ngspice_vanilla_opamp import TwoStageAmp
env_config = {'generalize': True, 'num_valid': 5, 'save_specs': False, 'run_valid': True}
env = TwoStageAmp(env_config)
print(f"Step 2: TwoStageAmp created. NGSpice cmd: {env.sim_env.ngspice_cmd}")

# Step 3: Reset and check specs
state = env.reset()
print(f"Step 3: Reset OK. State len={len(state)}")
print(f"  specs_id: {env.specs_id}")
print(f"  cur_specs: {env.cur_specs}")

# Step 4: Take one step and check if simulation works
action = [1] * 7  # KEEP all params
next_state, reward, done, info = env.step(action)
print(f"Step 4: Step OK. reward={reward:.4f}, done={done}")
print(f"  cur_specs after step: {env.cur_specs}")

# Step 5: Try running ngspice directly with debug
print("\nStep 5: Running NGSpice directly...")
cmd = f'"{env.sim_env.ngspice_cmd}" --version'
print(f"  Command: {cmd}")
exit_code = os.system(cmd)
print(f"  Exit code: {exit_code}")
