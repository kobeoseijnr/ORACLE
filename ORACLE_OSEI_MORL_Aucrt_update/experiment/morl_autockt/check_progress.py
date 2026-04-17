"""
Check progress of MORL evaluation
"""

from pathlib import Path
import json
import time

BASE_DIR = Path(__file__).parent
RAW_RESULTS = BASE_DIR / "results" / "morl_autockt_results_raw.json"

print("="*80)
print("MORL EVALUATION PROGRESS CHECK")
print("="*80)

if RAW_RESULTS.exists():
    print(f"\n[OK] Raw results file found: {RAW_RESULTS}")
    
    # Get file size
    size = RAW_RESULTS.stat().st_size
    print(f"  File size: {size:,} bytes ({size/1024/1024:.2f} MB)")
    
    # Get modification time
    mtime = RAW_RESULTS.stat().st_mtime
    mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
    print(f"  Last modified: {mod_time}")
    
    # Try to load and check contents
    try:
        with open(RAW_RESULTS, 'r') as f:
            data = json.load(f)
        
        total_solutions = len(data.get('all_solutions', []))
        reached_count = data.get('reached_count', 0)
        total_evaluated = data.get('total_evaluated', 0)
        
        print(f"\n  Total solutions: {total_solutions:,}")
        print(f"  Targets evaluated: {total_evaluated}/1000")
        print(f"  Targets reached: {reached_count}")
        
        if total_evaluated < 1000:
            print(f"\n  Status: IN PROGRESS ({total_evaluated*100/1000:.1f}% complete)")
            print(f"  Estimated remaining: ~{(1000-total_evaluated)*0.05:.1f} minutes")
        else:
            print(f"\n  Status: COMPLETE")
            print(f"\n  Ready to process results!")
            print(f"  Run: python main.py")
    except Exception as e:
        print(f"  Warning: Could not parse JSON ({e})")
        print(f"  File may still be writing...")
else:
    print(f"\n[WAIT] Raw results file not found yet")
    print(f"  Expected location: {RAW_RESULTS}")
    print(f"  Evaluation is still running...")
    print(f"  This may take 30-60 minutes to complete")

print("\n" + "="*80)
