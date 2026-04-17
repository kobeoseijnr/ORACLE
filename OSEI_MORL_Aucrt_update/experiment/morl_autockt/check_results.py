import pandas as pd

print("="*80)
print("MORL RESULTS SUMMARY")
print("="*80)

files = [
    ('Original Targets - MORL', 'morl_autockt_results_original_morl.csv'),
    ('Original Targets - Tolerance', 'morl_autockt_results_original_tolerance.csv'),
    ('15% Increased Targets - MORL', 'morl_autockt_results_15percent_morl.csv'),
    ('15% Increased Targets - Tolerance', 'morl_autockt_results_15percent_tolerance.csv')
]

for name, filename in files:
    try:
        df = pd.read_csv(filename)
        passed = len(df[df['target_reached'] == 'Yes'])
        total = len(df)
        pct = passed/total*100 if total > 0 else 0
        unique_specs = df['spec'].nunique()
        unique_passed = df[df['target_reached'] == 'Yes']['spec'].nunique() if passed > 0 else 0
        
        print(f'\n{name}:')
        print(f'  Total Solutions: {total:,}')
        print(f'  Passed: {passed:,} ({pct:.2f}%)')
        print(f'  Failed: {total-passed:,} ({(total-passed)/total*100:.2f}%)')
        print(f'  Unique Specs: {unique_specs}')
        print(f'  Unique Specs with Passed Solutions: {unique_passed}')
    except Exception as e:
        print(f'\n{name}: Error - {e}')

print("\n" + "="*80)
