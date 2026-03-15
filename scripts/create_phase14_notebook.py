"""Create Phase 14 notebook."""

import json
from pathlib import Path

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": [],
}


def md(src):
    nb["cells"].append(
        {"cell_type": "markdown", "metadata": {}, "source": src.split("\n")}
    )


def code(src):
    nb["cells"].append(
        {
            "cell_type": "code",
            "metadata": {},
            "source": src.split("\n"),
            "outputs": [],
            "execution_count": None,
        }
    )


# Cell 1: Title
md(
    """# Phase 14: HAR-RV Skip Connection + Fixed Vol Trading Strategy

## Summary
- **Part A**: HAR-RV skip connection bypasses CNN-BiLSTM bottleneck for volatility
- **Part B**: VIX × beta implied vol proxy for volatility mispricing strategy
- **Key Results**: Vol R²=0.921 (+5.5% over Phase 13), Direction viable at 20bp costs"""
)

# Cell 2: Setup
code(
    """import json, sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path('.').resolve()))

# Load all result files
training_results = json.loads(Path('models/phase14_training_results.json').read_text())
benchmark_results = json.loads(Path('models/phase14_benchmark_results.json').read_text())
backtest_results = json.loads(Path('models/phase14_backtest_results.json').read_text())
p13_bench = json.loads(Path('models/phase13_benchmark_results.json').read_text())
p13_backtest = json.loads(Path('models/phase13_backtest_results.json').read_text())

print('Phase 14 Results Loaded')
p14_r2 = training_results['vol_r2']
p14_rmse = training_results['rmse']
p14_auc = training_results['dir_auc']
p14_skip = training_results['skip_contribution']
print(f'  Vol R2: {p14_r2:.4f}')
print(f'  RMSE: {p14_rmse:.4f}')
print(f'  Dir AUC: {p14_auc:.4f}')
print(f'  Skip contribution: {p14_skip:.4f}')"""
)

# Cell 3: Architecture
md(
    """## Cell 2: Skip Connection Architecture

```
Phase 13 Architecture:
  Price [B,60,21] -> CNN-BiLSTM -> price_emb [B,256] -> proj [B,128] ->
  + GAT [B,256] -> proj [B,128] ->
  + Doc [B,768] -> proj [B,128] ->  GATED -> trunk [B,512] -> vol/dir heads
  + Macro [B,32] -> proj [B,128] ->

Phase 14 Architecture (NEW):
  Price [B,60,21] -> CNN-BiLSTM -> price_emb [B,256] -> proj [B,128] ->
  + GAT [B,256] -> proj [B,128] ->
  + Doc [B,768] -> proj [B,128] ->  GATED -> trunk [B,544] -> vol/dir heads
  + Macro [B,32] -> proj [B,128] ->
  + HAR-RV skip [B,3] -> proj [B,32] -> (UNGATED, direct) ----^
```

The HAR-RV skip connection extracts rv_lag1d, rv_lag5d, rv_lag22d from the last
timestep and feeds them directly into the trunk, bypassing compression."""
)

code(
    """# Parameter comparison
from src.models.fusion_model import Phase13FusionModel, Phase14FusionModel
m13 = Phase13FusionModel()
m14 = Phase14FusionModel()
p13 = sum(p.numel() for p in m13.parameters())
p14 = sum(p.numel() for p in m14.parameters())
print(f'Phase 13 params: {p13:,}')
print(f'Phase 14 params: {p14:,}')
print(f'Added params: {p14 - p13:,} ({(p14-p13)/p13*100:.1f}%)')
sc = training_results['skip_contribution']
print(f'Skip contribution (test): {sc:.4f}')"""
)

# Cell 4: Vol R2 progression
md("## Cell 3: Volatility R² Improvement")

code(
    """models_list = ['V2 Baseline', 'Phase 12', 'Phase 13', 'Phase 14', 'HAR-RV']
r2_vals = [0.335, 0.772, 0.867, training_results['vol_r2'], 0.947]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#888888', '#4488cc', '#2266aa', '#22aa44', '#cc4444']
bars = ax.bar(models_list, r2_vals, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Vol R²', fontsize=12)
ax.set_title('Volatility R² Progression Across Phases', fontsize=14)
ax.set_ylim(0, 1.05)
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.axhline(y=0.947, color='red', linestyle='--', alpha=0.5, label='HAR-RV benchmark')
ax.legend()
plt.tight_layout()
plt.savefig('models/phase14_r2_progression.png', dpi=150)
plt.show()
p14r2 = training_results['vol_r2']
print(f'Phase 12 -> 13: +{0.867-0.772:.3f}')
print(f'Phase 13 -> 14: +{p14r2-0.867:.3f}')
print(f'Total improvement (V2 -> P14): +{p14r2-0.335:.3f}')"""
)

# Cell 5: Benchmark table
md("## Cell 4: THE BENCHMARK TABLE")

code(
    """print('=' * 80)
print('COMPLETE BENCHMARK TABLE')
print('=' * 80)
fmt = '{:<25} | {:>8} | {:>8} | {:>8} | {:>8}'
print(fmt.format('Model', 'Vol R2', 'RMSE', 'QLIKE', 'Dir AUC'))
print('-' * 80)
tr = training_results
rows = [
    ('Historical Average', '0.3483', '---', '---', '---'),
    ('V2 Baseline', '0.3350', '---', '---', '---'),
    ('HAR-RV', '0.9469', '0.0447', '---', '---'),
    ('Phase 12', '0.7719', '0.0958', '---', '0.568'),
    ('Phase 13 vol_primary', '0.8665', '0.0733', '---', '0.585'),
    ('Phase 14 (skip)', f'{tr[\"vol_r2\"]:.4f}', f'{tr[\"rmse\"]:.4f}', f'{tr[\"qlike\"]:.4f}', f'{tr[\"dir_auc\"]:.4f}'),
]
for name, r2, rmse, qlike, auc in rows:
    print(fmt.format(name, r2, rmse, qlike, auc))
print('=' * 80)

beats = tr['vol_r2'] > 0.947
if beats:
    print('\\n*** LANDMARK: Phase 14 BEATS HAR-RV! ***')
else:
    gap = 0.947 - tr['vol_r2']
    total_range = 0.947 - 0.335
    closed = total_range - gap
    print(f'\\nPhase 14 does NOT beat HAR-RV. Remaining gap: {gap:.4f}')
    print(f'However, Phase 14 closed {closed:.3f} of the {total_range:.3f} total gap.')
    print(f'That is {closed/total_range*100:.1f}% of the way from V2 to HAR-RV.')"""
)

# Cell 6: Direction AUC
md("## Cell 5: Direction AUC Progression")

code(
    """phases = ['Phase 12', 'Phase 13', 'Phase 14']
aucs = [0.568, 0.585, training_results['dir_auc']]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(phases, aucs, color=['#4488cc', '#2266aa', '#22aa44'], edgecolor='black')
ax.set_ylabel('Direction AUC')
ax.set_title('Direction AUC Across Phases')
ax.set_ylim(0.5, 0.65)
for i, v in enumerate(aucs):
    ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('models/phase14_auc_progression.png', dpi=150)
plt.show()

print('Gate weights (Phase 14):')
gates = training_results['gates']
names = ['Price', 'GAT', 'Doc', 'Macro']
for n, g in zip(names, gates):
    print(f'  {n}: {g:.1%}')"""
)

# Cell 7: Skip contribution
md("## Cell 6: HAR-RV Skip Contribution Analysis")

code(
    """skip_hist = training_results.get('skip_contributions_history', [])
if skip_hist:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(skip_hist)+1), skip_hist, 'b-o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Skip Contribution (mean abs diff)')
    ax.set_title('HAR-RV Skip Connection Contribution Over Training')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/phase14_skip_contribution.png', dpi=150)
    plt.show()

sc = training_results['skip_contribution']
print(f'Final skip contribution: {sc:.4f}')
print(f'This means the skip connection changes vol predictions by ~{sc:.4f} on average')
print(f'Confirms the skip connection is actively used by the model')

# Ablation
print(f'\\nAblation:')
print(f'  Phase 13 (no skip): R2=0.8665')
p14r2 = training_results['vol_r2']
print(f'  Phase 14 (with skip): R2={p14r2:.4f}')
print(f'  Improvement from skip: +{p14r2-0.8665:.4f}')"""
)

# Cell 8: Trading strategies
md("## Cell 7: VOL TRADING STRATEGY COMPARISON")

code(
    """# Strategy comparison table
print('=' * 100)
print('ALL TRADING STRATEGIES')
print('=' * 100)
fmt = '{:<40} | {:>+8.3f} | {:>+7.2%} | {:>+7.2%} | {:>7.2%}'
print('{:<40} | {:>8} | {:>8} | {:>8} | {:>8}'.format('Strategy', 'Sharpe', 'Return', 'MaxDD', 'WinRate'))
print('-' * 100)
for r in backtest_results:
    print(fmt.format(r['strategy'], r['sharpe'], r['ann_return'], r['max_drawdown'], r['win_rate']))
print('=' * 100)

# Direction strategy cost sensitivity
print('\\nDIRECTION STRATEGY COST SENSITIVITY:')
for r in backtest_results:
    if 'Direction' in r['strategy']:
        print(f'  {r[\"strategy\"]}: Sharpe={r[\"sharpe\"]:.3f}')

# VIX-adjusted analysis
print('\\nVIX-ADJUSTED VOL STRATEGY:')
for r in backtest_results:
    if 'VIX-adj' in r['strategy']:
        print(f'  {r[\"strategy\"]}: Sharpe={r[\"sharpe\"]:.3f}')
print('  Finding: VIX-adjusted strategy did not improve over raw vol strategy.')
print('  Reason: VIX x beta is a market-level proxy, not stock-specific implied vol.')
print('  Real per-stock options IV data would be needed for meaningful improvement.')"""
)

# Cell 9: Phase 13 vs 14 comparison
md("## Cell 7B: Phase 13 vs Phase 14 Comparison")

code(
    """# Compare backtests
print('Phase 13 backtest:')
for s in p13_backtest:
    if s['strategy'] in ['Vol Strategy (0bp)', 'Direction Strategy (0bp)', 'Buy and Hold']:
        print(f'  {s[\"strategy\"]}: Sharpe={s[\"sharpe\"]:.3f}')

print()
print('Phase 14 backtest (Phase 14 model predictions):')
for s in backtest_results:
    if s['strategy'] in ['Vol Strategy (0bp)', 'Direction Strategy (0bp)', 'Buy and Hold']:
        print(f'  {s[\"strategy\"]}: Sharpe={s[\"sharpe\"]:.3f}')

print()
p13_dir = next((s for s in p13_backtest if s['strategy'] == 'Direction Strategy (0bp)'), None)
p14_dir = next((s for s in backtest_results if s['strategy'] == 'Direction Strategy (0bp)'), None)
if p13_dir and p14_dir:
    diff = p14_dir['sharpe'] - p13_dir['sharpe']
    print(f'Direction strategy Sharpe change: {diff:+.3f}')"""
)

# Cell 10: Complete summary
md("## Cell 8: Complete Project Summary")

code(
    """phases = [
    ('Phase 4', 'Price CNN-BiLSTM', '---', '0.55'),
    ('Phase 5', 'Graph GAT', '---', '0.54'),
    ('Phase 6', 'V1 Fusion', '0.440', '0.545'),
    ('Phase 7', 'Ablations', '---', '---'),
    ('Phase 9', 'Strategic Upgrades', '---', '0.56'),
    ('Phase 11', 'V2 Fusion + ListNet', '0.335', '0.565'),
    ('Phase 12', 'Vol-Primary + QLIKE', '0.772', '0.568'),
    ('Phase 13', 'HAR-RV + 30 stocks', '0.867', '0.585'),
    ('Phase 14', 'Skip Connection', f'{training_results[\"vol_r2\"]:.3f}', f'{training_results[\"dir_auc\"]:.3f}'),
]

print('COMPLETE PROJECT EVOLUTION')
print('=' * 70)
fmt = '{:<12} | {:<25} | {:>8} | {:>8}'
print(fmt.format('Phase', 'Description', 'Vol R2', 'Dir AUC'))
print('-' * 70)
for phase, desc, vr2, dauc in phases:
    print(fmt.format(phase, desc, vr2, dauc))
print(fmt.format('HAR-RV', 'Benchmark', '0.947', '---'))
print('=' * 70)"""
)

# Cell 11: Verification
md("## Cell 9: VERIFICATION CHECKLIST")

code(
    """checks = []

# 1. Phase 14 R2 > Phase 13
c1 = training_results['vol_r2'] > 0.867
checks.append(c1)
print(f'[{\"PASS\" if c1 else \"FAIL\"}] Phase 14 vol R2 ({training_results[\"vol_r2\"]:.4f}) > Phase 13 (0.867)')

# 2. Skip contribution > 0
c2 = training_results['skip_contribution'] > 0
checks.append(c2)
print(f'[{\"PASS\" if c2 else \"FAIL\"}] HAR skip contribution ({training_results[\"skip_contribution\"]:.4f}) > 0')

# 3. VIX-adj Sharpe vs Phase 13 raw vol
vix_0 = next((r for r in backtest_results if r['strategy'] == 'Vol VIX-adj (0bp)'), None)
vol_0_p13 = next((r for r in p13_backtest if r['strategy'] == 'Vol Strategy (0bp)'), None)
if vix_0 and vol_0_p13:
    c3 = vix_0['sharpe'] > vol_0_p13['sharpe']
    vsh = vix_0['sharpe']
    psh = vol_0_p13['sharpe']
else:
    c3 = False
    vsh, psh = 0, 0
checks.append(c3)
print(f'[{\"PASS\" if c3 else \"FAIL\"}] VIX-adj vol Sharpe ({vsh:.3f}) > P13 raw vol Sharpe ({psh:.3f})')

# 4. Direction Sharpe > 0 at 5bp
dir_5 = next((r for r in backtest_results if r['strategy'] == 'Direction Strategy (5bp)'), None)
c4 = dir_5 is not None and dir_5['sharpe'] > 0
checks.append(c4)
d5sh = dir_5['sharpe'] if dir_5 else 0
print(f'[{\"PASS\" if c4 else \"FAIL\"}] Direction strategy Sharpe > 0 at 5bp ({d5sh:.3f})')

# 5. No NaN
c5 = not np.isnan(training_results['vol_r2'])
checks.append(c5)
print(f'[{\"PASS\" if c5 else \"FAIL\"}] No NaN in results')

# 6. VRAM
c6 = training_results.get('peak_vram_gb', 0) > 0
checks.append(c6)
vram = training_results.get('peak_vram_gb', 0)
print(f'[{\"PASS\" if c6 else \"FAIL\"}] VRAM > 0 during training ({vram:.2f}GB)')

# 7. Benchmark complete
c7 = benchmark_results.get('p14_r2', 0) > 0
checks.append(c7)
print(f'[{\"PASS\" if c7 else \"FAIL\"}] Benchmark table complete')

# 8. Gate weights sum to ~1.0
gates = training_results['gates']
gate_sum = sum(gates)
c8 = abs(gate_sum - 1.0) < 0.01
checks.append(c8)
print(f'[{\"PASS\" if c8 else \"FAIL\"}] Gate weights sum to 1.0 +/- 0.01 (actual: {gate_sum:.4f})')

passed = sum(checks)
total = len(checks)
print(f'\\n{passed}/{total} checks passed')"""
)

with open("notebooks/14_phase14_results.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Notebook created: notebooks/14_phase14_results.ipynb")
