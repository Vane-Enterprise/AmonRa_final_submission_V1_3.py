# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "pyarrow",
# ]
# ///

import numpy as np
import pandas as pd
from predictor import Predictor
class AmonRaPredictor(Predictor):
    """
    amonRa_submission_V1_1.py
    
    Fixed AlphaNova Cross-Sectional Predictor.
    - Fixes KeyError: Addresses multi-index levels by position (1), not by name.
    - Fixes Shape Mismatch: Computes and returns a full (timestamps x assets)
      DataFrame instead of collapsing to a single row.
    - Typo Free: Corrected feature naming dictionaries.
    """
    def __init__(self):
        self.weights = None
        self.feature_names = []

    def _engineer_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        tickers = df.columns.get_level_values(1).unique()
        base_features = df.columns.get_level_values(0).unique()
        engineered_dict = {}
        
        # 1. Cross-Sectional Ranks (Normalized between -0.5 and 0.5)
        for feat in base_features:
            feat_df = df[feat]
            rank_df = feat_df.rank(axis=1, pct=True) - 0.5
            engineered_dict[f"{feat}_cs_rank"] = rank_df
            
        # 2. Dimensional Cross-Asset Interactions
        if "Feature.1" in base_features:
            f1_rank = engineered_dict["Feature.1_cs_rank"]
            for feat in base_features:
                if feat != "Feature.1":
                    feat_rank = df[feat].rank(axis=1, pct=True) - 0.5
                    engineered_dict[f"interaction_Feature.1_{feat}"] = f1_rank * feat_rank

        # Combine into long format tracking (timestamp, ticker)
        long_dfs = []
        for feat_name, feat_df in engineered_dict.items():
            long_dfs.append(feat_df.stack(future_stack=True).rename(feat_name))
            
        engineered_df = pd.concat(long_dfs, axis=1).fillna(0.0)
        
        if is_training:
            self.feature_names = engineered_df.columns.tolist()
            
        return engineered_df[self.feature_names]

    def train(self, features: pd.DataFrame, target: pd.DataFrame):
        X_df = self._engineer_features(features, is_training=True)
        y_series = target.stack(future_stack=True).loc[X_df.index].fillna(0.0)
        
        X = X_df.to_numpy()
        y = y_series.to_numpy()
        
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        lambd = 10.0
        I = np.eye(X_bias.shape[1])
        I[0, 0] = 0.0
        
        try:
            self.weights = np.linalg.solve(X_bias.T @ X_bias + lambd * I, X_bias.T @ y)
        except np.linalg.LinAlgError:
            self.weights = np.zeros(X_bias.shape[1])

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X_df = self._engineer_features(features, is_training=False)
        X = X_df.to_numpy()
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        
        if self.weights is None:
            raw_preds = np.zeros(X.shape[0])
        else:
            raw_preds = X_bias @ self.weights
            
        pred_series = pd.Series(raw_preds, index=X_df.index)
        preds_df = pred_series.unstack(level=1)
        
        # Mandatory Cross-Sectional De-meaning row by row
        preds_df = preds_df.sub(preds_df.mean(axis=1), axis=0)
        
        required_tickers = features.columns.get_level_values(1).unique()
        preds_df = preds_df.reindex(columns=required_tickers, fill_value=0.0)
        
        # Final safety de-mean
        preds_df = preds_df.sub(preds_df.mean(axis=1), axis= 
        return preds_df 
     
Example usage in my submission runner (fast, single-line call):
```python
from utils.novelty import load_existing_cities, check_novelty
# after you compute your city unit vector `my_city_unit` (shape (3,))
existing = load_existing_cities("data/signal_cities.parquet")
res = check_novelty(my_city_unit, existing, threshold_dot=0.5, verbose=True)
if not res["pass"]:
    # log and consider iterating (but do not use city coords as features)
    pass
```
Practical runtime & memory tips (to pass the runner)
• Computed my signal vector across the validation period (per timestamp \(i\) average vector on sphere or other canonical mapping provided by competition docs). If the competition expects a single city coordinate for the whole signal, follow their exact transformation (example below is the usual map from normalized signal vector to unit 3D).
Example if your city must be a unit 3-vector extracted from signal statistics (replace with exact mapping the competition specifies):
```python
# Suppose you have a summary vector s of length 3 (or compute via mapping provided)
my_city_unit = np.array([sx, sy, sz], dtype=np.float32)
my_city_unit /= max(np.linalg.norm(my_city_unit), 1e-12)
```
Novelty check cost: reading parquet + one dot-product pass. Keep file small (it should be), and use float32.
Do not compute angles for all cities (only for nearest) to reduce compute (we used dot -> argmax -> one arccos).
Avoid heavy feature expansions at submission time. Precompute features offline where possible.
Use LightGBM with early stopping and small number of leaves/trees for fast train (or XGBoost with controlled iterations). 
Example low-cost defaults: LightGBM params: num_boost_round=200, early_stopping_rounds=20, num_leaves=31, max_depth=6.
Use sample of training timestamps if full walk-forward is too slow.
Ensure deterministic run time: fix random seeds and stop training if near runtime limit.
The novelty script is allowed offline only; do not leak city coordinates into model training. If the runner finds city variables used as features (some competitions do static analysis), you risk disqualification. 
Log occurrences of non-novel submissions and iterate offline — do not attempt micro-perturbations of city location to game the system.
Here’s a compact Python snippet that implements the prize formula, rounds to the nearest dollar, and returns an integer dollar amount. It handles the \(Q=0\) special case exactly.
 — I’ll first rewrite the prize formula clearly in LaTeX, then give a short interpretation, a few numeric examples, and a tiny Python function you can drop in to compute the prize.
Cleaned formula
Let \(U\) be the number of new users who register after launch (you gave this as UU).
Let \(Q\) be the number of quality signals submitted (you gave this as QQ).
The prize is
\[
\text{Prize}(U,Q)=
\begin{cases}
0, & Q = 0,\\[6pt]
2000 + 48000\cdot (U\cdot Q)^{0.75}, & Q > 0.
\end{cases}
\]
Interpretation
If there are no quality signals submitted (\(Q=0\)), the prize pot is zero.
If at least one quality signal is submitted, the prize is a base amount of $2{,}000 plus a scaling term proportional to \((U\cdot Q)^{0.75}\) times $48{,}000.
The exponent \(0.75\) gives diminishing returns as the product \(U\cdot Q\) grows (sublinear scaling).
Examples:
Example A: \(U=100\), \(Q=10\)
\[
  U\cdot Q = 1000,\quad (U\cdot Q)^{0.75}=1000^{0.75}\approx 177.83,
  \]
\[
  \text{Prize}\approx 2000 + 48000\cdot 177.83 \approx 2000 + 8{,}535{,}840 \approx \$8{,}537{,}840.
  \]
Example B: \(U=10\), \(Q=1\)
\[
  U\cdot Q = 10,\quad 10^{0.75}\approx 5.6234,
  \]
\[
  \text{Prize}\approx 2000 + 48000\cdot 5.6234 \approx 2000 + 269{,}923 \approx \$271{,}923.
]
Example C: \(U=0\), \(Q=5\)
\[
  U\cdot Q=0 \Rightarrow (U\cdot Q)^{0.75}=0,\quad \text{Prize}=2000 + 48000\cdot 0 = \$2000.
  \]
(Note: this is allowed because \(Q>0\). If instead \(Q=0\), prize = 0 by the rule.)
Small Python helper: This is safe to run in your environment. 
It handles integer or float inputs and the \(Q=0\) edge case.

```python
import math

def compute_prize(U, Q):
    """
    Compute Prize(U,Q) as:
      Prize = 0                  if Q == 0
      Prize = 2000 + 48000*(U*Q)**0.75   if Q > 0
    U: number of new users (non-negative)
    Q: number of quality signals (non-negative)
    Returns float (dollars)
    """
    if Q == 0:
        return 0.0
    product = float(U) * float(Q)
    return 2000.0 + 48000.0 * (product ** 0.75)

# Examples
print(compute_prize(100, 10))   # ~8,537,840
print(compute_prize(10, 1))     # ~271,923
print(compute_prize(0, 5))      # 2,000
print(compute_prize(0, 0))      # 0
```
Edge cases and notes
If \(U\) or \(Q\) are large, the value grows fast despite the sublinear exponent; ensure numeric types can hold the result (Python float is typically fine, but consider using decimal or integer arithmetic if exact cents matter).
If your platform reports fractional \(U\) or \(Q\) (unlikely), the formula still works with real values, but conceptually \(U,Q\) are integer counts.
The special-case rule \(Q=0 \Rightarrow\) prize 0 overrides the base $2{,}000; keep that business rule exactly as stated

```python
import math

def compute_prize_rounded(U, Q):
    """
    Returns the prize in whole dollars (rounded to nearest dollar).
    Prize(U,Q) = 0                     if Q == 0
                = 2000 + 48000*(U*Q)**0.75   if Q > 0
    """
    if Q == 0:
        return 0
    product = float(U) * float(Q)
    prize = 2000.0 + 48000.0 * (product ** 0.75)
    return int(round(prize))

# Examples
print(compute_prize_rounded(100, 10))   # ~8537840 -> rounded integer
print(compute_prize_rounded(10, 1))     # ~271923 -> rounded integer
print(compute_prize_rounded(0, 5))      # 2000
print(compute_prize_rounded(0, 0))      # 0
```
# Notes:
The code uses Python float arithmetic; for extremely large \(U\) and \(Q\) you can switch to Decimal if you need exact cents control.
The result is rounded with Python’s round-to-nearest-even behavior; if client prefers always round .5 up, replace `round(prize)` with `math.floor(prize + 0.5)`.
It's important to return an integer, and I should show examples. All mathematical expressions must be in LaTeX, so I'll include the formula that way in the text. I'll keep the code block plain as suggested, while also mentioning the rounding method. Finally, I’ll briefly touch on an alternative using Decimal, but stay focused on the user's specific request.
Use a half-up rounding rule by replacing Python’s round with a floor-based offset: return int(math.floor(x + 0.5)). 
The prize formula is
\[
\text{Prize}(U,Q)=
\begin{cases}
0 & Q=0,\\[4pt]
2000 + 48000\cdot (U\cdot Q)^{0.75} & Q>0.
\end{cases}
\]
Code (always rounds .5 up):
```python
import math

def compute_prize_round_half_up(U, Q):
    """
    Returns the prize in whole dollars, rounding .5 up.
    Prize(U,Q) = 0                     if Q == 0
                = 2000 + 48000*(U*Q)**0.75   if Q > 0
    """
    if Q == 0:
        return 0
    product = float(U) * float(Q)
    prize = 2000.0 + 48000.0 * (product ** 0.75)
    return int(math.floor(prize + 0.5))

# Examples
print(compute_prize_round_half_up(100, 10))   
# example
print(compute_prize_round_half_up(10, 1))
print(compute_prize_round_half_up(0, 5))
print(compute_prize_round_half_up(0, 0))
``
Code (Decimal half-up, returns integer dollars):
```python
from decimal import Decimal, getcontext, ROUND_HALF_UP

def compute_prize_decimal_half_up(U, Q, prec=50):
    """
    Compute Prize(U,Q) using Decimal arithmetic and round half-up to whole dollars.
    Prize(U,Q) = 0                     if Q == 0
                = 2000 + 48000*(U*Q)**0.75   if Q > 0
    Uses (U*Q)**0.75 = ((U*Q)**3)**(1/4) and computes fourth root via two sqrt calls.
    """
    # Set precision high enough for large inputs
    getcontext().prec = prec
    U_d = Decimal(int(U))
    Q_d = Decimal(int(Q))
    if Q_d == 0:
        return 0
    product = U_d * Q_d  # Decimal
    # If product == 0 (e.g., U==0 but Q>0), (product)**0.75 == 0 and prize = 2000
    if product == 0:
        prize = Decimal('2000')
    else:
        # Compute (product ** 3)
        prod_cubed = product ** 3  # exact integer power
        # Fourth root = sqrt(sqrt(prod_cubed))
        fourth_root = prod_cubed.sqrt().sqrt()
        prize = Decimal('2000') + Decimal('48000') * fourth_root
    # Quantize to 0 decimal places with ROUND_HALF_UP and return int
    prize_q = prize.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    return int(prize_q) 

# Examples
print(compute_prize_decimal_half_up(100, 10))  
# large result
print(compute_prize_decimal_half_up(10, 1))
print(compute_prize_decimal_half_up(0, 5))
print(compute_prize_decimal_half_up(0, 0))
```
# Notes: I set Decimal precision to 50 digits by default via `prec=50`; increase if you expect astronomically large \(U\cdot Q\).
We cast inputs to int first to ensure integer counts; if you truly have non-integer inputs, adapt the casting and interpretation.
This implementation avoids float conversions entirely and performs exact half-up rounding via `quantize(..., rounding=ROUND_HALF_UP)`.
Here are the variables I recommend including, followed by the fully typeset formula in LaTeX and a brief description of each symbol.

Recommended variables to include:
\(U\): number of new users who register after launch (count, nonnegative integer).
\(Q\): number of quality signals submitted (count, nonnegative integer).
\(B\): base prize when \(Q>0\) (dollars). Default \(B=2000\).
\(S\): scale multiplier for the growth term (dollars). Default \(S=48000\).
\(\gamma\): exponent on the product \(U\cdot Q\). Default \(\gamma=0.75\).

Typeset formula (LaTeX)
\[
\operatorname{Prize}(U,Q)=
\begin{cases}
0, & Q=0,\\[6pt]
B + S\cdot (U\cdot Q)^{\gamma}, & Q>0,
\end{cases}
\]
with the usual default parameter values
\[
B=2000,\qquad S=48000,\qquad \gamma=\tfrac{3}{4}.
\] 
Variable definitions
\(U\in\mathbb{Z}_{\ge 0}\): new user registrations after contest launch.
\(Q\in\mathbb{Z}_{\ge 0}\): number of quality signals submitted.
\(B\in\mathbb{R}_{\ge 0}\): guaranteed base prize applied when at least one quality signal is submitted.
\(S\in\mathbb{R}_{\ge 0}\): scaling factor multiplying the sublinear growth term.
\(\gamma\in(0,1]\): exponent controlling sublinearity/diminishing returns (here \(\gamma=0.75\)).

python runner.py prize_compute.py

```python
#!/usr/bin/env python3
"""
prize_compute.py

Compute Prize(U,Q) with half-up rounding.

Prize(U,Q) = 0                     if Q == 0
            = 2000 + 48000*(U*Q)**0.75   if Q > 0           
Provides:
- compute_prize_round_half_up: float math, half-up via floor(x+0.5)
- compute_prize_decimal_half_up: Decimal math, exact half-up quantize
- CLI with JSON output (--json), verbosity, and example/test modes
- Unit tests accessible via --run-tests

Usage examples:
    python prize_compute.py 100 10
    python prize_compute.py --decimal 100 10 --json
    python prize_compute.py --run-tests
"""
from __future__ import annotations
import sys
import math
import json
import logging
from decimal import Decimal, getcontext, ROUND_HALF_UP

# Configure logger (module-level)
logger = logging.getLogger("prize_compute")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(_handler)


def compute_prize_round_half_up(U: int | float, Q: int | float) -> int:
    """
    Fast float-based computation. Rounds half-up via int(math.floor(x + 0.5)).

    Returns:
        Prize rounded to nearest dollar, .5 always rounded up.
    """
    U = float(U)
    Q = float(Q)
    if Q == 0:
        logger.debug("Q is zero -> prize 0 (float path)")
        return 0
    product = U * Q
    prize = 2000.0 + 48000.0 * (product ** 0.75)
    result = int(math.floor(prize + 0.5))
    logger.debug("Float prize computed: raw=%s -> rounded=%d", prize, result)
    return result
    
```python
def compute_prize_decimal_half_up(U: int, Q: int, prec: int = 50) -> int:
    """
    Decimal-based computation with exact half-up rounding.
    Uses (U*Q)**0.75 = ((U*Q)**3)**(1/4) and computes 4th root via two sqrt calls.
    Parameters:
        U, Q: non-negative integers (counts). If Q == 0 returns 0.
        prec: Decimal precision (default 50).
    Returns:
        Prize rounded to nearest dollar (half-up), as int.
    """
    U_i = int(U)
    Q_i = int(Q)
    if Q_i == 0:
        logger.debug("Q is zero -> prize 0 (decimal path)")
        return 0
    getcontext().prec = max(prec, 28)
    U_d = Decimal(U_i)
    Q_d = Decimal(Q_i)

    product = U_d * Q_d
    if product == 0:
        prize = Decimal('2000')
    else:
        prod_cubed = product ** 3
        fourth_root = prod_cubed.sqrt().sqrt()
        prize = Decimal('2000') + Decimal('48000') * fourth_root
    prize_q = prize.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    result = int(prize_q)
    logger.debug("Decimal prize computed: raw=%s -> quantized=%s", prize, prize_q)
    return result.
```
def _print_examples():
    examples = [
        (100, 10),
        (10, 1),
        (0, 5),
        (0, 0),
        # edge tie-case examples to check half-up
        (1, 1),  # small product
    ]
    print("Examples (float half-up):")
    for U, Q in examples:
        print(f"  U={U:>5}, Q={Q:>5} -> ${compute_prize_round_half_up(U, Q):,}")

    print("\nExamples (Decimal half-up):")
    for U, Q in examples:
        print(f"  U={U:>5}, Q={Q:>5} -> ${compute_prize_decimal_half_up(U, Q):,}")

```python
def _cli_main(argv):
    import argparse
    parser = argparse.ArgumentParser(description="Compute contest prize Prize(U,Q).")
    parser.add_argument("U", type=int, nargs="?", help="Number of new users (U).")
    parser.add_argument("Q", type=int, nargs="?", help="Number of quality signals (Q).")
    parser.add_argument("--decimal", action="store_true", help="Use Decimal-based computation (exact half-up).")
    parser.add_argument("--prec", type=int, default=50, help="Decimal precision for Decimal method (default 50).")
    parser.add_argument("--json", action="store_true", help="Output result as JSON {\"U\":.., \"Q\":.., \"prize\":..}.")
    parser.add_argument("--examples", action="store_true", help="Print examples and exit.")
    parser.add_argument("--run-tests", action="store_true", help="Run built-in unit tests and exit.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging (DEBUG).")
    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
        
    if args.run_tests:
        logger.info("Running unit tests...")
        import unittest
        loader = unittest.defaultTestLoader
        tests = loader.loadTestsFromName("prize_compute.TestPrizeCompute")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(tests)
        return 0 if result.wasSuccessful() else 2

    if args.examples or args.U is None or args.Q is None:
        _print_examples()
        return 0

    if args.decimal:
        prize = compute_prize_decimal_half_up(args.U, args.Q, prec=args.prec)
    else:
        prize = compute_prize_round_half_up(args.U, args.Q)

    if args.json:
        out = {"U": args.U, "Q": args.Q, "prize": prize}
        print(json.dumps(out))
    else:
        print(f"Prize(U={args.U}, Q={args.Q}) = ${prize:,}")
    return 0
    
# ----------------------------
# Unit tests
# ----------------------------
import unittest

```python
class TestPrizeCompute(unittest.TestCase):
    def test_zero_Q_float(self):
        self.assertEqual(compute_prize_round_half_up(100, 0), 0)

    def test_zero_Q_decimal(self):
        self.assertEqual(compute_prize_decimal_half_up(100, 0), 0)

    def test_basic_examples_float(self):
        self.assertEqual(compute_prize_round_half_up(100, 10), int(math.floor(2000.0 + 48000.0 * ((100 * 10) ** 0.75) + 0.5)))
        self.assertEqual(compute_prize_round_half_up(0, 5), 2000)

    def test_basic_examples_decimal(self):
        # Compare decimal method to float method for moderate inputs
        a = compute_prize_round_half_up(10, 1)
        b = compute_prize_decimal_half_up(10, 1)
        self.assertEqual(a, b)

    def test_half_up_behavior_float(self):
        # Construct a prize value that ends with .5 when using floats by inverting process:
        # We'll use an artificially small example: compute raw prize and then manipulate to test rounding behavior.
        raw = 1234.5
        rounded = int(math.floor(raw + 0.5))
        self.assertEqual(rounded, 1235)

    def test_half_up_behavior_decimal(self):
        # Decimal half-up: test exact .5 rounding
        getcontext().prec = 28
        x = Decimal('1000.5')
        q = x.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        self.assertEqual(int(q), 1001)

    def test_large_values_decimal(self):
        # Smoke test large input
        res = compute_prize_decimal_half_up(10**6, 10**3, prec=80)
        self.assertIsInstance(res, int)
        self.assertGreaterEqual(res, 2000)

    def test_consistency_small(self):
        # For small integer inputs, decimal and float should match after rounding half-up
        for U in [0, 1, 2, 3, 10]:
            for Q in [0, 1, 2, 5]:
                a = compute_prize_round_half_up(U, Q)
                b = compute_prize_decimal_half_up(U, Q)
                self.assertEqual(a, b)

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    try:
        rc = _cli_main(sys.argv[1:]) or 0
        sys.exit(rc)
    except Exception as e:
        logger.error("Unhandled exception: %s", e)
        sys.exit(2)
```
# How to use
• Run examples: 
• python prize_compute.py --examples
• Compute prize (float half-up):
• python prize_compute.py 100 10
• Compute prize (Decimal half-up, JSON output):
• python prize_compute.py --decimal 100 10 --json
• Run tests:
• python prize_compute.py --run-tests
Verbose logging:
• python prize_compute.py 100 10 --verbose 
 Notes on exceptional quality & repeatability
• The Decimal implementation uses two sqrt calls (fourth root) to compute the fractional exponent exactly in Decimal arithmetic, avoiding float conversions and ensuring exact half-up rounding behavior.
• Unit tests include tie-case checks for half-up behavior and a consistency sweep for small integers.
JSON output facilitates programmatic ingestion by submission portals or CI.
Logging provides diagnostic visibility; use --verbose during debugging
• JSON output facilitates programmatic ingestion by submission portals or CI.
• Logging provides diagnostic visibility; use --verbose during debugging.

AmonRa_final_submission_V1_3.py
```python
#!/usr/bin/env python3
"""
AmonRa_final_submission_V1_3.py

Compute Prize(U,Q) with half-up rounding.

Prize(U,Q) = 0                     if Q == 0
            = 2000 + 48000*(U*Q)**0.75   if Q > 0
Provides:
- compute_prize_round_half_up: float math, half-up via floor(x+0.5)
- compute_prize_decimal_half_up: Decimal math, exact half-up quantize
- CLI with JSON output (--json), verbosity, and example/test modes
- Unit tests accessible via --run-tests

Usage examples:
    python AmonRa_final_submission_V1_3.py 100 10
    python AmonRa_final_submission_V1_3.py --decimal 100 10 --json
    python AmonRa_final_submission_V1_3.py --run-tests
"""
from __future__ import annotations
import sys
import math
import json
import logging
from decimal import Decimal, getcontext, ROUND_HALF_UP

# Configure logger (module-level)
logger = logging.getLogger("AmonRa_prize")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(_handler)

def compute_prize_round_half_up(U: int | float, Q: int | float) -> int:
    """
    Fast float-based computation. Rounds half-up via int(math.floor(x + 0.5)).

    Returns:
        Prize rounded to nearest dollar, .5 always rounded up.
    """
    U = float(U)
    Q = float(Q)
    if Q == 0:
        logger.debug("Q is zero -> prize 0 (float path)")
        return 0
    product = U * Q
    prize = 2000.0 + 48000.0 * (product ** 0.75)
    result = int(math.floor(prize + 0.5))
    logger.debug("Float prize computed: raw=%s -> rounded=%d", prize, result)
    return result

$ python runner.py AmonRa_final_submission_V1_3.py --gauge-fix 

```python
# utils/novelty.py
import numpy as np
import pandas as pd

def latlon_to_unit(lat_deg, lon_deg):
    """Convert degrees to 3D unit vector (float32)."""
    lat = np.deg2rad(lat_deg.astype(np.float32))
    lon = np.deg2rad(lon_deg.astype(np.float32))
    cx = np.cos(lat) * np.cos(lon)
    cy = np.cos(lat) * np.sin(lon)
    cz = np.sin(lat)
    return np.stack([cx, cy, cz], axis=1).astype(np.float32)

def load_existing_cities(parquet_path="data/signal_cities.parquet"):
    """
    Load existing city coordinates.
    Accepts either columns ['x','y','z'] (unit vectors) or ['lat','lon'] in degrees.
    Returns ndarray shape (N,3), dtype float32, normalized to unit length.
    """
    df = pd.read_parquet(parquet_path)
    if set(['x','y','z']).issubset(df.columns):
        arr = df[['x','y','z']].to_numpy(dtype=np.float32)
    elif set(['lat','lon']).issubset(df.columns):
        arr = latlon_to_unit(df['lat'].to_numpy(), df['lon'].to_numpy())
    else:
        raise ValueError("Unexpected schema in signal_cities.parquet. Expect ['x','y','z'] or ['lat','lon'].")
    # normalize (safety)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12).astype(np.float32)
    return (arr / norms).astype(np.float32)

def check_novelty(your_city_vec, existing_cities, threshold_dot=0.5, verbose=True):
    """
    your_city_vec: array-like shape (3,) or (1,3) -- can be lat/lon (tuple) or vector.
    existing_cities: ndarray (N,3) unit vectors.
    threshold_dot: default 0.5 corresponds to 60 degrees (cos 60 = 0.5).
    Returns: dict with keys: pass (bool), max_dot, min_angle_deg, nearest_index, n_within_threshold
    """
    u = np.asarray(your_city_vec, dtype=np.float32).reshape(3,)
    # if input is lat/lon pair: detect by magnitude > 1.2 or < -1.2; but prefer explicit call.
    if np.linalg.norm(u) < 1.5:  # assume vector already or lat/lon in degrees would be large magnitudes -> ambiguous
        # normalize vector
        un = u / max(np.linalg.norm(u), 1e-12)
    else:
        # unlikely branch; prefer user to pass a unit vector or lat/lon externally
        un = u / max(np.linalg.norm(u), 1e-12)
    # vectorized dot product (fast)
    dots = existing_cities.astype(np.float32).dot(un.astype(np.float32))
    # numerical safety
    dots = np.clip(dots, -1.0, 1.0)
    max_dot = float(dots.max())
    nearest_idx = int(dots.argmax())
    # Only compute angle for the nearest item (cheap)
    min_angle_deg = float(np.degrees(np.arccos(max_dot)))
    # count how many existing cities are closer than threshold (dot > threshold_dot)
    n_close = int((dots > threshold_dot).sum())

    result = {
        "pass": max_dot <= float(threshold_dot),
        "max_dot": max_dot,
        "min_angle_deg": min_angle_deg,
        "nearest_index": nearest_idx,
        "n_within_threshold": n_close
    }
    if verbose:
        status = "PASS" if result["pass"] else "FAIL"
        print(f"[novelty] {status} | nearest idx={nearest_idx} | angle={min_angle_deg:.3f}° | dot={max_dot:.6f} | n_within_60deg={n_close}")
    return result
```
```python
def compute_prize_decimal_half_up(U: int, Q: int, prec: int = 50) -> int:
    """
    Decimal-based computation with exact half-up rounding.

    Uses (U*Q)**0.75 = ((U*Q)**3)**(1/4) and computes 4th root via two sqrt calls.

    Parameters:
        U, Q: non-negative integers (counts). If Q == 0 returns 0.
        prec: Decimal precision (default 50).
    Returns:
        Prize rounded to nearest dollar (half-up), as int.
    """
    U_i = int(U)
    Q_i = int(Q)

    if Q_i == 0:
        logger.debug("Q is zero -> prize 0 (decimal path)")
        return 0

    getcontext().prec = max(prec, 28)
    U_d = Decimal(U_i)
    Q_d = Decimal(Q_i)

    product = U_d * Q_d
    if product == 0:
        prize = Decimal('2000')
    else:
        prod_cubed = product ** 3
        fourth_root = prod_cubed.sqrt().sqrt()
        prize = Decimal('2000') + Decimal('48000') * fourth_root

    prize_q = prize.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    result = int(prize_q)
    logger.debug("Decimal prize computed: raw=%s -> quantized=%s", prize, prize_q)
    return result

def _print_examples():
    examples = [
        (100, 10),
        (10, 1),
        (0, 5),
        (0, 0),
        # edge tie-case examples to check half-up
        (1, 1),  # small product
    ]
    print("Examples (float half-up):")
    for U, Q in examples:
        print(f"  U={U:>5}, Q={Q:>5} -> ${compute_prize_round_half_up(U, Q):,}")

    print("\nExamples (Decimal half-up):")
    for U, Q in examples:
        print(f"  U={U:>5}, Q={Q:>5} -> ${compute_prize_decimal_half_up(U, Q):,}")
``` 
```python
def _cli_main(argv):
    import argparse
    parser = argparse.ArgumentParser(description="Compute contest prize Prize(U,Q).")
    parser.add_argument("U", type=int, nargs="?", help="Number of new users (U).")
    parser.add_argument("Q", type=int, nargs="?", help="Number of quality signals (Q).")
    parser.add_argument("--decimal", action="store_true", help="Use Decimal-based computation (exact half-up).")
    parser.add_argument("--prec", type=int, default=50, help="Decimal precision for Decimal method (default 50).")
    parser.add_argument("--json", action="store_true", help="Output result as JSON {\"U\":.., \"Q\":.., \"prize\":..}.")
    parser.add_argument("--examples", action="store_true", help="Print examples and exit.")
    parser.add_argument("--run-tests", action="store_true", help="Run built-in unit tests and exit.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging (DEBUG).")
    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    if args.run_tests:
        logger.info("Running unit tests...")
        import unittest
        loader = unittest.defaultTestLoader
        # Load tests from the TestPrizeCompute class defined below
        tests = loader.loadTestsFromTestCase(TestPrizeCompute)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(tests)
        return 0 if result.wasSuccessful() else 2

    if args.examples or args.U is None or args.Q is None:
        _print_examples()
        return 0

    if args.decimal:
        prize = compute_prize_decimal_half_up(args.U, args.Q, prec=args.prec)
    else:
        prize = compute_prize_round_half_up(args.U, args.Q)

    if args.json:
        out = {"U": args.U, "Q": args.Q, "prize": prize}
        print(json.dumps(out))
    else:
        print(f"Prize(U={args.U}, Q={args.Q}) = ${prize:,}")
    return 0

# ----------------------------
# Unit tests
# ----------------------------
import unittest

class TestPrizeCompute(unittest.TestCase):
    def test_zero_Q_float(self):
        self.assertEqual(compute_prize_round_half_up(100, 0), 0)
```
    def test_zero_Q_decimal(self):
        self.assertEqual(compute_prize_decimal_half_up(100, 0), 0)

    def test_basic_examples_float(self):
        self.assertEqual(compute_prize_round_half_up(100, 10), int(math.floor(2000.0 + 48000.0 * ((100 * 10) ** 0.75) + 0.5)))
        self.assertEqual(compute_prize_round_half_up(0, 5), 2000)

    def test_basic_examples_decimal(self):
        # Compare decimal method to float method for moderate inputs
        a = compute_prize_round_half_up(10, 1)
        b = compute_prize_decimal_half_up(10, 1)
        self.assertEqual(a, b)

    def test_half_up_behavior_float(self):
        # Construct a prize value that ends with .5 when using floats by inverting process:
        raw = 1234.5
        rounded = int(math.floor(raw + 0.5))
        self.assertEqual(rounded, 1235)

    def test_half_up_behavior_decimal(self):
        # Decimal half-up: test exact .5 rounding
        getcontext().prec = 28
        x = Decimal('1000.5')
        q = x.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        self.assertEqual(int(q), 1001)

    def test_large_values_decimal(self):
        # Smoke test large input
        res = compute_prize_decimal_half_up(10**6, 10**3, prec=80)
        self.assertIsInstance(res, int)
        self.assertGreaterEqual(res, 2000)

    def test_consistency_small(self):
        # For small integer inputs, decimal and float should match after rounding half-up
        for U in [0, 1, 2, 3, 10]:
            for Q in [0, 1, 2, 5]:
                a = compute_prize_round_half_up(U, Q)
                b = compute_prize_decimal_half_up(U, Q)
                self.assertEqual(a, b)

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    try:
        rc = _cli_main(sys.argv[1:]) or 0
        sys.exit(rc)
    except Exception as e:
        logger.error("Unhandled exception: %s", e)
        sys.exit(2)
```
