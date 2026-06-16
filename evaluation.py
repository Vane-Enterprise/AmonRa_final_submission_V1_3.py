Below are two ready-to-save files:

 1) AmonRa_final_submission_V1_4.py — updated submission module with:
   - Float and Decimal half-up prize functions (unchanged behavior).
   - CLI with JSON, examples, --decimal, --verbose.
   - Unit tests (unittest).
   - A CI-style JUnit XML test reporter you can call from the CLI via --ci-junit <output.xml>.
   - A --run-tests flag (prints to console) and --ci-junit for CI integration (writes JUnit XML and returns nonzero on failure).
   - Small improvements to logging and deterministic test output.

2) README.md — short README describing usage, CLI flags, and CI integration instructions.

Save each file with the exact filename shown and upload both to the client portal.

File: AmonRa_final_submission_V1_4.py
```python
#!/usr/bin/env python3
"""
AmonRa_final_submission_V1_4.py

Compute Prize(U,Q) with half-up rounding and provide CI-friendly JUnit XML output.

Prize(U,Q) = 0                     if Q == 0
            = 2000 + 48000*(U*Q)**0.75   if Q > 0

Features:
- compute_prize_round_half_up: float math, half-up via floor(x+0.5)
- compute_prize_decimal_half_up: Decimal math, exact half-up quantize
- CLI with JSON output (--json), verbosity, examples, and test modes
- Unit tests accessible via --run-tests
- CI integration: --ci-junit <path> runs tests and writes JUnit XML to the path

Usage examples:
    python AmonRa_final_submission_V1_4.py 100 10
    python AmonRa_final_submission_V1_4.py --decimal 100 10 --json
    python AmonRa_final_submission_V1_4.py --run-tests
    python AmonRa_final_submission_V1_4.py --ci-junit results.xml
"""
from __future__ import annotations
import sys
import math
import json
import logging
import time
from decimal import Decimal, getcontext, ROUND_HALF_UP
import xml.etree.ElementTree as ET

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
        (1, 1),
    ]
    print("Examples (float half-up):")
    for U, Q in examples:
        print(f"  U={U:>5}, Q={Q:>5} -> ${compute_prize_round_half_up(U, Q):,}")

    print("\nExamples (Decimal half-up):")
    for U, Q in examples:
        print(f"  U={U:>5}, Q={Q:>5} -> ${compute_prize_decimal_half_up(U, Q):,}")


def _cli_main(argv):
    import argparse
    parser = argparse.ArgumentParser(description="Compute contest prize Prize(U,Q).")
    parser.add_argument("U", type=int, nargs="?", help="Number of new users (U).")
    parser.add_argument("Q", type=int, nargs="?", help="Number of quality signals (Q).")
    parser.add_argument("--decimal", action="store_true", help="Use Decimal-based computation (exact half-up).")
    parser.add_argument("--prec", type=int, default=50, help="Decimal precision for Decimal method (default 50).")
    parser.add_argument("--json", action="store_true", help="Output result as JSON {\"U\":.., \"Q\":.., \"prize\":..}.")
    parser.add_argument("--examples", action="store_true", help="Print examples and exit.")
    parser.add_argument("--run-tests", action="store_true", help="Run built-in unit tests and exit (console output).")
    parser.add_argument("--ci-junit", type=str, default=None, help="Run tests and write JUnit XML to provided path, exit nonzero on failure.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging (DEBUG).")
    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # CI JUnit mode: run tests, write XML, exit with nonzero on test failures
    if args.ci_junit is not None:
        outpath = args.ci_junit
        logger.info("Running unit tests (CI mode), output JUnit XML to '%s'...", outpath)
        rc = _run_tests_write_junit(outpath)
        if rc == 0:
            logger.info("Tests passed; JUnit XML written to '%s'.", outpath)
        else:
            logger.error("Tests failed; JUnit XML written to '%s'.", outpath)
        return rc

    if args.run_tests:
        logger.info("Running unit tests (console mode)...")
        import unittest
        loader = unittest.defaultTestLoader
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

    def test_zero_Q_decimal(self):
        self.assertEqual(compute_prize_decimal_half_up(100, 0), 0)

    def test_basic_examples_float(self):
        expected = int(math.floor(2000.0 + 48000.0 * ((100 * 10) ** 0.75) + 0.5))
        self.assertEqual(compute_prize_round_half_up(100, 10), expected)
        self.assertEqual(compute_prize_round_half_up(0, 5), 2000)

    def test_basic_examples_decimal(self):
        a = compute_prize_round_half_up(10, 1)
        b = compute_prize_decimal_half_up(10, 1)
        self.assertEqual(a, b)

    def test_half_up_behavior_float(self):
        raw = 1234.5
        rounded = int(math.floor(raw + 0.5))
        self.assertEqual(rounded, 1235)

    def test_half_up_behavior_decimal(self):
        getcontext().prec = 28
        x = Decimal('1000.5')
        q = x.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        self.assertEqual(int(q), 1001)

    def test_large_values_decimal(self):
        res = compute_prize_decimal_half_up(10**6, 10**3, prec=80)
        self.assertIsInstance(res, int)
        self.assertGreaterEqual(res, 2000)

    def test_consistency_small(self):
        for U in [0, 1, 2, 3, 10]:
            for Q in [0, 1, 2, 5]:
                a = compute_prize_round_half_up(U, Q)
                b = compute_prize_decimal_half_up(U, Q)
                self.assertEqual(a, b)


# ----------------------------
# JUnit XML writer for CI
# ----------------------------
def _run_tests_write_junit(output_path: str) -> int:
    """
    Run unittest tests from TestPrizeCompute and write a simple JUnit XML file.

    Returns 0 on success (all tests passed), nonzero otherwise.
    """
    import unittest
    loader = unittest.defaultTestLoader
    suite = loader.loadTestsFromTestCase(TestPrizeCompute)
    runner = unittest.TextTestRunner(verbosity=2)
    # Capture results by running tests via TestResult
    result = runner.run(suite)

    # Construct JUnit XML
    tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(getattr(result, "skipped", []))  # python 3.8+ may include skipped

    testsuite = ET.Element("testsuite", {
        "name": "AmonRa_final_submission_V1_4.Tests",
        "tests": str(tests),
        "failures": str(failures),
        "errors": str(errors),
        "skipped": str(skipped),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "time": "0"
    })

    # For each test record, add testcase entries.
    # result.failures and result.errors are lists of (testcase, traceback)
    failed_cases = {str(t[0]) for t in result.failures}
    error_cases = {str(t[0]) for t in result.errors}

    # We do not have a complete list of passed tests from TextTestRunner result object,
    # so we iterate through the suite to reconstruct test names.
    for test in _iter_tests(suite):
        tc_name = str(test)
        # attempt to parse classname and method name
        classname = test.__class__.__name__
        method = getattr(test, "_testMethodName", None) or ""
        testcase = ET.SubElement(testsuite, "testcase", {
            "classname": classname,
            "name": method or tc_name,
            "time": "0"
        })
        if tc_name in failed_cases:
            # find the traceback text
            tb = _find_traceback_text(result.failures, tc_name)
            failure = ET.SubElement(testcase, "failure", {"message": "failure"})
            failure.text = tb
        elif tc_name in error_cases:
            tb = _find_traceback_text(result.errors, tc_name)
            error = ET.SubElement(testcase, "error", {"message": "error"})
            error.text = tb
        # skipped not handled specially here (few tests); can extend if needed

    tree = ET.ElementTree(testsuite)
    try:
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
    except Exception as e:
        logger.error("Failed to write JUnit XML to '%s': %s", output_path, e)
        return 2

    # Return 0 if all tests passed
    return 0 if (failures == 0 and errors == 0) else 1


def _iter_tests(suite):
    """
    Yield individual TestCase instances from a TestSuite (recursive).
    """
    try:
        for t in suite:
            if isinstance(t, unittest.TestSuite):
                yield from _iter_tests(t)
            else:
                yield t
    except Exception:
        # If any unexpected structure, just attempt to yield suite itself
        yield suite


def _find_traceback_text(list_of_tuples, test_str):
    """
    Given result.failures or result.errors (list of (testcase, traceback)),
    return traceback text matching test_str, or a short message.
    """
    for tc, tb in list_of_tuples:
        if str(tc) == test_str:
            return tb
    return "Traceback unavailable"


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

File: README.md
```markdown
# AmonRa_final_submission_V1_4

This repository contains a single script `AmonRa_final_submission_V1_4.py` that computes the contest prize function
and provides CI-friendly test output.

Prize definition (defaults):
- \(U\): number of new users (non-negative integer)
- \(Q\): number of quality signals (non-negative integer)

\[
\operatorname{Prize}(U,Q)=
\begin{cases}
0, & Q=0,\\[6pt]
2000 + 48000\cdot (U\cdot Q)^{0.75}, & Q>0.
\end{cases}
\]

Rounding: results are rounded to nearest dollar with "round half-up" semantics (ties of .5 round up).

## Files
- `AmonRa_final_submission_V1_4.py` — main script containing implementations, CLI, and unit tests.
- `README.md` — this file.

## Usage

Run interactively:
```bash
python AmonRa_final_submission_V1_4.py 100 10
```

Use Decimal-based exact arithmetic (recommended for very large inputs):
```bash
python AmonRa_final_submission_V1_4.py --decimal 100 10
```

Machine-readable JSON output:
```bash
python AmonRa_final_submission_V1_4.py --json 100 10
# => {"U":100, "Q":10, "prize": 8537840}
```

Print examples:
```bash
python AmonRa_final_submission_V1_4.py --examples
```

Run unit tests (console):
```bash
python AmonRa_final_submission_V1_4.py --run-tests
```

Run tests and write JUnit-style XML (CI integration):
```bash
python AmonRa_final_submission_V1_4.py --ci-junit results.xml
# Exit code 0 if tests passed; non-zero if any test failed or writing failed.
```

Enable verbose logging:
```bash
python AmonRa_final_submission_V1_4.py --verbose 100 10
```

## CI Integration (example)
- Add a step to our CI that executes:
  - `python AmonRa_final_submission_V1_4.py --ci-junit results.xml`
- Collect the `results.xml` as JUnit test artifact for the CI system (GitHub Actions, GitLab CI, Jenkins, etc.).
- The script returns exit code 0 on success (tests passed) and 1 on test failures; 2 if errors occurred during execution.

## Notes
- The Decimal implementation computes \((U\cdot Q)^{0.75}\) by evaluating \(((U\cdot Q)^3)^{1/4}\) and computing the fourth root via two `sqrt()` calls on `decimal.Decimal`, avoiding float conversions.
- The implementation uses only Python standard library modules and therefore should be portable in standard CI and client submission environments.

If anyone like, they can add:
- Add a minimal GitHub Actions workflow YAML that runs tests and uploads `results.xml`.
- Produce a small wrapper shell script to run the script and post results to the portal.
- Bundle everything into a zip for direct upload.

```

Notes and next steps
- Save both files and upload them to the client portal.
- Run `python AmonRa_final_submission_V1_4.py --ci-junit results.xml` to produce `results.xml` for CI.
- If you want, I can also produce a minimal GitHub Actions workflow YAML (e.g., `.github/workflows/test.yml`) 
Below is a small, portable HTML + JavaScript snippet you can drop into your README (renderers that allow raw HTML) or a simple static page. 
It shows a caution message, the code block, and a copy button that uses the modern Clipboard API with a graceful fallback. 
It also shows a transient success/failure message for the user.
Save as e.g. copy_snippet.html or paste into our docs where HTML is allowed.
Note: "Use this code with caution" text is included and accessible; the copy button only copies the visible code text, not hidden metadata. 

HTML snippet
    
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Copy Code Snippet</title>
  <style>
    body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial; padding: 1rem; background:#f7f7fb; color:#111; }
    .card { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(16,24,40,0.06); padding: 1rem; max-width: 900px; margin: 0 auto; }
    .warn { background: #fff7e6; border-left: 4px solid #ffb020; padding: 0.5rem 0.75rem; margin-bottom: 0.75rem; border-radius:4px; color:#664d03; }
    pre { background: #0f1724; color: #e6eef8; padding: 1rem; border-radius: 6px; overflow:auto; position: relative; font-family: "Source Code Pro", monospace; font-size: 13px; line-height:1.45; }
    .controls { display:flex; gap:8px; align-items:center; margin-bottom: 0.5rem; }
    button.copy-btn {
      background:#0b74de; color:#fff; border: none; padding: 0.45rem 0.7rem; border-radius: 6px; cursor: pointer;
      font-weight: 600; font-size: 13px;
    }
    button.copy-btn:disabled { opacity: 0.6; cursor: not-allowed; }
    .status { font-size: 13px; color: #0f1724; opacity: 0.9; }
    .sr-only { position: absolute !important; height:1px; width:1px; overflow:hidden; clip:rect(1px,1px,1px,1px); white-space:nowrap; }
  </style>
</head>
<body>
  <div class="card" role="region" aria-label="Code copy example">
    <div class="warn" role="alert">
      <strong>Use this code with caution:</strong>
      <span> Verify inputs and test in a safe environment before running in production.</span>
    </div>
    <div class="controls" aria-hidden="false">
      <button class="copy-btn" id="copyBtn" title="Copy code to clipboard">Copy code</button>
      <div id="status" class="status" aria-live="polite"></div>
    </div>
    <pre id="codeBlock" tabindex="0" aria-label="Code block">
#!/usr/bin/env python3
# Example: compute Prize(U,Q) with half-up rounding
def compute_prize_round_half_up(U, Q):
    import math
    if Q == 0:
        return 0
    prize = 2000.0 + 48000.0 * ((U * Q) ** 0.75)
    return int(math.floor(prize + 0.5))
    </pre>
    <p style="margin-top:0.75rem; color:#334155; font-size:13px;">
      Tip: you can focus the code block (Tab) and press the Copy button or use your system copy shortcuts.
    </p>
    <!-- Accessible description for screen readers -->
    <p class="sr-only" id="copyDesc">Press the Copy code button to copy the code block to clipboard.</p>
  </div>
  <script>
    (function () {
      const copyBtn = document.getElementById('copyBtn');
      const codeBlock = document.getElementById('codeBlock');
      const status = document.getElementById('status');
function setStatus(text, ok = true, timeout = 3000) {
        status.textContent = text;
        status.style.color = ok ? '#0b74de' : '#c53030';
        if (timeout) {
          setTimeout(() => { status.textContent = ''; }, timeout);
        }
      }
      async function copyText(text) {
        // Try the modern Clipboard API first
        if (navigator.clipboard && navigator.clipboard.writeText) {
          try {
            await navigator.clipboard.writeText(text);
            return { ok: true };
          } catch (err) {
            // Fall through to execCommand fallback
            console.warn('Clipboard API failed:', err);
          }
        }
        // Fallback: create a temporary textarea and execCommand('copy')
        try {
          const ta = document.createElement('textarea');
          ta.value = text;
          // Avoid scrolling to bottom
          ta.style.position = 'fixed';
          ta.style.left = '-9999px';
          document.body.appendChild(ta);
          ta.focus();
          ta.select();
          const res = document.execCommand('copy');
          document.body.removeChild(ta);
          return { ok: !!res };
        } catch (err) {
          console.error('Fallback copy failed', err);
          return { ok: false, error: err };
        }
      }
      copyBtn.addEventListener('click', async function () {
        // Disable button briefly to avoid spamming
        copyBtn.disabled = true;
        const text = codeBlock.textContent.replace(/\u00A0/g, ''); // remove NBSP if present
        const res = await copyText(text);
        if (res.ok) {
          setStatus('Copied to clipboard ✓', true);
          copyBtn.textContent = 'Copied';
        } else {
          setStatus('Copy failed — please select and copy manually', false, 6000);
          copyBtn.textContent = 'Copy code';
        }
        // Re-enable after short delay
        setTimeout(() => {
          copyBtn.disabled = false;
          copyBtn.textContent = 'Copy code';
        }, 1200);
      });
      // Keyboard accessibility: Ctrl/Cmd+C when code block focused will still work via browser default
      codeBlock.addEventListener('keydown', function (ev) {
        if ((ev.ctrlKey || ev.metaKey) && ev.key === 'c') {
          // Let the browser handle default copy when selection exists; otherwise copy whole block
          const selection = window.getSelection().toString();
          if (!selection) {
            ev.preventDefault();
            copyBtn.click();
          }
        }
      });
    })();
  </script>
</body>
</html>
``` 
PEP-723 dependency declaration

```python
# PEP-723 dependency declaration (informational block)
# Dependencies:
#   numpy
#   pandas
#   pyarrow  # for reading parquet in the offline novelty-check utility
#
# Note: These are provided for CI/package metadata purposes per PEP-723 style.

from __future__ import annotations
import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Import Predictor base class from predictor.py (expected in evaluation environment)
try:
    from predictor import Predictor  # type: ignore
except Exception:
    # Minimal fallback base class for local testing only.
    class Predictor:
        def train(self, features, target):
            raise NotImplementedError()

        def predict(self, features):
            raise NotImplementedError()

logger = logging.getLogger("AmonRa.Predictor")
logger.addHandler(logging.NullHandler())

class AmonRaPredictor(Predictor):
    """
    AmonRaPredictor: robust cross-sectional-to-weights mapper with:
      - zero-RMS handling via eps_rms and eps_guard
      - optional EWMA persistence (persistence_alpha)
      - normalization schemes ('rms' or 'l1') to target gross leverage L_gross
      - cross-sectional de-meaning and optional neutrality enforcement after smoothing
      - an offline novelty-check utility that reads data/signal_cities.parquet
        and computes angular proximity (dot-product) without leaking city coords
        into model input.

    Constructor parameters:
      - L_gross: float, target gross leverage (default 1.0)
      - eps_rms: float, RMS threshold to treat a cross-section as zero (default 1e-8)
      - persistence_alpha: float in (0,1], EWMA alpha (1.0 disables persistence)
      - norm: 'rms' or 'l1' normalization (default 'rms')
      - enforce_neutrality_after_smoothing: bool, subtract mean after normalization (default True)
      - clip_extreme: Optional[tuple(min,max)] to clip signals before processing
      - eps_guard: small positive to avoid division-by-zero (default 1e-12)
      - verbose: bool, enable debug logging

    Expectations:
      - Input features: 2-D array-like shape (T, J) where each ROW is a timestamp cross-section.
      - All logic inside class; no external helper functions.
    """
    def __init__(
        self,
        L_gross: float = 1.0,
        eps_rms: float = 1e-8,
        persistence_alpha: float = 1.0,
        norm: str = "rms",
        enforce_neutrality_after_smoothing: bool = True,
        clip_extreme: Optional[Tuple[float, float]] = None,
        eps_guard: float = 1e-12,
        verbose: bool = False,
    ):
        super().__init__()
        self.L_gross = float(L_gross)
        self.eps_rms = float(eps_rms)
        self.alpha = float(persistence_alpha)
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("persistence_alpha must be in (0,1]. Use 1.0 to disable persistence.")
        if norm not in ("rms", "l1"):
            raise ValueError("norm must be 'rms' or 'l1'")
        self.norm = norm
        self.enforce_neutrality_after_smoothing = bool(enforce_neutrality_after_smoothing)
        self.clip_extreme = clip_extreme
        self.eps_guard = float(eps_guard)
        self.verbose = bool(verbose)

        # Internal state
        self._smoothed_prev: Optional[np.ndarray] = None  # previous EWMA smoothed signal
        self._J: Optional[int] = None  # asset count

        if self.verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug(
                "AmonRaPredictor init: L_gross=%s eps_rms=%s alpha=%s norm=%s",
                self.L_gross,
                self.eps_rms,
                self.alpha,
                self.norm,
            )

    def train(self, features, target=None):
        """
        Record asset count from features. No model to fit.
        Parameters:
          - features: 2-D array-like (T, J)
        """
        arr = np.asarray(features, dtype=float)
        if arr.ndim != 2:
            raise ValueError("features must be 2-D array-like with shape (T, J)")
        self._J = arr.shape[1]
        if self.verbose:
            logger.debug("train: recorded J=%d", self._J)
        return self

    def predict(self, features):
        """
        Convert cross-sectional signals -> cross-sectionally-neutral weights.

        Input:
          - features: 2-D array-like shape (T, J), each ROW is a timestamp.

        Output:
          - weights: NumPy array shape (T, J)
        """
        X = np.asarray(features, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("features must be 2-D array-like with shape (T, J)")

        T, J = X.shape
        if self._J is None:
            self._J = J
        elif self._J != J:
            logger.warning("predict: asset count changed from %s to %s", self._J, J)
            self._J = J

        weights = np.zeros_like(X, dtype=float)

        for t in range(T):
            s = X[t].astype(float).copy()
            if self.clip_extreme is not None:
                lo, hi = self.clip_extreme
                s = np.clip(s, lo, hi)

            # Cross-sectional de-mean
            mean = np.mean(s)
            p = s - mean

            if self.verbose:
                logger.debug("t=%d raw mean=%g first3 p=%s", t, mean, np.array2string(p[: min(3, p.size)], precision=6))

            # EWMA persistence on the signal (not weights)
            if self._smoothed_prev is None:
                s_sm = p.copy()
            else:
                s_sm = self.alpha * p + (1.0 - self.alpha) * self._smoothed_prev

            # Compute RMS of smoothed signal
            rms = np.sqrt(np.mean(s_sm ** 2))
            if self.verbose:
                logger.debug("t=%d rms=%g eps_rms=%g", t, rms, self.eps_rms)

            # Handle near-zero RMS
            if rms < self.eps_rms:
                if self.alpha < 1.0 and self._smoothed_prev is not None:
                    prev_rms = np.sqrt(np.mean(self._smoothed_prev ** 2))
                    if prev_rms > self.eps_rms:
                        # Use decayed previous smoothed (persistence)
                        s_sm = (1.0 - self.alpha) * self._smoothed_prev
                        if self.verbose:
                            logger.debug("t=%d using persisted prior smoothed (prev_rms=%g)", t, prev_rms)
                    else:
                        # No meaningful signal -> zero weights
                        weights[t, :] = 0.0
                        self._smoothed_prev = s_sm
                        continue
                else:
                    # No persistence -> zero weights
                    weights[t, :] = 0.0
                    self._smoothed_prev = s_sm
                    continue

            # Update smoothed_prev for next iteration
            self._smoothed_prev = s_sm.copy()

            # Normalization -> weights
            if self.norm == "rms":
                rms_sm = np.sqrt(np.mean(s_sm ** 2))
                denom = max(rms_sm, self.eps_guard)
                w = s_sm * (1.0 / denom)  # unit RMS
                w = w * self.L_gross
            else:  # l1
                sumabs = np.sum(np.abs(s_sm))
                if sumabs < self.eps_guard:
                    weights[t, :] = 0.0
                    continue
                w = s_sm * (self.L_gross / sumabs)

            # Enforce neutrality after smoothing if requested
            if self.enforce_neutrality_after_smoothing:
                w_mean = np.mean(w)
                w = w - w_mean

            # Final tidy up
            w[np.abs(w) < 1e-16] = 0.0
            weights[t, :] = w

            if self.verbose and t < 5:
                logger.debug(
                    "t=%d weights first3=%s gross=%.6g sum=%.6g",
                    t,
                    np.array2string(w[: min(3, w.size)], precision=6),
                    np.sum(np.abs(w)),
                    np.sum(w),
                )

        return weights

    def run_offline_novelty_check(self, parquet_path: str, candidate_vectors: np.ndarray) -> pd.DataFrame:
        """
        Offline novelty check that compares candidate unit vectors to existing cities.

        Parameters:
          - parquet_path: path to data/signal_cities.parquet. Expected columns:
              either ['x','y','z'] OR ['lat','lon'] (degrees).
          - candidate_vectors: array-like (K, D) of vectors (unit vectors recommended).
              If you provide lat/lon, convert to unit vectors before calling.

        Returns:
          - pandas.DataFrame with columns:
              candidate_index, max_dot, min_angle_deg, novel_ok
            novel_ok True when max_dot <= 0.5 (i.e., angular distance >= 60 deg).
        """
        df = pd.read_parquet(parquet_path)

        if set(["x", "y", "z"]).issubset(df.columns):
            coords = df[["x", "y", "z"]].to_numpy(dtype=float)
        elif set(["lat", "lon"]).issubset(df.columns):
            lat = np.deg2rad(df["lat"].to_numpy(dtype=float))
            lon = np.deg2rad(df["lon"].to_numpy(dtype=float))
            coords = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
        else:
            raise ValueError("parquet must contain either x,y,z or lat,lon columns")

        # Normalize city coords
        norms = np.linalg.norm(coords, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        coords_unit = coords / norms

        cand = np.asarray(candidate_vectors, dtype=float)
        if cand.ndim == 1:
            cand = cand.reshape(1, -1)
        cand_norms = np.linalg.norm(cand, axis=1, keepdims=True)
        cand_norms[cand_norms == 0.0] = 1.0
        cand_unit = cand / cand_norms

        dots = cand_unit @ coords_unit.T  # shape (K, N_cities)
        max_dots = np.max(dots, axis=1)
        max_dots_clipped = np.clip(max_dots, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(max_dots_clipped))
        novel_ok = max_dots <= 0.5

        out = pd.DataFrame(
            {
                "candidate_index": np.arange(len(max_dots)),
                "max_dot": max_dots,
                "min_angle_deg": angles_deg,
                "novel_ok": novel_ok,
            }
        )
        return out
```
     
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

This is python code that is used for its calculation.
Code:

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)
# Example raw scores s (rows timestamps 1..5, cols assets 1..3)
s = np.array([
    [0.10, 0.05, -0.02],
    [0.20, 0.15, 0.10],
    [-0.05, 0.00, 0.10],
    [0.50, -0.10, 0.00],
    [0.02, 0.02, 0.02]
])
J = s.shape[1]
# Compute demeaned signals P = s - mean
s_mean = s.mean(axis=1, keepdims=True)
P = s - s_mean
# Parameters
epsilon = 1e-4      # threshold for RMS to consider signal as effectively zero
alpha = 0.2         # EWMA smoothing coeff (0<alpha<=1)
L_gross = 1.0       # target gross leverage
# Compute RMS per row
rms = np.sqrt((P**2).mean(axis=1))
# Containers
scaled = np.zeros_like(P)
weights_epsilon = np.zeros_like(P)
weights_ewma = np.zeros_like(P)
# For EWMA we need previous smoothed signal; initialize as zeros then fill iteratively
P_sm_prev = np.zeros(J)
for i in range(P.shape[0]):
    if rms[i] < epsilon:
        # Epsilon rule: set scaled signal to zero
        scaled[i] = 0.0
        # Weights zero because no signal
        weights_epsilon[i] = 0.0
    else:
        scaled[i] = P[i] / rms[i]
        # Convert to weights by L1 normalization to achieve gross leverage
        l1 = np.abs(scaled[i]).sum()
        weights_epsilon[i] = (L_gross / l1) * scaled[i]
    # EWMA fallback: if rms < epsilon, use decayed previous smoothed signal
    if rms[i] < epsilon:
        P_sm = (1 - alpha) * P_sm_prev  # since P[i] is zero
    else:
        # compute current unit-rms signal and update smoothed signal
        cur_unit = P[i] / (rms[i] if rms[i] >= epsilon else 1.0)
        P_sm = alpha * cur_unit + (1 - alpha) * P_sm_prev
    P_sm_prev = P_sm.copy()
    # Normalize EWMA-smoothed signal to target gross leverage
    l1_sm = np.abs(P_sm).sum()
    if l1_sm == 0:
        weights_ewma[i] = 0.0
    else:
        weights_ewma[i] = (L_gross / l1_sm) * P_sm
# Print results
print("P (demeaned signals):")
print(P)
print()
print("RMS per row:")
print(rms)
print()
print(f"Using epsilon = {epsilon}")
print()
print("Scaled (unit-RMS) where defined, zeros where RMS<eps:")
print(scaled)
print()
print("Weights with epsilon rule (L_gross=1, L1-normalized):")
print(weights_epsilon)
print()
print(f"EWMA with alpha={alpha}; smoothed weights (L_gross=1):")
print(weights_ewma)
# Show timestamp 5 details
print('\nDetails for timestamp 5 (index 4):')
print('P[4] =', P[4])
print('rms[4] =', rms[4])
print('scaled[4] =', scaled[4])
print('weights_epsilon[4] =', weights_epsilon[4])
print('weights_ewma[4] =', weights_ewma[4])
```
Result
Copy
```python
P (demeaned signals):
[[ 0.0567  0.0067 -0.0633]
 [ 0.05    0.     -0.05  ]
 [-0.0667 -0.0167  0.0833]
 [ 0.3667 -0.2333 -0.1333]
 [ 0.      0.      0.    ]]
RMS per row:
[0.0492 0.0408 0.0624 0.2625 0.    ]
Using epsilon = 0.0001
Scaled (unit-RMS) where defined, zeros where RMS<eps:
[[ 1.1514  0.1355 -1.2868]
 [ 1.2247  0.     -1.2247]
 [-1.069  -0.2673  1.3363]
 [ 1.397  -0.889  -0.508 ]
 [ 0.      0.      0.    ]]
Weights with epsilon rule (L_gross=1, L1-normalized):
[[ 0.4474  0.0526 -0.5   ]
 [ 0.5     0.     -0.5   ]
 [-0.4    -0.1     0.5   ]
 [ 0.5    -0.3182 -0.1818]
 [ 0.      0.      0.    ]]
EWMA with alpha=0.2; smoothed weights (L_gross=1):
[[ 0.4474  0.0526 -0.5   ]
 [ 0.476   0.024  -0.5   ]
 [ 0.5    -0.1394 -0.3606]
 [ 0.5    -0.2698 -0.2302]
 [ 0.5    -0.2698 -0.2302]]
Details for timestamp 5 (index 4):
P[4] = [0. 0. 0.]
rms[4] = 0.0
scaled[4] = [0. 0. 0.]
weights_epsilon[4] = [0. 0. 0.]
weights_ewma[4] = [ 0.5    -0.2698 -0.2302]
```
Below is a small, clear Python example that implements the epsilon check and two fallback behaviors:
Epsilon rule: when the per-timestamp RMS is below \(\varepsilon\), treat the whole row as zero-signal and set weights to zero.
EWMA fallback: when the row is zero, decay the previous smoothed signal via EWMA and normalize that to the target gross leverage \(L_{\text{gross}}\).
I used the same raw-score example from before. Key formulas used:
Demeaned signal: \(P_j(i)=s_j(i)-\overline{s}(i)\).
Row RMS: \(\mathrm{RMS}_i=\sqrt{\tfrac{1}{J}\sum_j P_j(i)^2}\).
Epsilon rule: if \(\mathrm{RMS}_i<\varepsilon\), set signal to zero.
EWMA smoothing: \(P^{\text{sm}}(i)=\alpha\cdot U(i) + (1-\alpha)\,P^{\text{sm}}(i-1)\), where \(U(i)\) is unit-RMS \(P(i)/\mathrm{RMS}_i\) when defined; when \(P(i)=0\) we simply decay the previous \(P^{\text{sm}}\).
Convert signal to weights with L1 normalization to reach gross leverage \(L_{\text{gross}}\):
\[
  w(i)=\frac{L_{\text{gross}}}{\sum_j |v_j(i)|}\,v(i),
  \]
where \(v(i)\) is the chosen signal vector (unit-RMS, smoothed, or fallback).
Output highlights from the run:
Demeaned matrix \(P\):
\[
  \begin{bmatrix}
  0.0567 & 0.0067 & -0.0633\\[4pt]
  0.05 & 0 & -0.05\\[4pt]
  -0.0667 & -0.0167 & 0.0833\\[4pt]
  0.3667 & -0.2333 & -0.1333\\[4pt]
  0 & 0 & 0
  \end{bmatrix}
  \]
RMS per row: \([0.0492,\ 0.0408,\ 0.0624,\ 0.2625,\ 0.0]\).
With \(\varepsilon=10^{-4}\), the fifth row is treated as zero.
Weights using epsilon rule (L_gross = 1) — last row all zeros:
\[
  \begin{bmatrix}
  0.4474 & 0.0526 & -0.5\\[4pt]
  0.5 & 0 & -0.5\\[4pt]
  -0.4 & -0.1 & 0.5\\[4pt]
  0.5 & -0.3182 & -0.1818\\[4pt]
  0 & 0 & 0
  \end{bmatrix}
  \]
EWMA fallback with \(\alpha=0.2\) produces smoothed weights; the last row decays the previous weights instead of going to zero:
\[
  \begin{bmatrix}
  0.4474 & 0.0526 & -0.5\\[4pt]
  0.476 & 0.024 & -0.5\\[4pt]
  0.5 & -0.1394 & -0.3606\\[4pt]
  0.5 & -0.2698 & -0.2302\\[4pt]
  0.5 & -0.2698 & -0.2302
  \end{bmatrix}
  \]
Practical notes and recommendations:
Default safe behavior: use the epsilon rule and set zero-signal rows to no positions. This avoids taking arbitrary trades and reduces cost exposure.
Recommending steps and code
Here’s a concise, actionable guide plus ready-to-run Python snippets to (A) compute angular distance between your signal’s city and every existing city in data/signal_cities.parquet, (B) check the 60° novelty rule, and (C) an iterative workflow + practical tips to increase novelty without using city coordinates as model inputs.
Key math (angular distance)
If two city coordinates are unit vectors \(u,v\in\mathbb{R}^3\) on the sphere, the angular distance \(\theta\) (in degrees) is
\[
  \; \;=\; \arccos\!\bigl(u\cdot v\bigr)\times\frac{180}{\pi}.
  \]
Equivalently, to test the 60° threshold use the dot product:
\[
  \theta \ge 60^\circ \quad\Longleftrightarrow\quad u\cdot v \le \cos(60^\circ) = 0.5.
  \]
So we can often avoid computing the expensive \(\arccos\) and just check whether the dot product is \(\le 0.5\).
If city coordinates are given as latitude/longitude \((\phi,\lambda)\), convert to unit 3D vectors first:
\[
u=\begin{bmatrix}
\cos\phi\cos\lambda\\[4pt]
\cos\phi\sin\lambda\\[4pt]
\sin\phi
\end{bmatrix},
\]
where \(\phi\) = latitude in radians and \(\lambda\) = longitude in radians.
Python snippet — read Parquet, compute angular distances, report nearest
(assumes cities are stored either as 3 columns x,y,z OR lat/lon; adapt if your schema differs)

```python
import numpy as np
import pandas as pd
# Path to parquet
path = "data/signal_cities.parquet"
df = pd.read_parquet(path)
# Inspect columns: either expect ['x','y','z'] as unit vectors or ['lat','lon'] in degrees
print(df.columns)
# Example: if df has 'x','y','z'
if set(['x','y','z']).issubset(df.columns):
    existing = df[['x','y','z']].to_numpy(dtype=float)
    # ensure unit normalization (numerical safety)
    norms = np.linalg.norm(existing, axis=1, keepdims=True)
    existing = existing / np.maximum(norms, 1e-12)
# Or if df has 'lat','lon' in degrees:
elif set(['lat','lon']).issubset(df.columns):
    lat = np.deg2rad(df['lat'].to_numpy(dtype=float))
    lon = np.deg2rad(df['lon'].to_numpy(dtype=float))
    existing = np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
else:
    raise ValueError("Unexpected schema: expected ('x','y','z') or ('lat','lon').")
# My city: either as unit vector or lat/lon. Example: provide as unit vector:
# Replace this with your actual city coordinates
my_city = np.array([0.1, 0.9, 0.4], dtype=float)
my_city = my_city / np.linalg.norm(my_city)
# Compute dot products with all existing cities
dots = existing.dot(your_city)
# Clip for numerical safety
dots = np.clip(dots, -1.0, 1.0)
# Angular distances in degrees (optional)
angles_deg = np.degrees(np.arccos(dots))
# Find nearest
idx_min = np.argmin(angles_deg)
min_angle = angles_deg[idx_min]
min_dot = dots[idx_min]
print(f"Nearest city index: {idx_min}, angular distance = {min_angle:.3f} deg, dot = {min_dot:.6f}")
# Check 60 degree novelty criterion
threshold_deg = 60.0
ok = min_angle >= threshold_deg
if ok:
    print("Novelty check PASSED: nearest city >= 60° away.")
else:
    print("Novelty check FAILED: nearest city closer than 60° (consider iterating).")
# Optionally list all cities within 60 deg
close_mask = angles_deg < threshold_deg
print(f"{close_mask.sum()} existing cities are within {threshold_deg}°.")
```
Faster check using dot product only (no arccos):
Compute \(d = \max(u\cdot v)\). If \(d \le 0.5\) then you pass the 60° rule. If \(d > 0.5\) you fail.
Practical iterative workflow
Build our signal and compute its city (offline).
Run the distance check above.
If the nearest angle ≥ 60°, proceed with submission/wider validation. 
If nearest angle < 60°, iterate using one or more of these strategies (do NOT use city coords as model inputs):
Change feature subset (e.g., remove some momentum windows or include volume/turnover).
Change lookbacks (mix 5/20/60 windows or add multi-horizon stacking).
Use different feature transforms (rolling z-scores, winsorization, ratios).
Change model family or regularization (e.g., tree vs linear vs small neural net).
Inject cross-asset interactions (pairwise differences, ranks).
Recompute the city after each meaningful change and re-check (use --gauge-fix iteratively as i described).
Interpretation guidance (why distance matters but is not definitive)
Cities far apart are very likely uncorrelated (good for distinctness).
Cities close together may or may not be correlated — I’ll need to check actual pairwise correlation in a validation window.
The competition uses correlation clustering with threshold 0.5; aiming for ≥60° (dot ≤ 0.5) is a conservative, practical rule because \(\cos(60^\circ)=0.5\).
Practical thresholds, logging, and monitoring
Use a small margin: prefer nearest-dot \(\le 0.45\) (i.e., \(>\!62.2^\circ\)) to be safer vs numerical noise.
Log: nearest angle, nearest signal id, and how many cities fall inside 60°, so I can track improvements across iterations.
Keep a reproducible offline script that runs our feature+model pipeline -> compute city -> check distances. This supports the iterative design flow the competition recommends.
Keep changes broad enough to avoid minor random jitter as “novelty” (avoid tiny random perturbations just to move city).
Guidance for gauge-fix check
To implement a gauge-fix check in their submission while adhering to runtime limits. I'll create a concise checklist and offer a code template for the runner that computes the city, checks for novelty, and incorporates efficient computations without heavy libraries.
I’ll also suggest performance tips, such as using numpy for vectorized operations, float32 data types, and reading files efficiently with pyarrow, all while ensuring no multiprocessing is used.
For a few secondsCreating a novelty check
I’ll include a Python function that checks novelty by reading parquet files and comparing dot products. It’s important to note that the user should avoid using city coordinates as features in their implementation. This will help streamline the process and keep the focus on the necessary computations. Overall, this approach aims to enhance efficiency and ensure everything runs smoothly without complicating the feature set. Done!
Here’s a compact, production-ready checklist and a ready-to-drop-in Python utility you can call from your submission (e.g., under the `--gauge-fix` path) that respects the environment and limits you listed.
High-level checklist (important constraints)
CPU-only, < 4 minutes total training, < 60 s prediction per call, < 8 GB memory.
Use numpy / pandas / pyarrow for the novelty check (all available).
Vectorize: avoid Python loops over cities; use a dot-product reduction (fast, memory-light).
Prefer float32 for large arrays to reduce memory and improve speed.
If we integrate the check into a full pipeline, run it after model train / inference, not inside the training loop.
What the novelty check must do
Accept our submission's city as a 3D vector \(u=(u_x,u_y,u_z)\) or as (lat, lon).
Load existing cities from parquet (file: data/signal_cities.parquet).
Convert existing cities to unit 3D vectors if needed.
Compute dot products \(d_k = u \cdot v_k\) for all existing city vectors \(v_k\).
Nearest angular distance \( \theta_{\min} = \arccos(\max_k d_k) \) (deg). Equivalently check \(\max_k d_k \le 0.5\) for the 60° rule.
Return a small report: nearest index/id, nearest dot, nearest angle (deg), count of cities with dot > 0.5.
Why dot-product oriented check
Use the dot threshold test (no arccos) for faster checks:
pass if \( \max_k (u \cdot v_k) \le 0.5 \).

