RAM EXPLOSION INVESTIGATION
===========================

TL;DR: reliability_guard() was called without memory limit.
       Generated code using permutations(12) allocated 68GB per process.
       Fix: reliability_guard(maximum_memory_bytes=1*1024**3)

FILES IN THIS FOLDER:
---------------------
main.txt                          - Full problem description
q21_analysis.txt                  - Analysis of Q21 "Stone XOR" problem
memory_bomb_examples.txt          - Examples of problematic generated code

testing_util_ORIGINAL_SNIPPET.py  - The buggy code (line 437)
testing_util_FIXED.py             - Fixed version
utils_execute_FIXED.py            - Also fixed for consistency

crash_log_excerpt.txt             - System monitor log showing crash
stable_log_excerpt.txt            - System monitor log after fix
eval0_detailed_excerpt.txt        - Per-evaluation memory tracking

cpu_evaluator.py                  - CPU evaluator with detailed logging
compute_code_generation_metrics.py - Evaluation orchestration
run_benchmark_optimal.py          - Main benchmark orchestrator

THE BUG (testing_util.py:437):
------------------------------
BEFORE: reliability_guard()           # No memory limit!
AFTER:  reliability_guard(maximum_memory_bytes=1*1024**3)

ROOT CAUSE:
-----------
Q21 has test cases with N=12 elements.
LLM generates code like: for perm in permutations(A)
12! = 479,001,600 permutations × 152 bytes = 68 GB per process
Multiple processes × 68 GB = 200-400 GB = OOM crash

EVIDENCE:
---------
Before fix: Evaluators 0,9,12 used 220GB, 120GB, 100GB respectively
After fix:  All evaluators stay at ~450MB each, system RAM stable at 45GB
