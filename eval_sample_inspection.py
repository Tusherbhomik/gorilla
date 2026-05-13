"""
BFCL Evaluation Pipeline — Step-by-Step Sample
================================================
Shows exactly how BFCL evaluates a model response for parallel_0.

Test:    parallel_0
User:    "Play songs from Taylor Swift and Maroon 5 for 20 and 15 minutes."
Correct: [spotify.play(artist="Taylor Swift", duration=20),
           spotify.play(artist="Maroon 5", duration=15)]

Run:  python eval_sample_inspection.py
"""

import ast
import json
import re
from pathlib import Path

BFCL_ROOT = Path(__file__).parent / "berkeley-function-call-leaderboard"
DATA_DIR  = BFCL_ROOT / "bfcl_eval/data"

SEP = "=" * 80

def hr(label=""):
    pad = (80 - len(label) - 2) // 2
    print("=" * pad + f" {label} " + "=" * pad)

# ── Step 0: Load the test entry and ground truth ─────────────────────────────

def load_entry(test_id):
    with open(DATA_DIR / "BFCL_v4_parallel.json") as f:
        for line in f:
            obj = json.loads(line)
            if obj["id"] == test_id:
                return obj

def load_ground_truth(test_id):
    with open(DATA_DIR / "possible_answer/BFCL_v4_parallel.json") as f:
        for line in f:
            obj = json.loads(line)
            if obj["id"] == test_id:
                return obj["ground_truth"]

# ── Step 1: decode_ast — parse raw model output string → list[dict] ──────────
# Source: bfcl_eval/model_handler/utils.py  default_decode_ast_prompting → ast_parse

def resolve_ast_call(elem):
    """Walk the Python AST node and produce {func_name: {param: value}}."""
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))

    args_dict = {}
    for kw in elem.keywords:
        val = kw.value
        if isinstance(val, ast.Constant):
            args_dict[kw.arg] = val.value
        elif isinstance(val, ast.List):
            args_dict[kw.arg] = [e.value for e in val.elts]
        else:
            args_dict[kw.arg] = ast.literal_eval(val)
    return {func_name: args_dict}

def decode_ast(raw_output: str) -> list[dict]:
    """
    Mirrors: default_decode_ast_prompting → ast_parse (python branch)
    Strips backticks/newlines, wraps in [], then uses Python's ast module.
    """
    s = raw_output.strip("`\n ")
    if not s.startswith("["): s = "[" + s
    if not s.endswith("]"):   s = s + "]"
    parsed = ast.parse(s, mode="eval")
    result = []
    if isinstance(parsed.body, ast.Call):
        result.append(resolve_ast_call(parsed.body))
    else:
        for elem in parsed.body.elts:
            result.append(resolve_ast_call(elem))
    return result

# ── Step 2: simple_function_checker — validate one call against ground truth ─
# Source: bfcl_eval/eval_checker/ast_eval/ast_checker.py  simple_function_checker

def simple_function_checker(func_description, model_call, possible_answer):
    """
    func_description: one tool definition dict (name, parameters)
    model_call:       one decoded call dict  e.g. {"spotify.play": {"artist": "Taylor Swift", "duration": 20}}
    possible_answer:  one ground truth dict  e.g. {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}}
    """
    func_name   = func_description["name"]
    param_defs  = func_description["parameters"]["properties"]
    required    = func_description["parameters"].get("required", [])
    gt_params   = list(possible_answer.values())[0]   # {"artist": ["Taylor Swift"], ...}

    errors = []

    # 1. Function name check
    if func_name not in model_call:
        return False, [f"Wrong function name. Expected '{func_name}', got {list(model_call.keys())}"]

    model_params = model_call[func_name]

    # 2. Required params present
    for p in required:
        if p not in model_params:
            return False, [f"Missing required param: '{p}'"]

    # 3. No unexpected params
    for p in model_params:
        if p not in param_defs:
            return False, [f"Unexpected param: '{p}'"]

    # 4. Value check — ground truth is a LIST of acceptable values; "" means optional/any
    for param, value in model_params.items():
        if param not in gt_params:
            return False, [f"Param '{param}' not in ground truth"]
        acceptable = gt_params[param]
        # "" in the list means the param is optional and any value is fine
        if "" in acceptable:
            continue
        if value not in acceptable:
            return False, [f"Wrong value for '{param}': got {repr(value)}, expected one of {acceptable}"]

    return True, []

# ── Step 3: parallel_function_checker_no_order ────────────────────────────────
# Source: bfcl_eval/eval_checker/ast_eval/ast_checker.py  parallel_function_checker_no_order
# Key insight: ORDER DOESN'T MATTER — it tries each ground truth against
# remaining unmatched model outputs.

def parallel_checker(func_descriptions, model_output, possible_answers):
    if len(model_output) != len(possible_answers):
        return False, [f"Wrong call count: got {len(model_output)}, expected {len(possible_answers)}"]

    matched = []
    for i, gt_call in enumerate(possible_answers):
        func_name_expected = list(gt_call.keys())[0]
        func_desc = next(f for f in func_descriptions if f["name"] == func_name_expected)

        found = False
        for j, model_call in enumerate(model_output):
            if j in matched:
                continue
            ok, errs = simple_function_checker(func_desc, model_call, gt_call)
            if ok:
                matched.append(j)
                found = True
                break

        if not found:
            return False, [f"No match found for ground truth call {i}: {gt_call}"]

    return True, []

# ── Main demo ─────────────────────────────────────────────────────────────────

def run_demo(model_output_raw: str, label: str):
    entry        = load_entry("parallel_0")
    ground_truth = load_ground_truth("parallel_0")
    tools        = entry["function"]

    print()
    hr(f"SCENARIO: {label}")
    print(f"\n  Raw model output: {repr(model_output_raw)}")

    # ── STEP 1: decode_ast ────────────────────────────────────────────────────
    print()
    hr("STEP 1 — decode_ast  (raw string → list of dicts)")
    print("  Source: model_handler/utils.py  default_decode_ast_prompting → ast_parse")
    print()
    print("  What it does:")
    print("    1. Strip backticks, newlines, spaces")
    print("    2. Ensure output is wrapped in  [ ... ]")
    print("    3. Run Python's  ast.parse()  on the string")
    print("    4. Walk each ast.Call node → extract func name + kwargs")
    print()

    try:
        decoded = decode_ast(model_output_raw)
        print(f"  Decoded output  →  {json.dumps(decoded, indent=4)}")
    except Exception as e:
        print(f"  DECODE FAILED: {e}")
        print("  → Result: INVALID (error_type: ast_decoder:decoder_failed)")
        return

    # ── STEP 2: route to checker ──────────────────────────────────────────────
    print()
    hr("STEP 2 — ast_checker  routes to parallel_function_checker_no_order")
    print("  Source: eval_checker/ast_eval/ast_checker.py  ast_checker()")
    print()
    print('  if "parallel" in test_category:')
    print('      → parallel_function_checker_no_order()')
    print('  elif "multiple" in test_category:')
    print('      → multiple_function_checker()')
    print('  else:')
    print('      → simple_function_checker()')
    print()
    print(f"  test_category = 'parallel'  →  parallel_function_checker_no_order()")

    # ── STEP 3: parallel checker ──────────────────────────────────────────────
    print()
    hr("STEP 3 — parallel_function_checker_no_order")
    print("  Source: eval_checker/ast_eval/ast_checker.py")
    print()
    print(f"  Ground truth (possible_answers):")
    for i, gt in enumerate(ground_truth):
        print(f"    [{i}] {json.dumps(gt)}")
    print()
    print(f"  Model decoded output:")
    for i, call in enumerate(decoded):
        print(f"    [{i}] {json.dumps(call)}")
    print()
    print("  Logic: ORDER-INSENSITIVE matching.")
    print("  For each ground truth call, try every unmatched model call via simple_function_checker.")
    print()

    matched = []
    overall_valid = True
    for i, gt_call in enumerate(ground_truth):
        func_name_expected = list(gt_call.keys())[0]
        func_desc = next(f for f in tools if f["name"] == func_name_expected)
        print(f"  ── Matching ground truth [{i}]: {func_name_expected}(...) ──")
        print(f"     acceptable values: {list(gt_call.values())[0]}")
        found = False
        for j, model_call in enumerate(decoded):
            if j in matched:
                continue
            ok, errs = simple_function_checker(func_desc, model_call, gt_call)
            print(f"     try model_output[{j}] = {model_call}  →  {'MATCH ✓' if ok else f'no match ({errs})'}")
            if ok:
                matched.append(j)
                found = True
                break
        if not found:
            overall_valid = False
            print(f"     FAILED: no match for ground truth [{i}]")

    # ── STEP 4: result ────────────────────────────────────────────────────────
    print()
    hr("STEP 4 — Final result")
    if overall_valid:
        print("  {\"valid\": true}")
        print()
        print("  → This entry is counted as CORRECT in the score CSV.")
    else:
        print("  {\"valid\": false, \"error\": [...]}")
        print()
        print("  → This entry is counted as WRONG. Error detail saved to score file.")


if __name__ == "__main__":
    print()
    print("BFCL Evaluation Pipeline Walkthrough")
    print("Test case: parallel_0")
    print("User:      'Play songs from Taylor Swift and Maroon 5 for 20 and 15 minutes.'")
    print("Tools:     [spotify.play(artist, duration)]")
    print()
    print("Ground truth (from possible_answer/BFCL_v4_parallel.json):")
    gt = load_ground_truth("parallel_0")
    for call in gt:
        fname = list(call.keys())[0]
        params = list(call.values())[0]
        print(f"  {fname}({', '.join(f'{k}={v}' for k, v in params.items())})")
    print("  Note: each value is a LIST of acceptable answers. \"\" means the param is optional.")

    # ── Case 1: Perfect answer ────────────────────────────────────────────────
    run_demo(
        '[spotify.play(artist="Taylor Swift", duration=20), spotify.play(artist="Maroon 5", duration=15)]',
        "CORRECT — exact match, correct order"
    )

    # ── Case 2: Reversed order (still valid) ─────────────────────────────────
    run_demo(
        '[spotify.play(artist="Maroon 5", duration=15), spotify.play(artist="Taylor Swift", duration=20)]',
        "CORRECT — reversed order (parallel checker is order-insensitive)"
    )

    # ── Case 3: Wrong duration ────────────────────────────────────────────────
    run_demo(
        '[spotify.play(artist="Taylor Swift", duration=20), spotify.play(artist="Maroon 5", duration=20)]',
        "WRONG — Maroon 5 duration should be 15, not 20"
    )

    # ── Case 4: Only one call instead of two ─────────────────────────────────
    run_demo(
        '[spotify.play(artist="Taylor Swift", duration=20)]',
        "WRONG — missing second call (wrong count)"
    )

    # ── Case 5: Bad output format ─────────────────────────────────────────────
    run_demo(
        'I will play Taylor Swift for 20 minutes.',
        "WRONG — not a function call format at all"
    )
