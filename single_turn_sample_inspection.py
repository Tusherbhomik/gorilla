"""
BFCL Single-Turn Inference Inspection
=======================================
Covers: non_live_multiple, non_live_parallel, live_multiple, live_parallel

Key difference from multi-turn:
  - Model receives ONE user message and must output ALL function calls in a SINGLE response.
  - There is NO tool-execution feedback loop.
  - The function definitions are stored INSIDE each test entry (not in a shared func_doc file).

Run:
  python single_turn_sample_inspection.py --category multiple          # non-live multiple
  python single_turn_sample_inspection.py --category parallel          # non-live parallel
  python single_turn_sample_inspection.py --category live_multiple     # live multiple
  python single_turn_sample_inspection.py --category live_parallel     # live parallel
  python single_turn_sample_inspection.py --category multiple --json   # raw JSON payload
"""

import json
import sys
from pathlib import Path

BFCL_ROOT = Path(__file__).parent / "berkeley-function-call-leaderboard"
DATA_DIR   = BFCL_ROOT / "bfcl_eval/data"

CATEGORY_MAP = {
    "multiple":      ("BFCL_v4_multiple.json",       "multiple_0"),
    "parallel":      ("BFCL_v4_parallel.json",        "parallel_0"),
    "live_multiple": ("BFCL_v4_live_multiple.json",   "live_multiple_0-0-0"),
    "live_parallel": ("BFCL_v4_live_parallel.json",   "live_parallel_0-0-0"),
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def load_entry(fname, test_id):
    for obj in load_jsonl(DATA_DIR / fname):
        if obj["id"] == test_id:
            return obj
    raise KeyError(test_id)

def load_ground_truth(fname, test_id):
    ans_path = DATA_DIR / "possible_answer" / fname
    for obj in load_jsonl(ans_path):
        if obj["id"] == test_id:
            return obj.get("ground_truth", [])
    return []

def build_system_prompt(tools: list[dict]) -> str:
    """
    Default system prompt format:
      ret_fmt=python, tool_call_tag=False, func_doc_fmt=json, prompt_fmt=plaintext, style=classic
    Source: bfcl_eval/constants/default_prompts.py  _DEFAULT_SYSTEM_PROMPT
    """
    output_format = "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]"
    functions_json = json.dumps(tools, indent=2)
    return (
        "You are an expert in composing functions. "
        "You are given a question and a set of possible functions. "
        "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
        "If none of the functions can be used, point it out. "
        "If the given question lacks the parameters required by the function, also point it out.\n\n"
        "You should only return the function calls in your response.\n\n"
        f"If you decide to invoke any of the function(s), you MUST put it in the format of "
        f"{output_format} "
        f"You SHOULD NOT include any other text in the response.\n\n"
        f"Here is a list of functions in JSON format that you can invoke.\n{functions_json}\n"
    )

SEP = "=" * 80

def hr(label=""):
    if label:
        pad = (80 - len(label) - 2) // 2
        print("=" * pad + f" {label} " + "=" * pad)
    else:
        print(SEP)

# ── human-readable display ────────────────────────────────────────────────────

def show_human_readable(category: str):
    fname, test_id = CATEGORY_MAP[category]
    entry        = load_entry(fname, test_id)
    tools        = entry["function"]          # tools live INSIDE the entry, not in a shared file
    ground_truth = load_ground_truth(fname, test_id)

    print()
    hr(f"CATEGORY: {category}   |   TEST ID: {test_id}")

    # ── overview ──────────────────────────────────────────────────────────────
    print()
    print(f"  Type         : SINGLE-TURN (model outputs all calls in ONE response)")
    print(f"  Tools count  : {len(tools)}")
    print(f"  Tool names   : {[t['name'] for t in tools]}")
    print()
    print("  ┌─ KEY DIFFERENCE vs multi-turn ─────────────────────────────────┐")
    if category in ("parallel", "live_parallel"):
        print("  │  PARALLEL: model must call the SAME function MULTIPLE TIMES    │")
        print("  │  simultaneously (e.g. weather for two cities at once).         │")
    else:
        print("  │  MULTIPLE: model must choose from MULTIPLE DIFFERENT functions  │")
        print("  │  and call whichever one(s) the question requires.              │")
    print("  │  No tool-execution feedback — model outputs everything at once. │")
    print("  └────────────────────────────────────────────────────────────────┘")

    # ── tool definitions ──────────────────────────────────────────────────────
    print()
    hr("TOOL DEFINITIONS  (embedded inside each test entry — not a shared file)")
    for tool in tools:
        req = tool["parameters"].get("required", [])
        props = tool["parameters"].get("properties", {})
        optional = [k for k in props if k not in req]
        print(f"\n  name        : {tool['name']}")
        print(f"  description : {tool['description'][:90]}")
        print(f"  required    : {req}")
        print(f"  optional    : {optional}")

    # ── system prompt ─────────────────────────────────────────────────────────
    system_prompt = build_system_prompt(tools)
    intro_end = system_prompt.index("Here is a list")
    print()
    hr("SYSTEM PROMPT")
    print(system_prompt[:intro_end])
    print("Here is a list of functions in JSON format that you can invoke.")
    print(f"  [ ... {len(tools)} tool definition(s) — shown above / see --json for full payload ... ]")

    # ── the single turn ───────────────────────────────────────────────────────
    print()
    hr("THE SINGLE TURN  (everything the LLM receives)")
    user_content = entry["question"][0][0]["content"]
    print(f"\n  [SYSTEM]  <system prompt with {len(tools)} tool(s)>")
    print(f"\n  [USER]    {user_content}")
    print()
    print("  ── model must respond with (ground truth):")
    for i, call in enumerate(ground_truth):
        print(f"     call {i+1}: {json.dumps(call)}")

    # ── full message list ─────────────────────────────────────────────────────
    print()
    hr("FULL MESSAGE LIST  (what goes into messages=[...])")
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_content},
    ]
    for i, m in enumerate(messages):
        preview = (m["content"][:75] + "...") if len(m["content"]) > 78 else m["content"]
        print(f"  [{i:02d}] role={m['role']:<10}  {preview}")
    print()
    print("  ← ONLY 2 messages. Model responds once. Done.")

    # ── field reference ───────────────────────────────────────────────────────
    print()
    hr("FIELD REFERENCE")
    rows = [
        ("id",         "Test entry identifier",                                  "NO"),
        ("question",   "Single user message (one turn)",                         "YES — USER message"),
        ("function",   "Tool definitions per entry (NOT a shared file)",         "YES — tool list"),
    ]
    print(f"  {'Field':<12} {'Meaning':<50} {'Sent to LLM?'}")
    print(f"  {'-'*12} {'-'*50} {'-'*12}")
    for field, meaning, sent in rows:
        print(f"  {field:<12} {meaning:<50} {sent}")
    print()


# ── raw JSON payload ──────────────────────────────────────────────────────────

def show_json_payload(category: str):
    fname, test_id = CATEGORY_MAP[category]
    entry = load_entry(fname, test_id)
    tools = entry["function"]
    system_prompt = build_system_prompt(tools)
    user_content  = entry["question"][0][0]["content"]

    payload = {
        "model": "<your-model>",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
    }
    print()
    print(f"====================== TURN 0  — JSON payload sent to LLM ======================")
    print(json.dumps(payload, indent=2))


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    use_json = "--json" in args
    args = [a for a in args if a != "--json"]

    category = "multiple"
    if "--category" in args:
        idx = args.index("--category")
        category = args[idx + 1]

    if category not in CATEGORY_MAP:
        print(f"Unknown category '{category}'. Choose from: {list(CATEGORY_MAP.keys())}")
        sys.exit(1)

    if use_json:
        show_json_payload(category)
    else:
        show_human_readable(category)
