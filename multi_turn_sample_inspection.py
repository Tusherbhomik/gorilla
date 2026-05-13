"""
BFCL Multi-Turn Inference Inspection
=====================================
Test case: multi_turn_base_1
Shows the EXACT input sent to the LLM at each turn.

Run:  python multi_turn_sample_inspection.py
      python multi_turn_sample_inspection.py --json   (raw API payload per turn)
"""

import json
import sys
from pathlib import Path

BFCL_ROOT = Path(__file__).parent / "berkeley-function-call-leaderboard"
FUNC_DOC_DIR = BFCL_ROOT / "bfcl_eval/data/multi_turn_func_doc"
TEST_DATA    = BFCL_ROOT / "bfcl_eval/data/BFCL_v4_multi_turn_base.json"
ANSWERS_FILE = BFCL_ROOT / "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def load_test_case(test_id):
    for obj in load_jsonl(TEST_DATA):
        if obj["id"] == test_id:
            return obj
    raise KeyError(f"{test_id} not found")

def load_ground_truth(test_id):
    for obj in load_jsonl(ANSWERS_FILE):
        if obj["id"] == test_id:
            return obj["ground_truth"]
    return None

def load_tools(involved_classes: list[str]) -> list[dict]:
    """Load all function docs for the given classes (same logic as bfcl_eval/utils.py:779-786)."""
    CLASS_TO_FILE = {
        "GorillaFileSystem": "gorilla_file_system.json",
        "TwitterAPI":        "posting_api.json",
        "TradingBot":        "trading_bot.json",
        "TravelAPI":         "travel_booking.json",
        "VehicleControlAPI": "vehicle_control.json",
        "MathAPI":           "math_api.json",
        "MessageAPI":        "message_api.json",
        "TicketAPI":         "ticket_api.json",
        "WebSearch":         "web_search.json",
        "MemoryAPI":         "memory_kv.json",
    }
    tools = []
    for cls in involved_classes:
        fname = CLASS_TO_FILE[cls]
        tools.extend(load_jsonl(FUNC_DOC_DIR / fname))
    return tools

def build_system_prompt(tools: list[dict]) -> str:
    """
    Reconstruct the system prompt exactly as bfcl_eval builds it.
    Source: bfcl_eval/constants/default_prompts.py  (DEFAULT_SYSTEM_PROMPT_FORMAT)
      ret_fmt=python, tool_call_tag=False, func_doc_fmt=json, prompt_fmt=plaintext, style=classic
    """
    output_format = "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]"
    functions_json = json.dumps(tools, indent=2)

    return (
        "You are an expert in composing functions. "
        "You are given a question and a set of possible functions. "
        "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
        "If none of the functions can be used, point it out. "
        "If the given question lacks the parameters required by the function, also point it out.\n\n"
        f"You should only return the function calls in your response.\n\n"
        f"If you decide to invoke any of the function(s), you MUST put it in the format of "
        f"{output_format} "
        f"You SHOULD NOT include any other text in the response.\n\n"
        "At each turn, you should try your best to complete the tasks requested by the user within "
        "the current turn. Continue to output functions to call until you have fulfilled the user's "
        "request to the best of your ability. Once you have no more functions to call, the system "
        "will consider the current turn complete and proceed to the next turn or task.\n\n"
        f"Here is a list of functions in JSON format that you can invoke.\n{functions_json}\n"
    )

# ── fake execution: what the virtual file system returns ─────────────────────
#
# In real BFCL, bfcl_eval/eval_checker/multi_turn_eval/func_source_code/gorilla_file_system.py
# is instantiated with initial_config and actually executes the calls.
# Below we hard-code the realistic return values so the walkthrough is self-contained.

FAKE_TOOL_RESULTS = {
    # Turn 0
    "ls(a=True)": {"current_directory_content": ["workspace"]},

    # Turn 1
    "cd(folder='workspace')": {"current_working_directory": "/alex/workspace"},
    "mv(source='log.txt',destination='archive')": {"result": "None"},

    # Turn 2
    "cd(folder='archive')": {"current_working_directory": "/alex/workspace/archive"},
    "grep(file_name='log.txt',pattern='Error')": {
        "matching_lines": ["Error: Something went wrong."]
    },

    # Turn 3
    "tail(file_name='log.txt',lines=20)": {
        "last_lines": (
            "This is a log file. No errors found. Another line. "
            "Yet another line. Error: Something went wrong. Final line."
        )
    },
}

# ── display ───────────────────────────────────────────────────────────────────

SEP  = "=" * 80
SEP2 = "-" * 80

def hr(label=""):
    if label:
        pad = (80 - len(label) - 2) // 2
        print("=" * pad + f" {label} " + "=" * pad)
    else:
        print(SEP)

def show_human_readable(test_case, tools, ground_truth):
    print()
    hr(f"TEST CASE: {test_case['id']}")

    # ── initial state (NOT sent to LLM — internal to BFCL) ───────────────────
    print()
    print("┌─ INITIAL CONFIG  (NEVER shown to LLM — used to init virtual filesystem) ─┐")
    print(json.dumps(test_case["initial_config"], indent=2))
    print("└──────────────────────────────────────────────────────────────────────────┘")

    # ── tools ─────────────────────────────────────────────────────────────────
    print()
    print(f"  involved_classes : {test_case['involved_classes']}")
    print(f"  excluded_function: {test_case.get('excluded_function', [])}  ← metadata only, NOT filtered from tool list")
    print(f"  total tools loaded: {len(tools)}")
    print(f"  tool names: {[t['name'] for t in tools]}")

    # ── system prompt ─────────────────────────────────────────────────────────
    system_prompt = build_system_prompt(tools)
    print()
    hr("SYSTEM PROMPT  (sent once, before Turn 0)")
    # Print the non-tools part in full, then summarise the tools JSON
    intro_end = system_prompt.index("Here is a list")
    print(system_prompt[:intro_end])
    print("Here is a list of functions in JSON format that you can invoke.")
    print(f"  [ ... {len(tools)} tool definitions — see --json flag for full payload ... ]")

    # ── turns ─────────────────────────────────────────────────────────────────
    questions = test_case["question"]
    messages  = [{"role": "system", "content": system_prompt}]

    for turn_idx, user_msgs in enumerate(questions):
        print()
        hr(f"TURN {turn_idx}")

        # Add user message
        for m in user_msgs:
            messages.append(m)
            print(f"\n  [USER]  {m['content']}")

        gt = ground_truth[turn_idx] if ground_truth else []
        print(f"\n  ── model must output (ground truth): {gt}")

        # Simulate model calls and tool responses
        for call in gt:
            result = FAKE_TOOL_RESULTS.get(call, {"result": "..."})
            model_msg = {"role": "assistant", "content": f"[{call}]"}
            tool_msg  = {"role": "tool",      "content": json.dumps(result)}
            messages.append(model_msg)
            messages.append(tool_msg)

            print(f"\n  [ASSISTANT OUTPUT]  [{call}]")
            print(f"  [TOOL RESULT]       {json.dumps(result)}")

        print(f"\n  ── after turn {turn_idx}: {len(messages)} messages in context ──")

    print()
    hr("FINAL CONTEXT  (all messages in order, roles only)")
    for i, m in enumerate(messages):
        content_preview = (m["content"][:70] + "...") if len(m["content"]) > 73 else m["content"]
        print(f"  [{i:02d}] role={m['role']:<12}  {content_preview}")

    print()
    hr("FIELD REFERENCE")
    rows = [
        ("question",          "4 sub-lists = 4 turns of user messages",                       "YES — becomes USER messages"),
        ("initial_config",    "Virtual filesystem starting state",                             "NO  — internal framework only"),
        ("involved_classes",  "Which func_doc JSON to load as tool list",                     "YES — determines all tools"),
        ("excluded_function", "Notes which tool to avoid (cp here)",                          "NO  — metadata, NOT filtered"),
        ("path",              "Expected function call sequence for eval",                     "NO  — evaluation only"),
    ]
    print(f"  {'Field':<20} {'Meaning':<50} {'Sent to LLM?'}")
    print(f"  {'-'*20} {'-'*50} {'-'*20}")
    for field, meaning, sent in rows:
        print(f"  {field:<20} {meaning:<50} {sent}")
    print()


def show_json_payload(test_case, tools, ground_truth):
    """Print the raw API request body for each turn (OpenAI chat-completion format)."""
    system_prompt = build_system_prompt(tools)
    messages = [{"role": "system", "content": system_prompt}]
    questions = test_case["question"]

    for turn_idx, user_msgs in enumerate(questions):
        for m in user_msgs:
            messages.append(m)

        payload = {
            "model": "<your-model>",
            "messages": messages,
        }
        print()
        hr(f"TURN {turn_idx}  — JSON payload sent to LLM")
        print(json.dumps(payload, indent=2))

        gt = ground_truth[turn_idx] if ground_truth else []
        for call in gt:
            result = FAKE_TOOL_RESULTS.get(call, {"result": "..."})
            messages.append({"role": "assistant", "content": f"[{call}]"})
            messages.append({"role": "tool",      "content": json.dumps(result)})


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_ID = "multi_turn_base_1"

    test_case    = load_test_case(TEST_ID)
    tools        = load_tools(test_case["involved_classes"])
    ground_truth = load_ground_truth(TEST_ID)

    if "--json" in sys.argv:
        show_json_payload(test_case, tools, ground_truth)
    else:
        show_human_readable(test_case, tools, ground_truth)
