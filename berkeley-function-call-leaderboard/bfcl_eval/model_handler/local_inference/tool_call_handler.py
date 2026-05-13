"""
Custom BFCL Handler for Tusher's GRPO-trained Tool-Calling Model
================================================================

Training format recap
---------------------
- Model was trained on multi-turn conversations where:
    * system   : task description + available tools (JSON list)
    * user     : query (possibly multi-turn)
    * assistant: [func_name(param=val, ...)]   ← pure function-call string
    * tool     : {"result": ...}               ← tool execution result (JSON)

- The model outputs tool calls in Python-call notation:
      [func1(a=1, b="x"), func2(c=True)]

- No <think> / <tool_call> XML tags — just the bracketed Python-call list.

BFCL interface contract
-----------------------
The handler must expose:
    decode_ast(result, language)   → list[dict]  e.g. [{"func": {"arg": val}}]
    decode_execute(result)         → list[str]   e.g. ["func(arg=val)"]
    _format_prompt(messages, function, turn_type) → str

Everything else (inference loop, multi-turn orchestration) is inherited from
OSSHandler / BaseHandler in the BFCL codebase.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tool_string(functions: list[dict] | dict) -> str:
    """Convert the BFCL function spec (dict or list[dict]) to a compact
    numbered list that was used at training time."""
    if isinstance(functions, dict):
        functions = [functions]

    lines = []
    for idx, fn in enumerate(functions, start=1):
        params = fn.get("parameters", {}).get("properties", {})
        lines.append(
            f"{idx}. Name: {fn['name']}\n"
            f"   Description: {fn.get('description', '')}\n"
            f"   Parameters: {json.dumps(params, ensure_ascii=False)}"
        )
    return "\n".join(lines)


def _build_system_prompt(tool_string: str) -> str:
    """Reproduce the exact system prompt seen during training."""
    return (
        "You are an expert in composing functions. You are given a question and "
        "a set of possible functions. Based on the question, you will need to make "
        "one or more function/tool calls to achieve the purpose. If none of the "
        "functions can be used, point it out. If the given question lacks the "
        "parameters required by the function, also point it out.\n\n"
        "You should only return the function calls in your response.\n\n"
        "If you decide to invoke any of the function(s), you MUST put it in the "
        "format of [func_name1(params_name1=params_value1, "
        "params_name2=params_value2...), func_name2(params)] "
        "You SHOULD NOT include any other text in the response.\n\n"
        "At each turn, you should try your best to complete the tasks requested by "
        "the user within the current turn. Continue to output functions to call "
        "until you have fulfilled the user's request to the best of your ability. "
        "Once you have no more functions to call, the system will consider the "
        "current turn complete and proceed to the next turn or task.\n\n"
        f"Here is a list of functions in JSON format that you can invoke.\n"
        f"{tool_string}\n"
    )


# ---------------------------------------------------------------------------
# Parsing logic
# ---------------------------------------------------------------------------

def _extract_call_block(raw: str) -> str:
    """
    Extract the bracketed call list from a model response.

    The model outputs something like:
        [func1(a=1, b="x"), func2(c=True)]
    Possibly with surrounding whitespace or stray text.
    """
    raw = raw.strip()

    # Fast path: response is already a clean bracketed list
    if raw.startswith("[") and raw.endswith("]"):
        return raw

    # Try to find the outermost [...] block
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        return match.group(0)

    return raw  # return as-is and let the parser fail gracefully


def _parse_call_string(call_str: str) -> list[dict]:
    """
    Parse a bracketed Python-call string into a list of
        {"name": str, "arguments": dict}
    dicts.

    Strategy:
    1. Use ast.parse on the bracketed expression.
    2. Walk each Call node to extract the function name and keyword args.
    3. Fall back to regex-based extraction for malformed outputs.
    """
    call_str = call_str.strip()

    # ---- primary: ast-based ------------------------------------------------
    try:
        tree = ast.parse(call_str, mode="eval")
        calls = _walk_ast_calls(tree.body)
        if calls:
            return calls
    except SyntaxError:
        pass

    # ---- fallback: regex ---------------------------------------------------
    return _regex_parse_calls(call_str)


def _walk_ast_calls(node) -> list[dict]:
    """Recursively collect Call nodes from an AST expression."""
    results = []

    if isinstance(node, ast.List):
        for elt in node.elts:
            results.extend(_walk_ast_calls(elt))
        return results

    if isinstance(node, ast.Call):
        name = _ast_name(node.func)
        if name is None:
            return results

        kwargs = {}
        for kw in node.keywords:
            kwargs[kw.arg] = ast.literal_eval(kw.value)

        # positional args – less common in BFCL but handle gracefully
        for i, arg in enumerate(node.args):
            try:
                kwargs[f"_pos{i}"] = ast.literal_eval(arg)
            except Exception:
                pass

        results.append({"name": name, "arguments": kwargs})
        return results

    return results


def _ast_name(node) -> str | None:
    """Extract a dotted name from an AST node (handles a.b.c style)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _ast_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


# Pre-compiled regex for fallback parsing
_FUNC_RE = re.compile(
    r"([\w.]+)\s*\(([^()]*)\)",  # func_name(args...)
    re.DOTALL,
)
_KW_RE = re.compile(r"(\w+)\s*=\s*(.+?)(?=,\s*\w+\s*=|$)", re.DOTALL)


def _regex_parse_calls(call_str: str) -> list[dict]:
    """Last-resort regex parser for malformed model outputs."""
    results = []
    for m in _FUNC_RE.finditer(call_str):
        name = m.group(1)
        args_str = m.group(2).strip()
        kwargs = {}
        for kw in _KW_RE.finditer(args_str):
            key = kw.group(1)
            val_str = kw.group(2).strip().rstrip(",").strip()
            try:
                kwargs[key] = ast.literal_eval(val_str)
            except Exception:
                kwargs[key] = val_str  # keep as string if unparseable
        results.append({"name": name, "arguments": kwargs})
    return results


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class TusherModelHandler(OSSHandler):
    """
    BFCL inference handler for the GRPO-trained tool-calling model.

    The model was fine-tuned to produce bare Python-call lists:
        [func1(a=1), func2(b="x")]

    This handler:
      1. Formats prompts in the exact style used during training.
      2. Parses the bracketed Python-call output robustly.
      3. Exposes decode_ast / decode_execute for BFCL evaluation.
      4. Handles single-turn, multi-turn, and parallel tool calls.
    """

    def __init__(self, model_name: str, temperature: float, **kwargs) -> None:
        super().__init__(model_name, temperature, **kwargs)

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    @override
    def _format_prompt(
        self,
        messages: list[dict],
        function: list[dict] | dict,
        turn_type: str = "single_turn",
    ) -> str:
        """
        Build the full prompt string in the training format.

        Structure:
            <|im_start|>system
            {system_with_tools}
            <|im_end|>
            <|im_start|>user
            {user_query}
            <|im_end|>
            [... assistant / tool turns ...]
            <|im_start|>assistant
        """
        tool_string = _build_tool_string(function)
        system_prompt = _build_system_prompt(tool_string)

        parts: list[str] = [
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        ]

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "").strip()

            if role == "system":
                # already handled above; skip duplicate system messages
                continue

            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")

            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")

            elif role == "tool":
                # Tool results are fed back as a user turn (mirrors training data)
                # We wrap them clearly so the model can distinguish them.
                tool_name = msg.get("name", "tool")
                result_text = (
                    f"<tool_response>\n"
                    f"Function '{tool_name}' returned:\n{content}\n"
                    f"</tool_response>"
                )
                parts.append(f"<|im_start|>user\n{result_text}<|im_end|>\n")

        # Generation prompt
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @override
    def decode_ast(self, result: str, language: str = "Python") -> list[dict]:
        """
        Parse model output into BFCL AST format.

        Returns:
            list of { func_name: { arg_name: arg_value } }
            e.g. [{"add": {"a": 1, "b": 2}}]

        Returns [] if no valid tool calls are detected.
        """
        call_block = _extract_call_block(result)

        # Detect explicit "no tool" signals
        if _is_no_tool_response(call_block):
            return []

        parsed = _parse_call_string(call_block)
        if not parsed:
            return []

        # Convert to BFCL's expected AST format
        decoded: list[dict] = []
        for call in parsed:
            name = call["name"].strip()
            args = call.get("arguments", {})

            # Skip explicit None-tool placeholders
            if name.lower() in ("none", "null", ""):
                continue

            decoded.append({name: args})

        return decoded

    @override
    def decode_execute(self, result: str) -> list[str]:
        """
        Parse model output into executable Python call strings.

        Returns:
            list of strings like ["func(a=1, b='x')"]

        Returns [] if no valid tool calls are detected.
        """
        call_block = _extract_call_block(result)

        if _is_no_tool_response(call_block):
            return []

        parsed = _parse_call_string(call_block)
        if not parsed:
            return []

        executable: list[str] = []
        for call in parsed:
            name = call["name"].strip()
            if name.lower() in ("none", "null", ""):
                continue
            args = call.get("arguments", {})
            args_str = ", ".join(
                f"{k}={repr(v)}" for k, v in args.items()
                if not k.startswith("_pos")  # skip anonymous positional args
            )
            executable.append(f"{name}({args_str})")

        return executable

    # ------------------------------------------------------------------
    # Response parsing (for multi-turn inference loop)
    # ------------------------------------------------------------------

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """Extract text from a vLLM/HF completion response."""
        model_response = api_response.choices[0].text
        return {
            "model_responses": model_response,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self,
        inference_data: dict,
        model_response_data: dict,
    ) -> dict:
        """Append the assistant message to conversation history."""
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": model_response_data["model_responses"],
            }
        )
        return inference_data


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _is_no_tool_response(text: str) -> bool:
    """
    Detect when the model explicitly says no tool is applicable.
    Training data included phrases like:
        "None of the functions can be used"
        "parameters required by the function"
    """
    lower = text.lower()
    signals = [
        "none of the functions",
        "no appropriate tools",
        "cannot be used",
        "parameters required",
        "i cannot",
        "i can't",
        "no tool",
    ]
    return any(s in lower for s in signals)