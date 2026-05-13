"""
Custom BFCL Handler for Tusher's GRPO-trained Tool-Calling Model
================================================================

Training format recap
---------------------
- Model was trained on multi-turn conversations where:
    * system   : task description + available tools (full JSON list)
    * user     : raw user query
    * assistant: [func_name(param=val, ...)]   <- bare Python-call list
    * tool     : {"result": ...}               <- tool execution result (JSON string)

- The model outputs tool calls in Python-call notation only:
      [func1(a=1, b="x"), func2(c=True)]
  There are NO <think> / <tool_call> XML tags in the training output.

BFCL interface contract
-----------------------
Constructor : __init__(model_name, temperature, registry_name, is_fc_model,
                       dtype="float16", **kwargs)
Prompt      : _format_prompt(messages, function, turn_type="single_turn") -> str
AST decode  : decode_ast(result, language, has_tool_call_tag)  -> list[dict]
Exec decode : decode_execute(result, has_tool_call_tag)        -> list[str]

Everything else (inference loop, vLLM server calls, multi-turn orchestration)
is inherited from OSSHandler / BaseHandler in the BFCL codebase.
"""

from __future__ import annotations

import ast
import copy
import json
import re

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override


# ---------------------------------------------------------------------------
# Schema / message normalisation  (carried over from ToolCallHandler)
# ---------------------------------------------------------------------------

def _normalize_schema_types(tool: dict) -> dict:
    """
    BFCL schemas use 'dict' as the parameters type; some training data used
    'object'.  Recursively rewrite 'dict' -> 'object' so the schema the model
    sees at eval time matches what it learned during fine-tuning.
    """
    if not isinstance(tool, dict):
        return tool
    tool = copy.deepcopy(tool)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "dict":
                node["type"] = "object"
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(tool)
    return tool


def _normalize_tool_message(msg: dict) -> dict:
    """
    BFCL sometimes passes tool results without a `name` field.  Training data
    always had {"role":"tool","name":"...","content":"..."}.  Ensure the name
    is always present and that content is a plain string.
    """
    if msg.get("role") != "tool":
        return msg
    msg = dict(msg)
    if not msg.get("name"):
        msg["name"] = "tool"
    if not isinstance(msg.get("content"), str):
        try:
            msg["content"] = json.dumps(msg["content"], ensure_ascii=False)
        except Exception:
            msg["content"] = str(msg.get("content", ""))
    return msg


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _build_tool_list_json(functions: list[dict] | dict) -> str:
    """
    Serialise the tool list exactly as it appeared in training system prompts:
    a JSON array string.
    """
    if isinstance(functions, dict):
        functions = [functions]
    tools = [_normalize_schema_types(f) for f in functions if f is not None]
    return json.dumps(tools, ensure_ascii=False, indent=2)


def _build_system_prompt(tool_json: str) -> str:
    """
    Reproduce the exact system prompt seen during training.
    The tool list is appended verbatim as JSON so the model recognises it.
    """
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
        "Here is a list of functions in JSON format that you can invoke.\n"
        f"{tool_json}\n"
    )


# ---------------------------------------------------------------------------
# Output parsing helpers
# ---------------------------------------------------------------------------

def _extract_call_block(raw: str) -> str:
    """
    Pull the outermost [...] block out of the raw model output.

    Training taught the model to emit ONLY a bracketed list, but under
    distribution shift it sometimes adds a brief explanation before or after.
    """
    raw = raw.strip()

    # Fast path -- already a clean bracketed list
    if raw.startswith("[") and raw.endswith("]"):
        return raw

    # Find the outermost [...] span (greedy -- gets the longest match)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        return match.group(0)

    # Nothing found -- return as-is so downstream parsers can signal empty
    return raw


def _walk_ast_calls(node) -> list[dict]:
    """Recursively collect Call nodes from an AST expression node."""
    results: list[dict] = []

    if isinstance(node, ast.List):
        for elt in node.elts:
            results.extend(_walk_ast_calls(elt))
        return results

    if isinstance(node, ast.Call):
        name = _ast_dotted_name(node.func)
        if not name:
            return results

        kwargs: dict = {}
        for kw in node.keywords:
            if kw.arg is None:        # skip **spread
                continue
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, TypeError):
                try:
                    kwargs[kw.arg] = ast.unparse(kw.value)
                except Exception:
                    pass              # drop unparseable arg rather than crash

        # Positional args are uncommon in BFCL but handle defensively
        for i, arg in enumerate(node.args):
            try:
                kwargs[f"_pos{i}"] = ast.literal_eval(arg)
            except Exception:
                pass

        results.append({"name": name, "arguments": kwargs})
        return results

    return results


def _ast_dotted_name(node) -> str | None:
    """Convert ast.Name / ast.Attribute back to a dotted string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _ast_dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


# Pre-compiled regex patterns for the fallback parser
_FUNC_RE = re.compile(r"([\w.]+)\s*\(([^()]*)\)", re.DOTALL)
_KW_RE   = re.compile(r"(\w+)\s*=\s*(.+?)(?=,\s*\w+\s*=|$)", re.DOTALL)


def _regex_parse_calls(call_str: str) -> list[dict]:
    """
    Last-resort regex parser for malformed model outputs where ast.parse
    fails entirely (e.g. an unclosed bracket, mismatched quotes).
    """
    results: list[dict] = []
    for m in _FUNC_RE.finditer(call_str):
        name = m.group(1)
        args_str = m.group(2).strip()
        kwargs: dict = {}
        for kw in _KW_RE.finditer(args_str):
            key = kw.group(1)
            val_str = kw.group(2).strip().rstrip(",").strip()
            try:
                kwargs[key] = ast.literal_eval(val_str)
            except Exception:
                kwargs[key] = val_str   # keep raw string rather than crash
        results.append({"name": name, "arguments": kwargs})
    return results


def _parse_call_string(call_str: str) -> list[dict]:
    """
    Parse a (possibly bracketed) Python-call string into a list of
        {"name": str, "arguments": dict}
    dicts.  Two-layer strategy:
      1. ast.parse  -- handles all valid Python literals correctly
      2. regex      -- handles partially malformed outputs
    """
    call_str = call_str.strip()
    if not call_str:
        return []

    # Wrap a bare single call so ast can parse it as a list expression
    candidate = call_str if call_str.startswith("[") else f"[{call_str}]"

    try:
        tree = ast.parse(candidate, mode="eval")
        calls = _walk_ast_calls(tree.body)
        if calls:
            return calls
    except (SyntaxError, ValueError):
        pass

    return _regex_parse_calls(call_str)


def _is_no_tool_response(text: str) -> bool:
    """
    Return True when the model is explicitly saying no tool applies,
    rather than emitting a broken or empty call list.
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
        "not possible",
    ]
    return any(s in lower for s in signals)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class ToolCallHandler(OSSHandler):
    """
    BFCL inference handler for the GRPO-trained tool-calling model.

    The model was fine-tuned to produce bare Python-call lists:
        [func1(a=1), func2(b="x")]

    This handler:
      1. Formats prompts in the exact style used during training.
      2. Parses the bracketed Python-call output robustly (2-layer strategy).
      3. Exposes decode_ast / decode_execute for BFCL evaluation.
      4. Handles single-turn, multi-turn, and parallel tool calls.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        dtype: str = "float16",
        **kwargs,
    ) -> None:
        super().__init__(
            model_name, temperature, registry_name, is_fc_model,
            dtype=dtype, **kwargs
        )

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    @override
    def _format_prompt(self, messages, function, turn_type="single_turn"):
        """
        Build the full prompt string in the training format.

        Final structure (Qwen chat-template tokens):
            <|im_start|>system
            {system prompt including JSON tool list}
            <|im_end|>
            <|im_start|>user
            {first user query}
            <|im_end|>
            [<|im_start|>assistant
            [func(...)]
            <|im_end|>
            <|im_start|>user
            <tool_response>...</tool_response>
            <|im_end|>]  <- repeated for each tool call round
            <|im_start|>assistant      <- generation prompt (no <|im_end|>)
        """
        tool_json   = _build_tool_list_json(function)
        system_text = _build_system_prompt(tool_json)

        parts: list[str] = [f"<|im_start|>system\n{system_text}<|im_end|>\n"]

        for msg in messages:
            role    = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                # System prompt already injected above; skip any extra system msg
                continue

            elif role == "user":
                content = content.strip() if isinstance(content, str) else content
                parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")

            elif role == "assistant":
                content = content.strip() if isinstance(content, str) else content
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")

            elif role == "tool":
                # Normalise tool message (ensures name + string content)
                msg         = _normalize_tool_message(msg)
                name        = msg.get("name", "tool")
                tc          = msg.get("content", "")
                result_text = (
                    f"<tool_response>\n"
                    f"Function '{name}' returned:\n{tc}\n"
                    f"</tool_response>"
                )
                parts.append(f"<|im_start|>user\n{result_text}<|im_end|>\n")

        # Append generation prompt -- no closing <|im_end|>
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    # ------------------------------------------------------------------
    # Output decoding
    # ------------------------------------------------------------------

    @override
    def decode_ast(self, result, language="Python", has_tool_call_tag=False):
        """
        Parse raw model output into BFCL AST format.

        Returns:
            [ { func_name: { arg_name: arg_value } }, ... ]
            e.g. [{"add": {"a": 1, "b": 2}}]

        Returns [] for empty output, prose-only output, or explicit
        "no tool available" replies.
        """
        call_block = _extract_call_block(result)

        if _is_no_tool_response(call_block):
            return []

        parsed = _parse_call_string(call_block)
        if not parsed:
            return []

        decoded: list[dict] = []
        for call in parsed:
            name = call["name"].strip()
            if name.lower() in ("none", "null", ""):
                continue
            args = {
                k: v for k, v in call.get("arguments", {}).items()
                if not k.startswith("_pos")          # drop positional placeholders
            }
            decoded.append({name: args})

        return decoded

    @override
    def decode_execute(self, result, has_tool_call_tag=False):
        """
        Parse raw model output into executable Python call strings.

        Returns:
            [ "func(a=1, b='x')", ... ]

        Returns [] for empty / no-tool outputs.

        Note: dotted names are preserved (e.g. "GorillaFileSystem.ls") because
        multi-turn eval injects the class instance into the execution namespace.
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
            args = {
                k: v for k, v in call.get("arguments", {}).items()
                if not k.startswith("_pos")
            }
            args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
            executable.append(f"{name}({args_str})")

        return executable

    # ------------------------------------------------------------------
    # Inference plumbing  (multi-turn / vLLM completion API)
    # ------------------------------------------------------------------

    @override
    def _parse_query_response_prompting(self, api_response):
        """Extract text + token counts from a vLLM/HF completion response."""
        model_response = api_response.choices[0].text
        return {
            "model_responses": model_response,
            "input_token":     api_response.usage.prompt_tokens,
            "output_token":    api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(self, inference_data, model_response_data):
        """Append the assistant turn to the live conversation history."""
        inference_data["message"].append(
            {
                "role":    "assistant",
                "content": model_response_data["model_responses"],
            }
        )
        return inference_data