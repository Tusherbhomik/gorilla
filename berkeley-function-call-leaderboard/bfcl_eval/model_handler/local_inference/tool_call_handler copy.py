import ast
import copy
import json
import re

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override


class ToolCallHandler(OSSHandler):


    def __init__(self, model_name, temperature, registry_name, is_fc_model, dtype="float16", **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, dtype=dtype, **kwargs)
        self._tokenizer = None  # loaded lazily on first use; cached thereafter

    def _ensure_tokenizer(self):
        """Load tokenizer once and cache it. Uses model_name_huggingface."""
        if self._tokenizer is not None:
            return
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_huggingface,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @staticmethod
    def _normalize_schema_types(tool):
        """
        BFCL schemas use 'dict' as the parameters type; training data used 'object'.
        Recursively rewrite 'dict' -> 'object' so the schema Qwen sees at eval time
        matches what the model learned during fine-tuning.
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

    @staticmethod
    def _normalize_tool_message(msg):
        """
        BFCL sometimes passes tool results without a `name` field. Training data
        always had `{"role":"tool","name":"...","content":"..."}`. Qwen's chat
        template is more reliable when the name is present.
        """
        if msg.get("role") != "tool":
            return msg
        msg = dict(msg)
        if "name" not in msg or not msg["name"]:
            msg["name"] = "tool"
        # Ensure content is a string (training data had JSON-encoded strings)
        if not isinstance(msg.get("content"), str):
            try:
                msg["content"] = json.dumps(msg["content"], ensure_ascii=False)
            except Exception:
                msg["content"] = str(msg.get("content", ""))
        return msg

    @override
    def _format_prompt(self, messages, function, turn_type="single_turn"):
        """
        Render prompt using Qwen's chat template with RAW BFCL tool schemas
        (no OpenAI wrapper), and 'dict' -> 'object' normalization to match training.
        """
        self._ensure_tokenizer()

        # 1) Normalize tool messages (ensure name + stringified content)
        norm_messages = [self._normalize_tool_message(m) for m in messages]

        # 2) Build tool list — raw schema, just normalize dict->object
        tools = function if isinstance(function, list) else [function]
        tools = [self._normalize_schema_types(t) for t in tools if t is not None]

        try:
            prompt = self._tokenizer.apply_chat_template(
                norm_messages,
                tools=tools if tools else None,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = self._manual_qwen_format(norm_messages, tools)

        # Defensive: if chat template silently returned empty/None, fall back
        if not prompt or not isinstance(prompt, str):
            prompt = self._manual_qwen_format(norm_messages, tools)

        return prompt

    def _manual_qwen_format(self, messages, tools):
        """Fallback Qwen-format prompt builder if apply_chat_template fails."""
        parts = []

        system_content = ""
        if messages and messages[0].get("role") == "system":
            system_content = messages[0]["content"]
            messages = messages[1:]

        if tools:
            tool_spec = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            tool_spec += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            for t in tools:
                tool_spec += json.dumps(t, ensure_ascii=False) + "\n"
            tool_spec += "</tools>\n\n"
            tool_spec += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            tool_spec += "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
            system_content = (system_content + tool_spec).strip()

        parts.append(f"<|im_start|>system\n{system_content}<|im_end|>\n")

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "tool":
                tool_name = msg.get("name", "tool")
                parts.append(
                    f"<|im_start|>user\n<tool_response>\n{tool_name}: {content}\n</tool_response><|im_end|>\n"
                )
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        """Parse model output for AST-based evaluation (non-live / live)."""
        tool_calls = self._extract_tool_calls(result)
        decoded = []
        for tc in tool_calls:
            name = tc.get("name", "").strip()
            if not name or name.lower() == "none":
                continue
            args = tc.get("arguments", tc.get("parameters", {}))
            if not isinstance(args, dict):
                args = {}
            decoded.append({name: args})
        return decoded

    @override
    def decode_execute(self, result, has_tool_call_tag):
        """Parse model output for execution-based evaluation (multi-turn)."""
        tool_calls = self._extract_tool_calls(result)
        python_calls = []
        for tc in tool_calls:
            name = tc.get("name", "").strip()
            if not name or name.lower() == "none":
                continue
            args = tc.get("arguments", tc.get("parameters", {}))
            if not isinstance(args, dict):
                args = {}

            # BFCL multi-turn uses flat method names. Strip class prefix
            # (GorillaFileSystem.mv -> mv) so eval() finds the bound method.
            if "." in name:
                name = name.split(".")[-1]

            args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
            python_calls.append(f"{name}({args_str})")

        return python_calls

    @staticmethod
    def _extract_tool_calls(text):
        """
        Permissive extractor for <tool_call>{...}</tool_call> blocks.

        Parsing order (each step only runs if previous found nothing):
          1. Regex match on <tool_call>...</tool_call> blocks (primary path)
          2. Split-based parsing of last <tool_call>...</tool_call> segment
          3. Python-syntax fallback for [func(arg=val), ...] notation
             (safety net for cases where the model drops the tags entirely)
        """
        if not text or not isinstance(text, str):
            return []

        calls = []

        # ── Step 1: Primary regex path ────────────────────────────────────────
        if "<tool_call>" in text:
            pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
            for match in re.findall(pattern, text, re.DOTALL):
                try:
                    obj = json.loads(match)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict) or "name" not in obj:
                    continue
                if "arguments" not in obj and "parameters" in obj:
                    obj["arguments"] = obj["parameters"]
                calls.append(obj)

            # ── Step 2: Split-based fallback (only if regex found nothing) ──
            if not calls:
                try:
                    segment = text.split("<tool_call>")[-1].split("</tool_call>")[0].strip()
                    for line in (l.strip() for l in segment.split("\n") if l.strip()):
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict) and "name" in obj:
                            if "arguments" not in obj and "parameters" in obj:
                                obj["arguments"] = obj["parameters"]
                            calls.append(obj)
                except Exception:
                    pass

        # ── Step 3: Python-syntax fallback ────────────────────────────────────
        # Only fires when no <tool_call>-based parsing succeeded. This handles
        # rare cases where the model emits Python call notation directly:
        #   [find_prime_numbers(start=50, end=150), get_fibonacci_sequence(count=150)]
        # Or single calls:
        #   func_name(arg=value)
        if not calls:
            calls = ToolCallHandler._parse_python_calls(text)

        return calls

    @staticmethod
    def _parse_python_calls(text):
        """
        Safely parse Python-style tool calls using ast. Returns [] on any failure.

        Accepts:
          - List of calls:  [func1(a=1), func2(b=2)]
          - Single call:    func1(a=1)
          - Dotted names:   module.func1(a=1)

        Safety properties:
          - Uses ast.parse(mode="eval") — no code execution
          - Uses ast.literal_eval for argument values — only literals allowed
          - Returns [] on any parse error or unexpected node type
        """
        if not text or not isinstance(text, str):
            return []

        candidate = text.strip()
        if not candidate:
            return []

        # Heuristic: must look like a call expression. Reject prose to avoid
        # accidentally matching natural language that happens to contain parens.
        if "(" not in candidate or ")" not in candidate:
            return []

        # Wrap single calls in a list for unified parsing
        if not candidate.startswith("["):
            candidate = f"[{candidate}]"

        try:
            tree = ast.parse(candidate, mode="eval")
        except (SyntaxError, ValueError):
            return []

        if not isinstance(tree.body, ast.List):
            return []

        calls = []
        for elt in tree.body.elts:
            if not isinstance(elt, ast.Call):
                continue

            name = ToolCallHandler._unparse_call_name(elt.func)
            if not name:
                continue

            args = {}
            ok = True
            for kw in elt.keywords:
                if kw.arg is None:  # skip **kwargs spreads
                    continue
                try:
                    args[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    # Try unparse fallback for things like `lambda x: x+1`
                    try:
                        args[kw.arg] = ast.unparse(kw.value)
                    except (AttributeError, ValueError):
                        ok = False
                        break

            if ok and name:
                calls.append({"name": name, "arguments": args})

        return calls

    @staticmethod
    def _unparse_call_name(node):
        """Convert ast.Name or ast.Attribute back to a dotted string."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = ToolCallHandler._unparse_call_name(node.value)
            if parent is None:
                return node.attr
            return f"{parent}.{node.attr}"
        return None