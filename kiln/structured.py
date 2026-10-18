"""Answers held to a grammar over the model's tokens (llguidance): JSON for OpenAI's response_format (json_schema,
json_object), and tool calls in Qwen's format (kiln/chat.py renders them) that name an offered tool and give the
parameters its schema asks for (strict tools, tool_choice "required" or a named tool). The answer is drawn from the logits
the grammar allows by the engine's own sampler, with drafts: the tokens the grammar forces, then the MTP head's guesses,
each position masked to what the grammar allows there (Matcher.masks, Engine.generate_masked)."""
import json

import numpy as np

VOCAB = 248320                                       # the language model head's rows (engine/common.h)
# Whitespace before the answer (the template leaves the model a blank line after its reasoning) and between a JSON value's
# tokens (indentation 8 levels deep) is bounded: unbounded, a model that would rather not answer in JSON pads with
# whitespace to the length limit (Flash-Next after its reasoning, measured; the one-token-a-step path as well).
WS = "/[ \\t\\n]{0,4}/"
JSON_WS = "[ \\t\\n]{0,32}"


def bounded(schema):
    """A JSON schema whose value llguidance writes with bounded whitespace (its x-guidance whitespace_pattern)."""
    guide = dict(schema.get("x-guidance") or {})
    guide.setdefault("whitespace_pattern", JSON_WS)
    return {**schema, "x-guidance": guide}


class Grammars:
    def __init__(self, tok):
        """tok: kiln.chat.Tokens. Building llguidance's view of the vocabulary takes ~0.9 s: one per server."""
        import llguidance
        self.ll = llguidance
        self.tokenizer = llguidance.LLTokenizer(tok.full.to_str(), n_vocab=VOCAB, eos_token=tok.id["<|im_end|>"])

    @staticmethod
    def schema_of(response_format):
        """The JSON schema a response_format asks for (None: plain text). Raises ValueError."""
        if not response_format:
            return None
        if not isinstance(response_format, dict):
            raise ValueError("response_format must be an object")
        kind = response_format.get("type")
        if kind == "text":
            return None
        if kind == "json_object":
            return {"type": "object"}
        if kind == "json_schema":
            schema = (response_format.get("json_schema") or {}).get("schema")
            if not isinstance(schema, dict):
                raise ValueError("response_format json_schema needs a schema")
            return schema
        raise ValueError(f"unknown response_format type {kind!r}")

    def matcher(self, schema=None, tools=None, only=None, parallel=True, inside=False):
        """A matcher for an answer: `schema`'s JSON, or tool calls (tools: {name: parameters schema}), or either. only: the
        one tool it must call (tool_choice naming it), once; parallel=False: one call at most. inside: the answer starts in
        a call whose <tool_call> the model has written (strict calls in an answer otherwise free), and is calls to the end.
        Raises ValueError for what llguidance cannot follow."""
        rules = []
        if tools is not None:
            rules += call_rules(tools, only)
        if inside:
            start = "body" + ("" if not parallel or only else ' ("\\n" call)*')
        else:
            alts = ([f"%json {json.dumps(bounded(schema))}"] if schema is not None else []) + \
                   (["call" + ("" if not parallel or only else ' ("\\n" call)*')] if tools is not None else [])
            start = f"{WS} ({' | '.join(alts)})"
        lark = "\n".join([f"start: {start}"] + rules)
        err = self.ll.LLMatcher.validate_grammar(lark, self.tokenizer)
        if err:
            raise ValueError(f"cannot follow this {'JSON schema' if tools is None else 'tool schema'}: {err}")
        return Matcher(self.ll.LLMatcher(self.tokenizer, lark))


def call_rules(tools, only=None):
    """Lark rules for a tool call as the template writes one: `call` from its <tool_call>, `body` after it. A parameter
    the schema types as a string (or leaves untyped) is free text up to its closing tag (a pattern or length is left to
    the model); an enum of strings one of them; any other type JSON of that schema. Parameters come in the schema's order,
    the required ones always."""
    names = [only] if only is not None else list(tools)
    rules = ["call: <tool_call> body",
             'body: "\\n<function=" (' + " | ".join(f"f{i}" for i in range(len(names))) + ') "</function>\\n" </tool_call>',
             'text[lazy]: /(.|\\n)*/ "\\n</parameter>\\n"']
    for i, name in enumerate(names):
        schema = tools[name] if isinstance(tools[name], dict) else {}
        props = schema.get("properties") or {}
        need = set(schema.get("required") or [])
        defs = {k: schema[k] for k in ("$defs", "definitions") if k in schema}
        parts = []
        for j, (key, sub) in enumerate(props.items()):
            rules.append(f"p{i}_{j}: {json.dumps(f'<parameter={key}>' + chr(10))} {value(sub if isinstance(sub, dict) else {}, defs)}")
            parts.append(f"p{i}_{j}" + ("" if key in need else "?"))
        rules.append(f"f{i}: {json.dumps(name + '>' + chr(10))} {' '.join(parts)}".rstrip())
    return rules


def value(schema, defs):
    """A parameter's value and its closing tag, as Lark."""
    close = json.dumps("\n</parameter>\n")
    enum = schema["enum"] if "enum" in schema else [schema["const"]] if "const" in schema else None
    if enum and all(isinstance(v, str) for v in enum):
        return "(" + " | ".join(json.dumps(v) for v in enum) + f") {close}"
    return "text" if stringy(schema) else f"%json {json.dumps(bounded({**schema, **defs}))} {close}"


def stringy(schema):
    """Whether a value may be a string, which the template writes as it is (chat.parse_call reads it back so)."""
    kind = schema.get("type")
    if kind == "string" or isinstance(kind, list) and "string" in kind:
        return True
    branches = (schema.get("anyOf") or []) + (schema.get("oneOf") or [])
    if branches:
        return any(isinstance(b, dict) and stringy(b) for b in branches)
    return kind is None and not any(k in schema for k in ("properties", "items", "allOf", "$ref", "enum", "const"))


class Matcher:
    def __init__(self, m):
        self.m = m

    def bitmask(self):
        """Which tokens may come next: bit t % 32 of word t / 32 (uint32 [VOCAB / 32]; llguidance's has a word more)."""
        return np.frombuffer(self.m.compute_bitmask(), np.uint32)[:VOCAB // 32]

    def take(self, tok):
        """Advance past a token; then the tokens the grammar forces after it (already taken)."""
        self.consume([tok])
        forced = [] if self.m.is_stopped() else list(self.m.compute_ff_tokens())
        if forced and not self.m.consume_tokens(forced):
            raise RuntimeError(f"the grammar refused its own forced tokens: {self.m.get_error()}")
        return forced

    def consume(self, toks):
        for t in toks:
            if not self.m.consume_token(t):
                raise RuntimeError(f"the grammar refused token {t}: {self.m.get_error()}")

    def forced(self):
        """The tokens the grammar forces next (not taken)."""
        return [] if self.m.is_stopped() else list(self.m.compute_ff_tokens())

    def masks(self, drafts):
        """A step's bitmasks: the next position's, then the one after each draft in turn while the grammar allows them.
        Returns (masks [k + 1][VOCAB / 32], k): drafts past the first it refuses are not worth verifying."""
        rows, k = [self.bitmask()], 0
        for t in drafts:
            if self.m.is_stopped() or not (rows[-1][t >> 5] >> (t & 31)) & 1 or not self.m.consume_token(t):
                break
            k += 1
            rows.append(self.bitmask())                  # once the grammar is done, the end of the reply alone
        if k:
            self.m.rollback(k)
        return np.stack(rows), k

    def done(self):
        return self.m.is_stopped()
