import json

import numpy as np

VOCAB = 248320
WS = "/[ \\t\\n]{0,4}/"
JSON_WS = "[ \\t\\n]{0,32}"


def bounded(schema):
    guide = dict(schema.get("x-guidance") or {})
    guide.setdefault("whitespace_pattern", JSON_WS)
    return {**schema, "x-guidance": guide}


class Grammars:
    def __init__(self, tok):
        import llguidance
        self.ll = llguidance
        self.tokenizer = llguidance.LLTokenizer(tok.full.to_str(), n_vocab=VOCAB, eos_token=tok.id["<|im_end|>"])

    @staticmethod
    def schema_of(response_format):
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
    close = json.dumps("\n</parameter>\n")
    enum = schema["enum"] if "enum" in schema else [schema["const"]] if "const" in schema else None
    if enum and all(isinstance(v, str) for v in enum):
        return "(" + " | ".join(json.dumps(v) for v in enum) + f") {close}"
    return "text" if stringy(schema) else f"%json {json.dumps(bounded({**schema, **defs}))} {close}"


def stringy(schema):
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
        return np.frombuffer(self.m.compute_bitmask(), np.uint32)[:VOCAB // 32]

    def take(self, tok):
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
        return [] if self.m.is_stopped() else list(self.m.compute_ff_tokens())

    def masks(self, drafts):
        rows, k = [self.bitmask()], 0
        for t in drafts:
            if self.m.is_stopped() or not (rows[-1][t >> 5] >> (t & 31)) & 1 or not self.m.consume_token(t):
                break
            k += 1
            rows.append(self.bitmask())
        if k:
            self.m.rollback(k)
        return np.stack(rows), k

    def done(self):
        return self.m.is_stopped()
