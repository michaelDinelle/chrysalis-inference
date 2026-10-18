"""The mechanical cases of AGENTS.md's comment rule: section dividers, bare file-name labels, Arrange/Act/Assert labels,
TODOs without an issue link, commented-out code, JSDoc types or empty @param / @returns tags, and comments that only
repeat the name on the line below. Judging whether a comment says why is left to people.
  python3 tools/lint_comments.py [file ...]      (default: every source file git does not ignore)"""
import ast, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_LIKE = (".cpp", ".hip", ".h", ".ts", ".css")
SOURCES = C_LIKE + (".py", ".sh", ".html")


def c_comments(src, ts):
    """(offset, text) of every comment; string, char, template and raw string literals are skipped."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i) and not src.startswith("://", i - 1 if i else 0):
            j = src.find("\n", i); j = n if j < 0 else j
            out.append((i, src[i:j])); i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2); j = n if j < 0 else j + 2
            out.append((i, src[i:j])); i = j
        elif src.startswith('R"', i) and not (i and (src[i - 1].isalnum() or src[i - 1] == "_")):
            m = re.match(r'R"([^(\s]*)\(', src[i:])
            end = src.find(")" + m.group(1) + '"', i) if m else -1
            i = i + 1 if not m else (n if end < 0 else end + len(m.group(1)) + 2)
        elif c in "\"'" or (ts and c == "`"):
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            i = j + 1
        elif ts and c == "/" and regex_starts(src, i):
            # a regex literal (a quote in /[A-Za-z']/ would otherwise open a string that hides the code after it)
            j, in_class = i + 1, False
            while j < n and src[j] != "\n" and (in_class or src[j] != "/"):
                if src[j] == "\\":
                    j += 1
                elif src[j] in "[]":
                    in_class = src[j] == "["
                j += 1
            i = j + 1
        else:
            i += 1
    return out


def regex_starts(src, i):
    """A / that begins a regex literal rather than dividing: after an operator, an opening bracket, punctuation, a
    keyword that takes an expression, or nothing."""
    k = i - 1
    while k >= 0 and src[k] in " \t\r\n":
        k -= 1
    if k < 0 or src[k] in "(,=:[!&|?{};+-*%<>~^":
        return True
    word = re.search(r"[A-Za-z_$]+$", src[:k + 1])
    return bool(word) and word.group(0) in ("return", "typeof", "case", "of", "in", "void", "delete", "throw", "new", "else", "do", "yield", "await")


def hash_comments(src, python):
    if python:
        import io, tokenize
        offs, out = [0], []
        for l in src.split("\n"):
            offs.append(offs[-1] + len(l) + 1)
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type == tokenize.COMMENT and not (t.start[0] == 1 and t.string.startswith("#!")):
                out.append((offs[t.start[0] - 1] + t.start[1], t.string))
        return out
    out, off = [], 0
    for ln, line in enumerate(src.split("\n")):
        q = None
        for k, ch in enumerate(line):
            if q:
                q = None if ch == q else q
            elif ch in "\"'":
                q = ch
            elif ch == "#" and (k == 0 or line[k - 1].isspace()) and not (ln == 0 and line.startswith("#!")):
                if line[k - 1:k + 1] != "$#":
                    out.append((off + k, line[k:])); break
        off += len(line) + 1
    return out


def comments(path, src):
    if path.endswith(".py"): return hash_comments(src, True)
    if path.endswith(".sh") or os.path.basename(path) == "Makefile": return hash_comments(src, False)
    if path.endswith(C_LIKE): return c_comments(src, path.endswith(".ts"))
    if path.endswith(".html"):
        out = [(m.start(), m.group(0)) for m in re.finditer(r"<!--.*?-->", src, re.S)]
        for m in re.finditer(r"<(script|style)[^>]*>(.*?)</\1>", src, re.S):
            out += [(m.start(2) + s, t) for s, t in c_comments(m.group(2), True)]
        return out
    return []


def body(text):
    """A comment's words: markers, the closing */ and decoration stripped (first line of a block comment)."""
    t = text.split("\n")[0].strip()
    for a, b in (("<!--", "-->"), ("/*", "*/"), ("//", ""), ("#", "")):
        if t.startswith(a):
            t = t[len(a):]
            if b and t.rstrip().endswith(b): t = t.rstrip()[:-len(b)]
            break
    return t.strip()


DIRECTIVE = re.compile(r"^(noqa|type:|pylint:|fmt:|pragma|eslint|prettier|ts-|@ts-|clang-format|NOLINT|knip)", re.I)


PROSE = re.compile(r"\b[A-Za-z']+(?: [A-Za-z']+){3}\b")    # four plain words in a row: a sentence, not code
C_PROSE = re.compile(r"\b(the|a|an|of|in|one|no|is|are|its|it|to|by|with|at|as|when|then|and|or|per|each)\b")
STATEMENTS = (ast.Assign, ast.AugAssign, ast.Import, ast.ImportFrom, ast.Return, ast.If, ast.For, ast.While,
              ast.FunctionDef, ast.ClassDef, ast.Raise, ast.With, ast.Try)


def looks_like_code(t, path):
    if not t or DIRECTIVE.match(t) or PROSE.search(t):
        return False
    if path.endswith(".py"):
        try:
            tree = ast.parse(t)
        except SyntaxError:
            return False
        s = tree.body[0] if tree.body else None
        return isinstance(s, STATEMENTS) or isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
    if C_PROSE.search(t):
        return False
    if path.endswith(".css"):
        return bool(re.fullmatch(r"[\w\s.#:>,*\[\]=\"'()-]+\{[^}]*\}?|[\w-]+\s*:\s*[^;]+;", t))
    return bool(re.search(r"[=(].*;\s*$|^[{}]\s*$|^(if|for|while|return|else)\b.*[;{]\s*$", t))


def words(s):
    return [w.lower() for w in re.findall(r"[A-Za-z][a-z]*|[A-Z]+(?![a-z])|\d+", s)]


def lint(path):
    src = open(os.path.join(ROOT, path), encoding="utf-8").read()
    lines = src.split("\n")
    starts = [0]
    for l in lines:
        starts.append(starts[-1] + len(l) + 1)
    found = []
    for off, text in comments(path, src):
        ln = next(i for i in range(len(lines)) if starts[i] <= off < starts[i + 1])
        t = body(text)
        full_line = not lines[ln][:off - starts[ln]].strip()
        rule = None
        if re.match(r"^[-=*#~]{3,}", t) or re.search(r"[-=*#~]{3,}$", t):
            rule = "section divider"
        elif re.fullmatch(r"[\w./-]+\.(hip|cpp|h|py|ts|css|html|sh)", t.strip(" :-")):
            rule = "file-name label"
        elif re.fullmatch(r"(arrange|act|assert)\b\W*", t, re.I):
            rule = "Arrange/Act/Assert label"
        elif re.search(r"\b(TODO|FIXME|XXX|HACK)\b", t) and not re.search(r"#\d+|https?://", t):
            rule = "TODO without an issue link"
        elif path.endswith(".ts") and re.search(r"@(param|returns?|type)\s*\{", text):
            rule = "JSDoc type (TypeScript has the type)"
        elif re.search(r"@param\s+\w+\s*(\*/)?$|@returns?\s*(\*/)?$", text, re.M):
            rule = "empty @param / @returns"
        elif looks_like_code(t, path):
            rule = "commented-out code"
        elif full_line and 0 < len(words(t)) <= 3:
            nxt = next((lines[k] for k in range(ln + 1, len(lines)) if lines[k].strip()), "")
            ids = {"".join(words(i)) for i in re.findall(r"[A-Za-z_]\w*", nxt)}
            if "".join(words(t)) in ids:
                rule = "repeats the name below"
        if rule:
            found.append(f"{path}:{ln + 1}: {rule}: {text.split(chr(10))[0].strip()[:100]}")
    return found


if __name__ == "__main__":
    files = sys.argv[1:] or [f for f in subprocess.run(["git", "-C", ROOT, "ls-files", "--cached", "--others", "--exclude-standard"],
                                                     capture_output=True, text=True, check=True).stdout.split()
                             if f.endswith(SOURCES) or os.path.basename(f) == "Makefile"]
    found = [x for f in files for x in lint(f)]
    print("\n".join(found) if found else f"comments ok ({len(files)} files)")
    sys.exit(1 if found else 0)
