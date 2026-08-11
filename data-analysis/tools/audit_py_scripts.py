#!/usr/bin/env python3
import ast
from pathlib import Path

ROOT = Path(".").resolve()
PATH_EXTS = {
    ".csv", ".json", ".txt", ".tsv", ".html", ".png", ".pdf",
    ".md", ".npy", ".npz", ".yaml", ".yml", ".pkl", ".parquet"
}
OLD_HINTS = ("analyse-confiance", "analyse-consensus", "analyse-entropie",
             "computational_model", "full_log_", "benchmark_")
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache"}

def is_path_like(s: str) -> bool:
    return ("/" in s or "\\" in s or s.startswith((".", "/"))
            or any(s.lower().endswith(ext) for ext in PATH_EXTS))

def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{call_name(node.value)}.{node.attr}" if call_name(node.value) else node.attr
    return ""

def summarize_script(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return {"syntax_error": str(e), "paths": [], "funcs": [], "doc": None}

    doc = ast.get_docstring(tree)
    funcs = []
    paths = []  # (line, call, value, status)

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            funcs.append(node.name)
            self.generic_visit(node)

        def visit_Call(self, node):
            cal = call_name(node.func)
            checked = set()

            # 1er arg + keyword classiques
            args = list(node.args)
            for i, a in enumerate(args[:2]):
                self.check_arg(a, node.lineno, cal, slot=f"arg{i+1}")

            for kw in node.keywords:
                if kw.arg in {"file", "filename", "filepath_or_buffer", "path", "path_or_buf",
                              "fname", "cache_file", "log_file", "data", "output", "out_file"}:
                    self.check_arg(kw.value, node.lineno, cal, slot=f"kw:{kw.arg}")
            self.generic_visit(node)

        def check_arg(self, a, line, cal, slot=""):
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                s = a.value
                if is_path_like(s):
                    if Path(s).is_absolute():
                        exists = Path(s).exists()
                    else:
                        exists = (path.parent / s).exists() or (ROOT / s).exists()
                    status = "OK" if exists else "MISSING"
                    if any(h in s for h in OLD_HINTS):
                        status += " | OLD_TOKEN"
                    paths.append((line, cal, slot, s, status))
            elif isinstance(a, ast.JoinedStr):
                paths.append((line, cal, slot, "<f-string dynamique>", "CHECK_MANU"))

    V().visit(tree)
    return {"syntax_error": None, "paths": paths, "funcs": funcs, "doc": doc}

def py_files():
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        # ajuste si tu veux exclure aussi 'build'/'dist'
        yield p

total = 0
bad = 0
for f in sorted(py_files()):
    rel = f.relative_to(ROOT)
    r = summarize_script(f)
    total += 1
    print(f"\n=== {rel} ===")
    if r["syntax_error"]:
        print(f"  SYNTAX ERROR: {r['syntax_error']}")
        continue

    doc = (r["doc"].splitlines()[0] if r["doc"] else "—")
    print(f"  DOC: {doc}")
    print(f"  fonctions: {', '.join(r['funcs']) if r['funcs'] else '—'}")

    if not r["paths"]:
        print("  paths détectés: AUCUN")
        continue

    print("  paths détectés:")
    for line, cal, slot, val, status in r["paths"]:
        marker = "⚠" if ("MISSING" in status or "OLD_TOKEN" in status or "CHECK_MANU" in status) else "✓"
        print(f"    {marker} L{line:4} [{slot}] {cal}: {val} -> {status}")
        if marker == "⚠":
            bad += 1

print(f"\nRAPPEL: scripts analysés = {total}, alertes = {bad}")
