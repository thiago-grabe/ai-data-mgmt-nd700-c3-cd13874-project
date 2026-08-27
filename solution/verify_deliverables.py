"""Acceptance gate: every rubric artifact exists, is substantive, and is committed.

Complements tests/test_rubric.py, which checks behaviour. This checks the
deliverables — including the one thing a test in the working tree cannot see:
whether the evidence is actually tracked by git.

    python verify_deliverables.py

Exits non-zero on any failure.
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

SOLUTION = Path(__file__).resolve().parent
STARTER = SOLUTION.parent / "starter"
EVIDENCE = SOLUTION / "docs" / "evaluation"

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

checks = []


def check(label, fn):
    try:
        detail = fn()
        checks.append((label, True, detail))
    except Exception as e:
        checks.append((label, False, f"{type(e).__name__}: {e}"))


def tracked_by_git(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=SOLUTION, capture_output=True, text=True, check=False,
    )
    return result.returncode == 0


# ---------------------------------------------------------------- deliverables
REQUIRED = [
    "chat.py",
    "architecture.md",
    "README.md",
    "agents/structured_data_agent.py",
    "agents/unstructured_data_agent.py",
    "agents/multimodal_data_agent.py",
    "agents/timeseries_data_agent.py",
    "audit.py",
    "router_llm.py",
    "evaluate.py",
    "requirements.txt",
    "constraints.txt",
    "docs/AZURE_SETUP.md",
    "docs/RUBRIC.md",
    "docs/DEVIATIONS.md",
    "docs/EVALUATION.md",
    "scripts/provision_azure.sh",
    "scripts/seed_data.sh",
    "scripts/verify_azure.py",
    "scripts/teardown_azure.sh",
    "tests/test_rubric.py",
    "tests/conftest.py",
]


def _deliverables():
    missing = [r for r in REQUIRED if not (SOLUTION / r).exists()]
    if missing:
        raise AssertionError(f"missing: {missing}")
    thin = [r for r in REQUIRED if (SOLUTION / r).stat().st_size < 200]
    if thin:
        raise AssertionError(f"stub files: {thin}")
    return f"{len(REQUIRED)} files present"


check("Required deliverables exist and are substantive", _deliverables)


def _no_todos():
    marker = "<" + "TODO"
    offenders = [
        p.relative_to(SOLUTION)
        for p in SOLUTION.rglob("*.py")
        if ".venv" not in p.parts and "__pycache__" not in p.parts
        and marker in p.read_text(encoding="utf-8", errors="replace")
    ]
    if offenders:
        raise AssertionError(f"unfinished placeholders in {offenders}")
    return "0 placeholders remaining"


check("All TODO placeholders completed", _no_todos)


def _starter_pristine():
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", str(STARTER)],
        cwd=SOLUTION.parent, capture_output=True, text=True, check=False,
    )
    if result.stdout.strip():
        raise AssertionError(f"starter/ has uncommitted changes:\n{result.stdout}")
    return "starter/ unmodified"


check("Starter left untouched", _starter_pristine)


def _data_unchanged():
    for folder in ["structured", "unstructured", "multimodal"]:
        for original in sorted((STARTER / "data" / folder).rglob("*")):
            if not original.is_file() or original.name == ".DS_Store":
                continue
            mirror = SOLUTION / "data" / folder / original.relative_to(
                STARTER / "data" / folder
            )
            if not mirror.exists():
                raise AssertionError(f"missing {mirror}")
            if hashlib.sha256(mirror.read_bytes()).hexdigest() != \
               hashlib.sha256(original.read_bytes()).hexdigest():
                raise AssertionError(f"{mirror} differs from the course original")
    return "3 data folders byte-identical to starter/"


check("Course-provided data unchanged", _data_unchanged)


# ------------------------------------------------------------------- evidence
def _transcripts():
    if not EVIDENCE.exists():
        raise AssertionError("docs/evaluation/ does not exist; run evaluate.py")
    by_route = {}
    for log in EVIDENCE.glob("*.log"):
        route = log.stem.rsplit("-", 1)[0]
        by_route.setdefault(route, []).append(log)
    for route in ["structured", "unstructured", "multimodal", "timeseries"]:
        if len(by_route.get(route, [])) < 2:
            raise AssertionError(
                f"the rubric requires >=2 transcripts for {route}, "
                f"found {len(by_route.get(route, []))}"
            )
    total = sum(len(v) for v in by_route.values())
    return f"{total} transcripts, >=2 per agent"


check("Evaluation transcripts cover every agent", _transcripts)


def _safeguard_evidence():
    def text_for(route):
        return "\n".join(p.read_text() for p in EVIDENCE.glob(f"{route}-*.log"))

    structured = text_for("structured")
    for needle in ["Selection Rate by Group", "Demographic Parity Difference"]:
        if needle not in structured:
            raise AssertionError(f"structured transcripts missing {needle!r}")

    unstructured = text_for("unstructured")
    if not re.search(r"confidence=\d\.\d\d", unstructured):
        raise AssertionError("unstructured transcripts lack PII confidence scores")

    multimodal = text_for("multimodal")
    for category in ["Hate", "SelfHarm", "Sexual", "Violence"]:
        if category not in multimodal:
            raise AssertionError(f"multimodal transcripts missing {category} severity")

    timeseries = text_for("timeseries")
    if "Anomaly rule" not in timeseries:
        raise AssertionError("timeseries transcripts lack anomaly output")

    return "Fairlearn, Presidio, Content Safety and anomaly output all present"


check("Ethical-check output captured for every agent", _safeguard_evidence)


def _cli_starts():
    """R2.3 — the offline suite cannot cover this; it patches the agent constructors."""
    log = EVIDENCE / "cli_startup.log"
    if not log.exists():
        raise AssertionError("no cli_startup.log; run scripts/smoke_cli.sh")
    text = log.read_text()
    if "Agent Chat Ready" not in text:
        raise AssertionError("chat.py did not reach the ready banner")
    if "Traceback" in text:
        raise AssertionError("chat.py raised during startup")
    loaded = len(re.findall(r"^Loading .* Data Agent", text, re.MULTILINE))
    if loaded != 4:
        raise AssertionError(f"expected 4 agents to load, transcript shows {loaded}")
    return "chat.py loads 4 agents and reaches the prompt"


check("chat.py starts against live Azure", _cli_starts)


def _figures():
    pngs = list(EVIDENCE.glob("*.png"))
    if not pngs:
        raise AssertionError("no multimodal similarity figures captured")
    return f"{len(pngs)} PNG figures"


check("Multimodal figures captured", _figures)


def _evidence_is_committed():
    """The root .gitignore has a blanket *.log that would silently drop all of this."""
    untracked = [
        p.name for p in list(EVIDENCE.glob("*.log")) + list(EVIDENCE.glob("*.png"))
        if not tracked_by_git(p)
    ]
    if untracked:
        raise AssertionError(
            f"evidence present on disk but NOT tracked by git: {untracked}. "
            "Check the *.log rule in the root .gitignore."
        )
    return "all evidence tracked by git"


check("Evidence is committed, not just present", _evidence_is_committed)


# -------------------------------------------------------------------- secrets
SECRET_PATTERNS = [
    (re.compile(r"AccountKey=[A-Za-z0-9+/]{20,}"), "storage account key"),
    (re.compile(r"mongodb://[^\"'\s]*:[^\"'\s@]{8,}@"), "mongo connection string"),
    (re.compile(r"""password\s*=\s*["'][^"'{}$]{8,}["']"""), "hardcoded password"),
]
ALLOWED = ("<", "fake-", "example", "AccountKey=aaaa", "AccountKey=bbbb", "acct:key")


def _no_secrets():
    offenders = []
    for path in SOLUTION.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".md", ".txt", ".ini", ".toml", ".log", ".jsonl"}:
            continue
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern, label in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                if any(token in match.group(0) for token in ALLOWED):
                    continue
                offenders.append(f"{path.relative_to(SOLUTION)}: {label}")
    if offenders:
        raise AssertionError(f"possible secrets in committed files: {offenders}")
    return "no secret-shaped strings in any committed file"


check("No secrets in committed files", _no_secrets)


def _env_ignored():
    for name in [".pg_password", ".env"]:
        path = SOLUTION / name
        if path.exists() and tracked_by_git(path):
            raise AssertionError(f"{name} exists and is tracked by git")
    return ".pg_password / .env not tracked"


check("Local credential files are git-ignored", _env_ignored)


# ---------------------------------------------------------------------- suite
def _pytest():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=SOLUTION, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout.strip().splitlines()[-1])
    summary = [ln for ln in result.stdout.splitlines() if "passed" in ln]
    return summary[-1] if summary else "passed"


check("Rubric conformance suite passes", _pytest)


# -------------------------------------------------------------------- report
print("\n" + "=" * 72)
print("DELIVERABLE VERIFICATION")
print("=" * 72)
for label, passed, detail in checks:
    mark = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"{mark}  {label}")
    print(f"        {detail}")

failed = [c for c in checks if not c[1]]
print("=" * 72)
if failed:
    print(f"{RED}{len(failed)} of {len(checks)} checks failed.{RESET}")
    sys.exit(1)
print(f"{GREEN}All {len(checks)} checks passed.{RESET}")
