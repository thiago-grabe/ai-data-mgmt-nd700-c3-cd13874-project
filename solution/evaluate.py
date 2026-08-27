"""Evaluation harness: run every rubric prompt against live Azure and capture the evidence.

    python evaluate.py                # all prompts
    python evaluate.py --route structured

Produces, under docs/evaluation/:
  <route>-<n>.log            full terminal transcript per prompt, including the
                             ethical-check output the rubric asks to see
  multimodal_matches_<n>.png the similarity figure, saved rather than shown

Two things this handles that an interactive run does not:

1. matplotlib. show_results ends in plt.show(), which blocks on a GUI window.
   The Agg backend plus figure_dir turns that into a committed PNG.
2. Secret leakage. A psycopg or pymongo traceback prints the full connection
   string. Everything written to the transcript passes through a redactor first.
"""

import argparse
import io
import sys
import time
from pathlib import Path

import matplotlib

# Must precede any import that pulls in pyplot, i.e. before `import chat`.
matplotlib.use("Agg")

SOLUTION = Path(__file__).resolve().parent
EVIDENCE = SOLUTION / "docs" / "evaluation"

# (prompt, expected_route). Two per agent minimum, as the rubric requires.
# The first six are the project brief's own example queries verbatim.
PROMPTS = [
    ("Which demographic group has the most expensive homes?", "structured"),
    ("How many homes are owned by hispanics in Ashford neighborhood?", "structured"),
    ("Was a permit approved for a restaurant in any of the neighborhoods?", "unstructured"),
    ("Show me all the permits for pools in each neighborhood", "unstructured"),
    ("Find me a house like mine", "multimodal"),
    ("Show me houses similar to my house", "multimodal"),
    ("Are there any unusual air quality readings in Maplewood?", "timeseries"),
    ("What is the average noise level per neighborhood?", "timeseries"),
]


class Redactor(io.TextIOBase):
    """Mask secret values in anything written to the transcript.

    A traceback from a database driver will happily print the whole connection
    string, and these transcripts get committed to the repository.
    """

    def __init__(self, wrapped, secrets):
        self.wrapped = wrapped
        # Longest first, so a full connection string is masked before the account
        # key embedded inside it.
        self.secrets = sorted(
            {s for s in secrets if s and len(s) > 7}, key=len, reverse=True
        )

    def write(self, text):
        for secret in self.secrets:
            text = text.replace(secret, "***REDACTED***")
        return self.wrapped.write(text)

    def flush(self):
        self.wrapped.flush()


class Tee(io.TextIOBase):
    """Write to the console and the per-prompt transcript at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", help="only run prompts for this agent")
    args = parser.parse_args()

    EVIDENCE.mkdir(parents=True, exist_ok=True)

    print("Importing chat.py (this authenticates to Key Vault once for the whole run)...")
    import chat

    secrets = [
        chat.structuredazureapikey,
        chat.structuredpostgresqlpassword,
        chat.unstructuredazurekey,
        chat.unstructuredmongourl,
        chat.multimodalazureconnstring,
        chat.multimodalazurecontentsafetykey,
        chat.timeseriesazureconnstring,
    ]

    manager = chat.AgentChatManager(figure_dir=str(EVIDENCE))

    selected = [p for p in PROMPTS if not args.route or p[1] == args.route]
    counters = {}
    failures = []

    for prompt, expected_route in selected:
        counters[expected_route] = counters.get(expected_route, 0) + 1
        log_path = EVIDENCE / f"{expected_route}-{counters[expected_route]}.log"

        real_stdout = sys.stdout
        with log_path.open("w", encoding="utf-8") as fh:
            redacted = Redactor(Tee(real_stdout, fh), secrets)
            sys.stdout = redacted
            sys.stderr = redacted
            try:
                print("=" * 80)
                print(f"PROMPT: {prompt}")
                print(f"EXPECTED ROUTE: {expected_route}")
                print("=" * 80)

                start = time.time()
                answer = manager.chat(prompt)
                elapsed = time.time() - start

                print("\n" + "-" * 80)
                print("ANSWER")
                print("-" * 80)
                print(answer)
                print(f"\n[completed in {elapsed:.1f}s]")

                if answer.startswith("An error occurred"):
                    failures.append((prompt, answer))
            except Exception as e:
                print(f"\nUNHANDLED ERROR: {type(e).__name__}: {e}")
                failures.append((prompt, str(e)))
            finally:
                sys.stdout = real_stdout
                sys.stderr = sys.__stderr__

        print(f"  -> {log_path.relative_to(SOLUTION)}")

        # Content Safety F0 allows 5 requests/second and one multimodal prompt
        # fires a dozen; space the calls out.
        if expected_route == "multimodal":
            time.sleep(2)

    print("\n" + "=" * 80)
    print(f"Ran {len(selected)} prompts. Transcripts in {EVIDENCE.relative_to(SOLUTION)}/")
    if failures:
        print(f"{len(failures)} prompt(s) errored:")
        for prompt, err in failures:
            print(f"  - {prompt}: {err[:160]}")
        sys.exit(1)
    print("All prompts completed without error.")


if __name__ == "__main__":
    main()
