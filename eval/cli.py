"""CLI entry point for eval harness."""
from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _retrieval() -> None:
    from eval.runs.retrieval_grid import run

    out = run()
    print(f"Retrieval grid done -> {out}")


def _llm() -> None:
    from eval.runs.llm_grid import run

    out = run()
    print(f"LLM grid done -> {out}")


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python3 -m eval.cli [retrieval|llm|all]")
        sys.exit(1)

    cmd = args[0]
    if cmd == "retrieval":
        _retrieval()
    elif cmd == "llm":
        _llm()
    elif cmd == "all":
        _retrieval()
        _llm()
    else:
        print(f"Unknown command: {cmd}. Use retrieval, llm, or all.")
        sys.exit(1)


if __name__ == "__main__":
    main()
