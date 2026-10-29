"""Every offline eval in one go: retrieval against the golden set and its baseline, and -- given a file of saved
answers -- the deterministic checks on them. Spends nothing.

    python -m evals.run_evals                                   retrieval
    python -m evals.run_evals evals/reports/answers_<date>.jsonl   retrieval, then the checks on those answers
"""

import sys


def main(argv):
    from evals.generation import deterministic_checks
    from evals.retrieval import evaluate_retrieval
    failed = evaluate_retrieval.main([])
    if argv:
        print(f"\nchecks on {argv[0]}:")
        failed |= deterministic_checks.main(argv[:1])
    return failed


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
