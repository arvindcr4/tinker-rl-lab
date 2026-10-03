"""Public-only command line for parser diagnostics and binomial arithmetic."""
import argparse
import json
import sys

from . import parser_v1, parser_v2, parser_v3
from .selection import interval


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    parser = commands.add_parser("parse", help="Parse supplied text; use - for standard input")
    parser.add_argument("text")
    parser.add_argument("--policy", type=int, choices=(1, 2, 3), default=2)
    counts = commands.add_parser("counts", help="Compute a selected-cohort binomial interval")
    counts.add_argument("events", type=int)
    counts.add_argument("denominator", type=int)
    args = cli.parse_args(argv)
    if args.command == "parse":
        text = sys.stdin.read() if args.text == "-" else args.text
        module = {1: parser_v1, 2: parser_v2, 3: parser_v3}[args.policy]
        result = module.parse_answer(parser_v1.strip_terminal_tokens(text))
    else:
        try:
            result = interval(args.events, args.denominator)
        except ValueError as exc:
            cli.error(str(exc))
    print(json.dumps({"label": "ARITHMETIC_OR_PARSER_DIAGNOSTIC_ONLY", "result": result},
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
