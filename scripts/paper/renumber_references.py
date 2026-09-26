"""Renumber the draft's references by order of first citation.

Citations are "[12]", ranges "[7]–[10]", and placeholders "[NEW_<KEY>]" for entries
added while editing; every placeholder must also start a line in the References list.
The body (everything before "## References") is rewritten with the new numbers and the
list is reordered. Uncited entries are reported and kept at the end.

Usage: python scripts/paper/renumber_references.py paper_evidence/paper_draft_KSII_TIIS_ko.md [--check]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CITE = re.compile(r"\[(\d+|NEW_[A-Z0-9_]+)\](?:–\[(\d+)\])?")
ENTRY = re.compile(r"^\[(\d+|NEW_[A-Z0-9_]+)\] ", re.M)


def keys_in_order(body: str) -> list[str]:
    order: list[str] = []
    for m in CITE.finditer(body):
        first, last = m.group(1), m.group(2)
        keys = [str(k) for k in range(int(first), int(last) + 1)] if last else [first]
        order += [k for k in keys if k not in order]
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("draft", type=Path)
    parser.add_argument("--check", action="store_true", help="only report; exit 1 if numbering is not in order")
    args = parser.parse_args()
    text = args.draft.read_text(encoding="utf-8")
    head, sep, refs = text.partition("## References")
    if not sep:
        sys.exit("no '## References' heading")
    starts = list(ENTRY.finditer(refs))
    entries = {m.group(1): refs[m.end():(starts[i + 1].start() if i + 1 < len(starts) else len(refs))].strip()
               for i, m in enumerate(starts)}
    order = keys_in_order(head)
    missing = [k for k in order if k not in entries]
    if missing:
        sys.exit(f"cited but not in the list: {missing}")
    uncited = [k for k in entries if k not in order]
    mapping = {k: str(i + 1) for i, k in enumerate(order + uncited)}
    in_order = all(mapping[k] == k for k in entries)
    print(f"{len(order)} cited, {len(uncited)} uncited {uncited}; " + ("already in order" if in_order else "renumbering"))
    if args.check:
        sys.exit(0 if in_order and not uncited else 1)

    def replace(m: re.Match) -> str:
        first, last = m.group(1), m.group(2)
        if not last:
            return f"[{mapping[first]}]"
        new = sorted(int(mapping[str(k)]) for k in range(int(first), int(last) + 1))
        if new == list(range(new[0], new[-1] + 1)):
            return f"[{new[0]}]–[{new[-1]}]"
        return ", ".join(f"[{n}]" for n in new)

    body = CITE.sub(replace, head)
    listing = "\n\n".join(f"[{mapping[k]}] {entries[k]}" for k in order + uncited)
    args.draft.write_text(body + sep + "\n\n" + listing + "\n", encoding="utf-8", newline="\n")
    for k in order + uncited:
        if mapping[k] != k:
            print(f"  [{k}] -> [{mapping[k]}]")


if __name__ == "__main__":
    main()
