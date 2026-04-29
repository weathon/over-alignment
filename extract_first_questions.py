"""Extract the first user message from each chat in full_chat.txt, deduplicated.

The export format alternates implicit "turns":

    <chat title>
    user
    <user message lines...>
    ChatGPT
    <assistant response lines...>
    user
    <user message lines...>
    ChatGPT
    ...
    <next chat title>
    user
    ...

The format is ambiguous in principle: when scanning sequentially, a non-role
line immediately preceding a `user` marker can be either (a) the tail of the
previous assistant response (same chat continues) or (b) the title of the
next chat. We disambiguate with a title-shaped heuristic on that line:
short, no terminal sentence punctuation, not a markdown bullet/heading.

This is good enough for a first pass; tune the heuristic if a sample of
boundaries is misclassified.
"""

import json
import re
from pathlib import Path

SRC = Path(__file__).parent / "full_chat.txt"
OUT = Path(__file__).parent / "all_chats.json"

SENTENCE_END = tuple(".!?,;:”\"')]}")
MAX_TITLE_LEN = 100
BULLET_RE = re.compile(r"^\s*([-*#>]|\d+[.)])\s")


def looks_like_title(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > MAX_TITLE_LEN:
        return False
    if s.endswith(SENTENCE_END):
        return False
    if BULLET_RE.match(s):
        return False
    return True


def extract_first_questions(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    n = len(lines)
    first_questions: list[str] = []
    i = 0

    # Find the first non-blank line; it's the title of the first chat.
    while i < n and lines[i].strip() == "":
        i += 1
    if i >= n:
        return first_questions
    # Skip the title.
    i += 1

    in_chat = True  # we're inside a chat, expecting "user" next

    while i < n:
        # Skip blanks.
        while i < n and lines[i].strip() == "":
            i += 1
        if i >= n:
            break

        if in_chat:
            # Expect "user" marker.
            if lines[i] == "user":
                i += 1
                # Collect first user message.
                msg: list[str] = []
                while i < n and lines[i] != "ChatGPT":
                    msg.append(lines[i])
                    i += 1
                while msg and msg[0].strip() == "":
                    msg.pop(0)
                while msg and msg[-1].strip() == "":
                    msg.pop()
                first_questions.append("\n".join(msg))
                # Skip the "ChatGPT" marker.
                if i < n and lines[i] == "ChatGPT":
                    i += 1
                in_chat = False  # now scanning assistant response
            else:
                # Should not happen if the file is well-formed; advance.
                i += 1
        else:
            # We're scanning the assistant response. Walk forward until we
            # see a "user" marker. The line(s) just before that marker may
            # be either tail-of-response (same chat) or a chat title (new
            # chat). Use the title heuristic to decide.
            last_nonblank_before_user: tuple[int, str] | None = None
            while i < n:
                line = lines[i]
                if line == "user":
                    break
                if line.strip() != "":
                    last_nonblank_before_user = (i, line)
                i += 1
            if i >= n:
                break  # EOF; no more chats
            # i points to "user". Decide if last_nonblank is a title.
            is_new_chat = last_nonblank_before_user is not None and looks_like_title(
                last_nonblank_before_user[1]
            )
            in_chat = True
            if not is_new_chat:
                # Continuation of same chat: skip "user", user-message, "ChatGPT".
                i += 1  # past "user"
                while i < n and lines[i] != "ChatGPT":
                    i += 1
                if i < n and lines[i] == "ChatGPT":
                    i += 1
                in_chat = False  # still scanning assistant response of same chat

    return first_questions


def dedupe(questions: list[str]) -> list[str]:
    """Drop duplicates while preserving first-occurrence order. Compares on
    whitespace-normalized, case-folded text so trivial variations collapse."""
    seen: set[str] = set()
    out: list[str] = []
    for q in questions:
        key = " ".join(q.split()).casefold()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def main() -> None:
    questions = extract_first_questions(SRC)
    deduped = dedupe(questions)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)
    print(
        f"Wrote {len(deduped)} first-questions to {OUT} "
        f"(deduped from {len(questions)})"
    )


if __name__ == "__main__":
    main()
