#!/usr/bin/env python3

from get_thesis_text import get_thesis_text
import os
import re

import sys

def read_acronyms(file=os.path.join(os.path.dirname(__file__), '..', 'content',
                                    'frontmatter', 'acronyms.tex')):

    pattern = re.compile(R"\\NewAcronym{([^}]*)}{([^}]*)}{([^}]*)}")

    acronyms = []

    with open(file, 'r') as f:
        content = f.read()

        for match in pattern.findall(content):
            acronyms.append(match[1])

    return acronyms

if __name__ == "__main__":
    replacements = {}
    acronyms = read_acronyms()
    for file, content in get_thesis_text():
        replacements[file] = []
        found_match = False
        for acronym in acronyms:
            for suffix in ["", "s"]:
                search = acronym + suffix
                for match in re.finditer(f"(.*\\b((?:{search}))\\b.*)", content):
                    si = match.start(2) - match.start(1)
                    match_str = match.group(0)
                    s = match_str.strip()
                    if s.startswith("§") or s.startswith("CHAPTER:"):
                        continue
                    if si > 0:
                        if match_str[si - 1] == "-":
                            continue
                        brck_cnt = 0
                        while si >= 0:
                            if match_str[si] == "]":
                                brck_cnt -= 1
                            elif match_str[si] == "[":
                                brck_cnt += 1
                            si -= 1
                        if brck_cnt > 0:
                            continue
                    if not found_match:
                        print(f"\033[37;1m{file}\033[0m")
                        found_match = True
                    print("\t", match_str.replace(search, f"\033[31;1m{search}\033[31;0m"))

                    # Prepare an automatic replacement
                    if match_str.strip() and match_str.strip()[0] == "[":
                        continue

                    si = match.start(2) - match.start(1)
                    sj = match.end(2) - match.start(1)
                    s0 = max(0, si - 15)
                    s1 = min(len(match_str) - 1, si + 15)
                    src = match_str[s0:s1].strip()
                    tar = (match_str[s0:si] + "\\" + acronym + ("pl" if suffix == "s" else "") + match_str[sj:s1]).strip()
                    if len(src) < sj - si + 15:
                        continue
                    replacements[file].append((src.encode("utf-8"), tar.encode("utf-8")))

        if found_match:
            print()

    if len(sys.argv) == 2 and sys.argv[1]  == "--auto":
        for file in replacements.keys():
            if len(replacements[file]) == 0:
                continue

            print(f"Processing {file}...")
            with open(file, "rb") as f:
                content = f.read()
            for src, tar in replacements[file]:
                content = content.replace(src, tar)
            with open(file, "wb") as f:
                f.write(content)

