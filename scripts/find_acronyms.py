#!/usr/bin/env python3

from get_thesis_text import get_thesis_text
import os
import re


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
    acronyms = read_acronyms()
    for file, content in get_thesis_text():
        found_match = False
        for acronym in acronyms:
            for suffix in ["", "s"]:
                search = acronym + suffix
                for match in re.findall(f"(.*\\b(?:{search})\\b.*)", content):
                    s = str(match).strip()
                    if s.startswith("§"):
                        continue
#                    if re.match(f"\\[[^]]*{search}.*]", s):
#                        continue
                    if not found_match:
                        print(f"\033[37;1m{file}\033[0m")
                        found_match = True
                    print("\t", match.replace(search, f"\033[31;1m{search}\033[31;0m"))

        if found_match:
            print()


