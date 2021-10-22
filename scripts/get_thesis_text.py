#!/usr/bin/env python3

from pylatexenc.latex2text import LatexNodes2Text
import os


def get_thesis_text(path=os.path.join(os.path.dirname(__file__), '..',
                                      'content', 'chapters')):
    for root, dirs, files in sorted(os.walk(path, topdown=False)):
        for name in files:
            file = os.path.abspath(os.path.join(root, name))
            with open(file, 'r') as f:
                text = LatexNodes2Text(math_mode="remove").latex_to_text(
                        f.read())
                yield (file, text)

if __name__ == "__main__":
    print(list(get_thesis_text()))

