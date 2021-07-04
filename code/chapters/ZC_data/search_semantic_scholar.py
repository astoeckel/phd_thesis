#!/usr/bin/env python3

import argparse
import glob
import gzip
import json
import multiprocessing
import os
import re
import sys
import tqdm

parser = argparse.ArgumentParser(
    description=
    'Search in the semantic scholar database. Produces a JSON file containing publications with titles and abstracts matching certain phrases.'
)
parser.add_argument('--dir',
                    type=str,
                    default=".",
                    help='Directory containing s2-corpus-*.gz files')
parser.add_argument(
    'phrases',
    type=str,
    nargs='+',
    help='One or multiple phrases (joined via OR) to search for')

args = parser.parse_args()

files = sorted(glob.glob(os.path.join(glob.escape(args.dir), 's2-corpus-*.gz')))

# Construct a regular expression matching for the phrases
re_str = "(\\b" + "\\b|\\b".join(map(re.escape, args.phrases)) + "\\b)"
re_cmp = re.compile(re_str, re.IGNORECASE)

def process_file(file):
    matches = []
    with gzip.open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            for field in ["title", "abstract", "paperAbstract"]:
                if (field in data) and (re_cmp.search(data[field])):
                    matches.append(data)
                    break
    return matches

sys.stdout.write('[\n\t')
first = True
with multiprocessing.Pool() as pool:
    for matches in tqdm.tqdm(pool.imap_unordered(process_file, files), total=len(files)):
        for match in matches:
            if not first:
                sys.stdout.write(',\n\t')
            json.dump(match, sys.stdout)
            first = False
        sys.stdout.flush()
sys.stdout.write('\n]')
