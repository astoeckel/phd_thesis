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

###############################################################################
# Parse and check command line arguments                                      #
###############################################################################

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
    nargs='*',
    help='One or multiple phrases (joined via OR) to search for.')
parser.add_argument(
    '--cites',
    type=str,
    nargs='*',
    help=
    'Include papers that cite ONE of the given papers (specify a list of S2 paper IDs). This is combined via AND with the phrases.',
)

args = parser.parse_args()

has_phrases = len(args.phrases) > 0
has_cites = (not args.cites is None) and (len(args.cites) > 0)

if not (has_phrases or has_cites):
    parser.print_help()
    print()
    print("Error: Must either specify phrases or citations to search for")
    sys.exit(1)

# Make sure the paper IDs are valid
if has_cites:
    for citation in args.cites:
        if not re.match("^[a-f0-9]{40}$", citation):
            print(f"Error: \"{citation}\" is not a 40-character hex string")
            sys.exit(1)

# Find the S2 ORC files
files = sorted(glob.glob(os.path.join(glob.escape(args.dir),
                                      's2-corpus-*.gz')))
if len(files) == 0:
    print(
        "Error: No semantic scholar corpus files (s2-corpus-*.gz) files found at",
        args.dir)
    sys.exit(1)

###############################################################################
# Process data                                                                #
###############################################################################

# Construct a regular expression matching for the phrases
re_str = "(\\b" + "\\b|\\b".join(map(re.escape, args.phrases)) + "\\b)"
re_cmp = re.compile(re_str, re.IGNORECASE)

def process_file(file):
    matches = []
    with gzip.open(file, 'r') as f:
        for line in f:
            data = json.loads(line)

            # Check phrases
            if has_phrases:
                match = False
                for field in ["title", "abstract", "paperAbstract"]:
                    if (field in data) and (re_cmp.search(data[field])):
                        match = True
                        break
            else:
                match = True

            # Check citations
            if has_cites and ("outCitations" in data):
                found = False
                for citation in args.cites:
                    if citation in data["outCitations"]:
                        found = True
                        break
                match = match and found

            if match:
                matches.append(data)
    return matches


sys.stdout.write('[\n\t')
first = True
with multiprocessing.Pool() as pool:
    for matches in tqdm.tqdm(pool.imap_unordered(process_file, files),
                             total=len(files)):
        for match in matches:
            if not first:
                sys.stdout.write(',\n\t')
            json.dump(match, sys.stdout)
            first = False
        sys.stdout.flush()
sys.stdout.write('\n]')

