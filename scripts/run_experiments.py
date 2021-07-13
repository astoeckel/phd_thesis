#!/usr/bin/env python3

#   Code for the PhD Thesis
#   "Harnessing Neural Dynamics as a Computational Resource"
#   Copyright (C) 2021  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys, os
import toml
import run_in_container
import tempfile
import multiprocessing
import hashlib
import re
import json

import logging

logger = logging.getLogger(__name__)


def _parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs one or all of the experiments listed in the manifest'
    )
    parser.add_argument(
        '--manifest',
        default="~/code/Manifest.toml",
        help=
        "Manifest file describing all experiments. ~/ is extended to the Git root."
    )
    parser.add_argument('--list',
                        action='store_true',
                        help="Checks whether the output files exist")
    parser.add_argument('--prune',
                        action='store_true',
                        help="Deletes no longer required output files")
    parser.add_argument('--dry',
                        action='store_true',
                        help="Do not perform any actions")
    parser.add_argument(
        'filter',
        nargs='*',
        help=
        "Set of regular expressions applied to the code file names (multiple filters are combined via OR)"
    )

    return parser.parse_args(argv[1:])


def _elaborate_manifest(manifest, manifest_file, root):
    # Elaborate all "generates" sections into "runs"
    for file, obj in manifest.items():
        if ("generates" in obj) == ("runs" in obj):
            raise RuntimeError(
                f"Either \"generates\" or \"runs\" must be set in entry \"{file}\""
            )

        if "generates" in obj:
            obj["runs"] = [{"args": [], "generates": obj["generates"]}]

        if not "is_multithreaded" in obj:
            obj["is_multithreaded"] = False
        obj["is_multithreaded"] = bool(obj["is_multithreaded"])

    # Elaborate all "args" list by prepending adding the complete command
    manifest_path = os.path.abspath(os.path.dirname(manifest_file))
    manifest_path_rel = os.path.relpath(manifest_path, root)
    for file, obj in manifest.items():
        if not os.path.isfile(os.path.join(manifest_path, file)):
            raise RuntimeError(f"Referenced file \"{file}\" does not exist!")

        for run in obj["runs"]:
            run["args"] = [os.path.join(manifest_path_rel, file), *run["args"]]
            if isinstance(run["generates"], str):
                run["generates"] = [run["generates"]]

    # Merge some required data into the individual runs
    for file, obj in manifest.items():
        for run in obj["runs"]:
            if (not "commit" in run) and ("commit" in obj):
                run["commit"] = obj["commit"]
            if (not "dockerfile" in run) and ("dockerfile" in obj):
                run["dockerfile"] = obj["dockerfile"]

    # Elaborate the commit IDs
    commit_cache = {}
    for file, obj in manifest.items():
        for run in obj["runs"]:
            if not "commit" in run:
                raise RuntimeError(f"Missing \"commit\" in entry \"{file}\"")
            if not "dockerfile" in run:
                raise RuntimeError(f"Missing \"dockerfile\" in entry \"{file}\"")
            if not run["commit"] in commit_cache:
                commit_cache[run["commit"]] = run_in_container.git_rev_parse(root, run["commit"])
            run["commit"] = commit_cache[run["commit"]]


def _load_dockerfiles(args):
    root, commit, files = args
    res = {}
    with tempfile.TemporaryDirectory() as dir_tmp:
        run_in_container.git_clone(root, dir_tmp, commit)

        for file in files:
            file_dockerfile = os.path.join(
                dir_tmp, "docker",
                os.path.basename(file) + ".dockerfile")
            if not os.path.isfile(file_dockerfile):
                raise RuntimeError(
                    f'Docker file "{file_dockerfile}" does not exist in revision "{commit[:8]}"'
                )

            with open(file_dockerfile, 'rb') as f:
                res[file] = f.read()

    return commit, res


def _compute_target_hashes(manifest, root):
    # Fetch all commit-dockerfile pairs
    manifest_by_commits = {}
    for file, obj in manifest.items():
        for run in obj["runs"]:
            if not run["commit"] in manifest_by_commits:
                manifest_by_commits[run["commit"]] = set()
            manifest_by_commits[run["commit"]].add(run["dockerfile"])

    # Load all docker files in parallel
    docker_file_contents = {}
    tasks = [(root, commit, files)
             for commit, files in manifest_by_commits.items()]
    with multiprocessing.Pool() as pool:
        for commit, contents in pool.imap_unordered(_load_dockerfiles, tasks):
            docker_file_contents[commit] = contents

    # Compute the target hashes
    for file, obj in manifest.items():
        for run in obj["runs"]:
            docker_hash = hashlib.sha256()
            docker_hash.update(
                docker_file_contents[run["commit"]][run["dockerfile"]])
            docker_hash = docker_hash.hexdigest()[:16]
            run["docker_hash"] = docker_hash

            target_hash = hashlib.sha256()
            target_hash.update(run["commit"].encode('utf-8'))
            target_hash.update(" ".join(run["args"]).encode("utf-8"))
            target_hash.update(
                docker_file_contents[run["commit"]][run["dockerfile"]])
            target_hash = target_hash.hexdigest()[:16]
            run["target_hash"] = target_hash


def _compute_target_files(manifest, root):
    for file, obj in manifest.items():
        for run in obj["runs"]:
            run["exists"] = {}
            run["generates_orig"] = [None] * len(run["generates"])
            for i, target in enumerate(run["generates"]):
                tar = os.path.join("data", "generated", os.path.dirname(file),
                                   run["target_hash"] + "_" + target)
                run["generates"][i] = tar
                run["generates_orig"][i] = target
                run["exists"][tar] = os.path.isfile(os.path.join(root, tar))


def _print_list(manifest, expr, root):
    from colored import fg, bg, attr
    for file, obj in manifest.items():
        first = True
        for run in obj["runs"]:
            if len(expr.findall(run["args"][0])) == 0:
                continue

            if first:
                sys.stdout.write("{}{}{}\n".format(attr("bold"), file,
                                                   attr("reset")))
                first = False

            sys.stdout.write("{} {} {}\n".format(run["commit"][:8],
                                                 run["dockerfile"],
                                                 " ".join(run["args"])))
            for tar in run["generates"]:
                if run["exists"][tar]:
                    sys.stdout.write("\t{}{}✔{}".format(
                        fg("green"), attr("bold"), attr("reset")))
                else:
                    sys.stdout.write("\t{}{}✘{}".format(
                        fg("red"), attr("bold"), attr("reset")))
                sys.stdout.write("\t{}\n\n".format(tar))

        if not first:
            sys.stdout.write("\n")


def _prune_files(manifest, root, dry=False):
    # Delete all obsolete target files
    files_present = set(
        run_in_container.list_all_files(os.path.join(root, "data",
                                                     "generated")))
    files_desired = set()
    for obj in manifest.values():
        for run in obj["runs"]:
            for tar in run["generates"]:
                files_desired.add(os.path.abspath(os.path.join(root, tar)))
                pass
    for file in (files_present - files_desired):
        if not dry:
            print("Deleting", file)
            os.unlink(file)
        else:
            print("Would delete", file)

    # Delete all obsolete Docker images
    files_present = set(
        run_in_container.list_all_files(os.path.join(root, "docker",
                                                     "images")))
    files_desired = set()
    for obj in manifest.values():
        for run in obj["runs"]:
            files_desired.add(
                os.path.abspath(
                    os.path.join(
                        root, "docker", "images", run["dockerfile"] + "_" +
                        run["docker_hash"] + ".tar")))
            pass
    for file in (files_present - files_desired):
        if file.endswith(".tar"):
            if not dry:
                print("Deleting", file)
                os.unlink(file)
            else:
                print("Would delete", file)


def _run_task(run, dry=False):
    args = [
        "run_in_container.py", "--expected_hash", run["target_hash"],
        run["commit"], run["dockerfile"], "--", *run["args"]
    ]
    print(" ".join(args))
    try:
        if not dry:
            run_in_container.main(args)
    except RuntimeError:
        logger.exception("Error while executing the experiment!")


def load_manifest(manifest_file="~/code/Manifest.toml", root=None):
    # Fetch the git top-level directory
    if root is None:
        root = os.path.abspath(
            run_in_container.git_top_level(os.path.dirname(__file__)))

    # Fetch the manifest file
    if manifest_file.startswith("~/"):
        manifest_file = os.path.abspath(os.path.join(root, manifest_file[2:]))

    # Load the TOML file
    manifest = toml.load(manifest_file)

    # Load and elaborate the manifest
    _elaborate_manifest(manifest, manifest_file, root=root)
    _compute_target_hashes(manifest, root=root)
    _compute_target_files(manifest, root=root)

    return manifest, manifest_file, root


def target_map(manifest, root):
    # List all files in the "data/generated" directory
    try:
        existing_files = run_in_container.list_all_files(os.path.join(root, "data", "generated"))
    except FileExistsError:
        existing_files = []
    existing_files = set(existing_files)

    # For each suffix in the set of existing files, list the newest file
    prefix_re = re.compile("([0-9a-fA-F]*_)?([^/]*)$")
    newest_files = {}
    for file in existing_files:
        mtime = os.stat(file).st_mtime
        match = prefix_re.match(os.path.basename(file))
        if match:
            prefix, suffix = match[1], match[2]
            if not (suffix in newest_files) or (newest_files[suffix][1] > mtime):
                newest_files[suffix] = (file, mtime)

    # Get the desired target files
    mapped_files = {}
    for obj in manifest.values():
        for run in obj["runs"]:
            for i, tar in enumerate(run["generates"]):
                mapped_files[run["generates_orig"][i]] = os.path.abspath(os.path.join(root, tar))

    # Potentially look for older files
    for suffix, file in list(mapped_files.items()):
        if not file in existing_files:
            if suffix in newest_files:
                mapped_files[suffix] = newest_files[suffix][0]
                logger.warning("Using outdated version of file \"%s\"", suffix)
            else:
                del mapped_files[suffix]
                logger.warning("File \"%s\" does not exist", suffix)

    return mapped_files

def _gather_runs(manifest, expr):
    # Gather experiments to run. Bin experiments into multithreaded and not
    # multithreaded experiments.
    bin_run_concurrently = []
    bin_run_serially = []
    for obj in manifest.values():
        for run in obj["runs"]:
            # Sort the experiment into the concurrent or serial bin
            if len(expr.findall(run["args"][0])):
                run = json.loads(json.dumps(run))
                if obj["is_multithreaded"]:
                    bin_run_serially.append(run)
                else:
                    bin_run_concurrently.append(run)

    return bin_run_concurrently, bin_run_serially


def _filter_runs(bin_run_concurrently, bin_run_serially):
    def required(run):
        for tar in run["generates"]:
            if not run["exists"][tar]:
                return True
        return False

    bin_run_concurrently = list(filter(required, bin_run_concurrently))
    bin_run_serially = list(filter(required, bin_run_serially))
    return bin_run_concurrently, bin_run_serially


def main(argv):
    args = _parse_args(argv)
    if sum((args.list, args.prune)) > 1:
        raise RuntimeError("Conflicting modes of operation!")

    # Load the manifest
    manifest, manifest_file, root = load_manifest(args.manifest)

    # Assemble a regular expression from the given filters
    if len(args.filter) == 0:
        expr = ""
    else:
        expr = "(" + "|".join(re.escape(x) for x in args.filter) + ")"
    expr = re.compile(expr)

    # If so desired, print a list of all files
    if args.list:
        _print_list(manifest, expr=expr, root=root)
        return

    # If so desired, prune all unlisted files
    if args.prune:
        _prune_files(manifest, root=root, dry=args.dry)
        return

    # Gather experiments
    bin_run_concurrently, bin_run_serially = _gather_runs(manifest, expr)
    if len(bin_run_concurrently) == 0 and len(bin_run_serially) == 0:
        print("Nothing matches the given filters!")
        return

    bin_run_concurrently, bin_run_serially = _filter_runs(
        bin_run_concurrently, bin_run_serially)
    if len(bin_run_concurrently) == 0 and len(bin_run_serially) == 0:
        print("Nothing to do!")
        return

    with multiprocessing.Pool() as pool:
        list(pool.imap_unordered(_run_task, bin_run_concurrently))

    for run in bin_run_serially:
        _run_task(run, dry=args.dry)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv)

