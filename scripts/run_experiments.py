#!/usr/bin/env python3

#   Code for the PhD Thesis
#   "Harnessing Neural Dynamics as a Computational Resource: Building Blocks
#   for Computational Neuroscience and Artificial Agents"
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
    parser.add_argument(
        'filter',
        nargs='*',
        help="Set of regular expressions applied to the code file names (multiple filters are combined via OR)")

    return parser.parse_args(argv[1:])


def _load_manifest(manifest_file, root):
    # Load the TOML file
    manifest = toml.load(manifest_file)

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

    # Elaborate the commit IDs
    for file, obj in manifest.items():
        if not "commit" in obj:
            raise RuntimeError(f"Missing \"commit\" in entry \"{file}\"")
        obj["commit"] = run_in_container.git_rev_parse(root, obj["commit"])

    # Copy some required data across
    for file, obj in manifest.items():
        for run in obj["runs"]:
            run["commit"] = obj["commit"]
            run["dockerfile"] = obj["dockerfile"]

    return manifest


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
        if not "commit" in obj:
            raise RuntimeError(f"Missing \"commit\" in entry \"{file}\"")
        if not "dockerfile" in obj:
            raise RuntimeError(f"Missing \"dockerfile\" in entry \"{file}\"")

        if not obj["commit"] in manifest_by_commits:
            manifest_by_commits[obj["commit"]] = set()
        manifest_by_commits[obj["commit"]].add(obj["dockerfile"])

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
                docker_file_contents[obj["commit"]][obj["dockerfile"]])
            docker_hash = docker_hash.hexdigest()[:16]
            run["docker_hash"] = docker_hash

            target_hash = hashlib.sha256()
            target_hash.update(obj["commit"].encode('utf-8'))
            target_hash.update(" ".join(run["args"]).encode("utf-8"))
            target_hash.update(
                docker_file_contents[obj["commit"]][obj["dockerfile"]])
            target_hash = target_hash.hexdigest()[:16]
            run["target_hash"] = target_hash


def _compute_target_files(manifest, root):
    for file, obj in manifest.items():
        for run in obj["runs"]:
            run["exists"] = {}
            for i, target in enumerate(run["generates"]):
                tar = os.path.join("data", "generated", os.path.dirname(file),
                                   run["target_hash"] + "_" + target)
                run["generates"][i] = tar
                run["exists"][tar] = os.path.isfile(os.path.join(root, tar))


def _print_list(manifest, expr, root):
    from colored import fg, bg, attr
    for file, obj in manifest.items():
        first = True
        for run in obj["runs"]:
            if len(expr.findall(run["args"][0])) == 0:
                continue

            if first:
                sys.stdout.write("{}{}{}\n".format(attr("bold"), file, attr("reset")))
                first = False

            sys.stdout.write("{} {} {}\n".format(obj["commit"][:8],
                                                 obj["dockerfile"],
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


def _run_task(run):
    args = ["run_in_container.py", "--expected_hash", run["target_hash"], run["commit"], run["dockerfile"], "--", *run["args"]]
    print(" ".join(args))
    run_in_container.main(args)

def main(argv):
    args = _parse_args(argv)
    if sum((args.list, args.prune)) > 1:
            raise RuntimeError("Conflicting modes of operation!")

    # Fetch the manifest file
    manifest_file = args.manifest
    if manifest_file.startswith("~/"):
        manifest_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', manifest_file[2:]))

    # Fetch the git top-level directory
    root = os.path.abspath(
        run_in_container.git_top_level(os.path.dirname(manifest_file)))

    # Load and elaborate the manifest
    manifest = _load_manifest(manifest_file, root=root)
    _compute_target_hashes(manifest, root=root)
    _compute_target_files(manifest, root=root)

    # Assemble a regular expression from the given files
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
        # Delete all obsolete target files
        files_present = set(run_in_container.list_all_files(os.path.join(root, "data", "generated")))
        files_desired = set()
        for obj in manifest.values():
            for run in obj["runs"]:
                for tar in run["generates"]:
                    files_desired.add(os.path.abspath(os.path.join(root, tar)))
                    pass
        for file in (files_present - files_desired):
            print("Deleting", file)
            os.unlink(file)

        # Delete all obsolete Docker images
        files_present = set(run_in_container.list_all_files(os.path.join(root, "docker", "images")))
        files_desired = set()
        for obj in manifest.values():
            for run in obj["runs"]:
                files_desired.add(os.path.abspath(os.path.join(root, "docker", "images",
                run["dockerfile"] + "_" + run["docker_hash"] + ".tar")))
                pass
        for file in (files_present - files_desired):
            if file.endswith(".tar"):
                print("Deleting", file)
                os.unlink(file)

        return

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

    if len(bin_run_concurrently) == 0 and len(bin_run_serially) == 0:
        print("Nothing matches the given filters!")
        return

    def required(run):
        for tar in run["generates"]:
            if not run["exists"][tar]:
                return True
        return False

    bin_run_concurrently = list(filter(required, bin_run_concurrently))
    bin_run_serially = list(filter(required, bin_run_serially))
    if len(bin_run_concurrently) == 0 and len(bin_run_serially) == 0:
        print("Nothing to do!")
        return

    with multiprocessing.Pool() as pool:
        list(pool.imap_unordered(_run_task, bin_run_concurrently))

    for run in bin_run_serially:
        _run_task(run)

    pass


if __name__ == "__main__":
    main(sys.argv)

