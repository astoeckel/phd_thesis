#!/usr/bin/env python3

import sys, os
import toml
import run_in_container
import tempfile
import multiprocessing
import hashlib

def _parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs one or all of the experiments listed in the manifest'
    )
    parser.add_argument('--manifest',
                        default="~/code/Manifest.toml",
                        help="Manifest file describing all experiments. ~/ is extended to the Git root.")
    parser.add_argument('--list',
                        action='store_true',
                        help="Checks whether the output files exist")
    parser.add_argument('files', nargs='*', help="Files to execute (relative to the manifest file directory)")

    return parser.parse_args(argv[1:])


def _load_manifest(manifest_file, root):
    # Load the TOML file
    manifest = toml.load(manifest_file)

    # Elaborate all "generates" sections into "runs"
    for file, obj in manifest.items():
        if ("generates" in obj) == ("runs" in obj):
            raise RuntimeError(f"Either \"generates\" or \"runs\" must be set in entry \"{file}\"")

        if "generates" in obj:
            obj["runs"] = [{"args": [], "generates": obj["generates"]}]

    # Elaborate all "args" list by prepending adding the complete command
    manifest_path = os.path.abspath(os.path.dirname(manifest_file))
    manifest_path_rel = os.path.relpath(manifest_path, root)
    for file, obj in manifest.items():
        for run in obj["runs"]:
            run["args"] = [os.path.join(manifest_path_rel, file), *run["args"]]

    # Elaborate the commit IDs
    for file, obj in manifest.items():
        if not "commit" in obj:
            raise RuntimeError(f"Missing \"commit\" in entry \"{file}\"")
        obj["commit"] = run_in_container.git_rev_parse(root, obj["commit"])

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
    tasks = [(root, commit, files) for commit, files in manifest_by_commits.items()]
    with multiprocessing.Pool() as pool:
        for commit, contents in pool.imap_unordered(_load_dockerfiles, tasks):
            docker_file_contents[commit] = contents

    # Compute the target hashes
    for file, obj in manifest.items():
        for run in obj["runs"]:
            target_hash = hashlib.sha256()
            target_hash.update(obj["commit"].encode('utf-8'))
            target_hash.update(" ".join(run["args"]).encode("utf-8"))
            target_hash.update(docker_file_contents[obj["commit"]][obj["dockerfile"]])
            target_hash = target_hash.hexdigest()[:16]
            run["target_hash"] = target_hash


def _compute_target_files(manifest, root):
    for file, obj in manifest.items():
        for run in obj["runs"]:
            for i, target in enumerate(run["generates"]):
                run["generates"][i] = os.path.join("data", os.path.dirname(file), run["target_hash"] + "_" + target)

def main(argv):
    args = _parse_args(argv)

    # Fetch the manifest file
    manifest_file = args.manifest
    if manifest_file.startswith("~/"):
        manifest_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', manifest_file[2:]))

    # Fetch the git top-level directory
    root = os.path.abspath(run_in_container.git_top_level(os.path.dirname(manifest_file)))

    # Load and elaborate the manifest
    manifest = _load_manifest(manifest_file, root=root)
    _compute_target_hashes(manifest, root=root)
    _compute_target_files(manifest, root=root)

    if args.list:
        from colored import fg, bg, attr
        for file, obj in manifest.items():
            sys.stdout.write("{}{}{}\n".format(attr("bold"), file, attr("reset")))
            for run in obj["runs"]:
                sys.stdout.write("{} {} {}\n".format(obj["commit"][:8], obj["dockerfile"], " ".join(run["args"])))
                for tar in run["generates"]:
                    if os.path.isfile(tar):
                        sys.stdout.write("\t{}{}✔{}".format(fg("green"), attr("bold"), attr("reset")))
                    else:
                        sys.stdout.write("\t\t{}{}✘{}".format(fg("red"), attr("bold"), attr("reset")))
                    sys.stdout.write("\t{}\n\n".format(tar))
            sys.stdout.write("\n")
        return


    pass


if __name__ == "__main__":
    main(sys.argv)

