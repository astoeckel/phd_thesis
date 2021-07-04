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
"""
This script clones a revision of this repository into a temporary directory and
runs the given command inside the Docker container that collects all
dependencies.
"""

import subprocess
import sys, os
import tempfile
import hashlib
import shutil

import logging

logger = logging.getLogger(__name__)


def file_hash(filename, hasher):
    with open(filename, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            hasher.update(data)


def git_rev_parse(repo_dir, commit):
    res = subprocess.run(["git", "rev-parse", commit],
                         capture_output=True,
                         cwd=repo_dir,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Could not resolve git commit \"{commit}\" in repository \"{repo_dir}\":\n{res.stderr}"
        )
    return res.stdout.strip().lower()


def git_top_level(repo_dir):
    res = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                         capture_output=True,
                         cwd=repo_dir,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Could fetch git top-level directory")
    return res.stdout.strip()


def git_clone(repo_dir, tar_dir, commit):
    if subprocess.run([
            "git", "clone", "-q", "--progress", repo_dir, tar_dir
    ]).returncode != 0:
        raise RuntimeError("Cloning the git repository failed.")
    if subprocess.run(["git", "checkout", "-q", "--progress", commit],
                      cwd=tar_dir).returncode != 0:
        raise RuntimeError("Failed to checkout the specificed commit.")


def _parse_docker_table(output):
    lines = list(filter(bool, (s.strip() for s in output.split("\n"))))
    min_len = min(map(len, lines))
    split_point = [True] * min_len
    for i in range(len(lines)):
        for j in range(min_len - 1):
            if (lines[i][j] != ' ') or (lines[i][j + 1] == ' '):
                split_point[j] = False
    splits = [[] for _ in range(len(lines))]
    for i in range(len(lines)):
        i0 = 0
        for j in range(1, min_len):
            if not split_point[j - 1] and split_point[j]:
                i1 = j
            if split_point[j - 1] and not split_point[j]:
                splits[i].append(lines[i][i0:i1].strip())
                i0 = j
        splits[i].append(lines[i][i0:])
    res = []
    for i in range(1, len(splits)):
        entry = {}
        for j in range(len(splits[0])):
            entry[splits[0][j].replace(" ", "_").lower()] = splits[i][j]
        res.append(entry)
    return res


def _docker_image_exists(name):
    res = subprocess.run(["docker", "images", "--no-trunc", "-a", name],
                         capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError("Could not list docker images!")
    tbl = _parse_docker_table(res.stdout)
    res = None
    for line in tbl:
        if (line["repository"] == name):
            if not res is None:
                raise RuntimeError(f"Cannot unambigously resolve image {name}")
            res = line["image_id"]
    return res


def _docker_build(dockerfile, name, export_target):
    # Build the image
    res = subprocess.run([
        "docker", "build", "-f", dockerfile, "-t", name,
        os.path.dirname(dockerfile)
    ])
    if res.returncode != 0:
        raise RuntimeError("Building the Docker image failed!")

    # Export the image to a tarball
    res = subprocess.run([
        "docker",
        "save",
        "-o",
        export_target,
        name,
    ])
    if res.returncode != 0:
        raise RuntimeError("Exporting the Docker image failed!")


def _docker_import(name, import_source):
    res = subprocess.run(["docker", "load", "-i", import_source])
    if res.returncode != 0:
        raise RuntimeError("Error while importing existing docker image")


def _docker_run(image_id, repository_dir, cmd, *args):
    if os.path.isfile(os.path.join(repository_dir, cmd)):
        cmd = os.path.join("/work", cmd)

    with tempfile.TemporaryDirectory() as home_dir:
        res = subprocess.run([
            "docker", "run", "-it", "-u",
            "{:d}:{:d}".format(os.getuid(), os.getpid()), "-v",
            "{}:{}:z".format(home_dir, "/home/user"), "-v",
            "{}:{}:z".format(repository_dir, "/work"), "-e", "HOME=/home/user",
            "-e", "USER=user", "-w", "/work", image_id, cmd, *args
        ])
        if res.returncode != 0:
            logger.error("Error while executing the given command!")
            return False
        return True


def _list_all_files(path):
    res = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        res.extend((os.path.join(dirpath, file) for file in filenames))
    return res


def _parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs an experiment in a reproducible environment.')
    parser.add_argument("commit", help='Git commit of this repository to use.')
    parser.add_argument(
        "dockerfile",
        help=
        'Dockerfile to use for the experiment. This file must be located int he "docker" directory; the ".dockerfile" extension will be added automatically.'
    )
    parser.add_argument("args", nargs="*")
    parser.add_argument(
        '--hash',
        dest='hash',
        action='store_true',
        help=
        'Compute the hash that will be prefixed to the generated data files and print it to stdout'
    )
    parser.add_argument(
        '--expected_hash',
        required=False,
        default=None,
        help='Expected target hash computed by some other program. Errors out if the target hash does not match'
    )

    return parser.parse_args(argv[1:])


def main(argv):
    # Parse the arguments
    args = _parse_args(argv)

    # Fetch the root directory
    dir_root = os.path.abspath(git_top_level(os.path.dirname(__file__)))
    logger.debug(f"Root directory is {dir_root}")

    # Resolve the commit id
    commit = git_rev_parse(dir_root, args.commit)
    logger.debug(f"Commit resolves to {commit}")

    # Derive some more directory names and make sure they exist
    dir_docker = os.path.join(dir_root, "docker")
    dir_docker_images = os.path.join(dir_docker, "images")
    dir_data_tar = os.path.join(dir_root, "data")
    os.makedirs(dir_docker, exist_ok=True)
    os.makedirs(dir_docker_images, exist_ok=True)
    os.makedirs(dir_data_tar, exist_ok=True)

    # Generate the temporary directroy with the version of the source code we
    # are running
    with tempfile.TemporaryDirectory() as dir_tmp:
        # Create the target directory
        logger.debug("Cloning the git repository to {dir_tmp}")
        git_clone(dir_root, dir_tmp, commit)

        # Assemble some directories within the temporary checkout
        dir_docker_tmp = os.path.join(dir_tmp, "docker")
        dir_data_tmp = os.path.join(dir_tmp, "data")
        os.makedirs(dir_data_tmp, exist_ok=True)
        os.makedirs(dir_docker_tmp, exist_ok=True)

        # Create an inventory of the currently existing data fiels
        data_files_before = set(_list_all_files(dir_data_tmp))

        # Assemble the relevant Docker file names
        file_dockerfile = os.path.join(
            dir_docker_tmp,
            os.path.basename(args.dockerfile) + ".dockerfile")
        if not os.path.isfile(file_dockerfile):
            raise RuntimeError(
                f'Docker file "{file_dockerfile}" does not exist in revision "{commit[:8]}"'
            )

        # Compute the hash that should be used for the docker container
        docker_hash = hashlib.sha256()
        file_hash(file_dockerfile, docker_hash)
        docker_hash = docker_hash.hexdigest()[:16]
        logger.debug(f"Dockerfile hash is {docker_hash}")

        # Compute the hash that should be used for the generated output files
        target_hash = hashlib.sha256()
        target_hash.update(commit.encode('utf-8'))
        target_hash.update(" ".join(args.args).encode("utf-8"))
        file_hash(file_dockerfile, target_hash)
        target_hash = target_hash.hexdigest()[:16]
        logger.debug(f"Build artefact prefix is {target_hash}")

        if args.expected_hash and target_hash.lower() != args.expected_hash.lower():
            raise RuntimeError(f"Expected hash {args.target_hash}, but got {target_hash}!")

        # Just return if "--hash" was set
        if args.hash:
            print(target_hash)
            return

        # Build the Docker container
        image_name = os.path.basename(args.dockerfile) + "_" + docker_hash[:16]
        file_docker_image = os.path.join(dir_docker, 'images',
                                         image_name + ".tar")

        docker_image_id = _docker_image_exists(image_name)
        if docker_image_id is None:
            if os.path.isfile(file_docker_image):
                logger.info(f"Reading Docker image from tarball")
                _docker_import(image_name, file_docker_image)
            else:
                logger.info(f"Building Docker image")
                _docker_build(file_dockerfile, image_name, file_docker_image)

            docker_image_id = _docker_image_exists(image_name)
            if docker_image_id is None:
                raise RuntimeError(
                    "Just built a docker image, but can't find its ID?")

        logger.info(f"Using docker image {image_name} --> {docker_image_id}")

        logger.info("Entering the Docker environment")
        if len(args.args) == 0:
            raise RuntimeError("Must provide at least one command!")
        cmd = args.args[0]
        _docker_run(docker_image_id, dir_tmp, cmd, *args.args[1:])

        # List all files that were placed in the data directory
        data_files_after = set(_list_all_files(dir_data_tmp))

        # Copy the files to the "data" directory
        for file_src in data_files_after - data_files_before:
            # Determine the file path relative to the data directory
            file_rel = os.path.relpath(file_src, start=dir_data_tmp)
            file_rel_path, file_rel_name = os.path.split(file_rel)

            # If the command was a file in the repository, use that to assemble
            # the target directory
            if cmd.startswith("code/") and os.path.isfile(
                    os.path.join(dir_tmp, cmd)):
                tar_dir = os.path.join(
                    "generated", os.path.relpath(os.path.dirname(cmd), "code"))
            else:
                tar_dir = "generated"

            file_tar = os.path.join(dir_data_tar, file_rel_path, tar_dir,
                                    target_hash + "_" + file_rel_name)

            os.makedirs(os.path.dirname(file_tar), exist_ok=True)
            logger.info(f"Copying {file_src} --> {file_tar}")

            shutil.copy(file_src, file_tar)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv)

