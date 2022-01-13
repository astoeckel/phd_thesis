#!/usr/bin/bash

# chdir into the directory this file is located in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR"

# Copy the bibliography file into this directory
cp ../bibliography/bibliography.bib .

# Create the "build directories" if they do not already exist
mkdir -p build/generated
mkdir -p build/work

# Create temporary a overlay filesystem
TMP="$( mktemp -d )"
fuse-overlayfs -o lowerdir=.,upperdir=build/generated,workdir=build/work "$TMP"
trap "cd ~ && fusermount -u ""$TMP"" && rmdir ""$TMP" EXIT

# Use latexmk to build the document
cd "$TMP"
rm -f .datafile_cache.json
latexmk
rm -f .datafile_cache.json

# Edit the syntex file to point at the right directory
for STEX in *.synctex.gz; do
	[ -f "$STEX" ] || continue
	zcat "$STEX" | sed "s|/tmp/tmp[^/]*/\.|$DIR|g" | gzip > "$STEX.tmp"
	mv "$STEX.tmp" "$STEX"
done

# Copy the output file back
cp -f astoeckel_phd_thesis_2021.pdf "$DIR"
cp -f astoeckel_phd_thesis_2021.log "$DIR"
cp -f astoeckel_phd_thesis_2021.synctex.gz "$DIR"

