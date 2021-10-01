#!/usr/bin/env python3

# Implementation of a multiprocessing pool that (in contrast to Pythons's)
# actually works. Uses GNU Make in the background.

import hashlib
import tempfile
import os
import subprocess
import pickle
import base64
import shlex
import env_guard


def run(argv, init_cback, runner_cback, serialise_cback, n_threads=None):
    if (n_threads is None) or (n_threads <= 0):
        n_threads = os.cpu_count()

    if len(argv) == 4:
        _, program, fn, data = argv
        if program == "child":
            res = runner_cback(pickle.loads(base64.b64decode(data)))
            with open(fn, 'wb') as f:
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run the initialisation routine
            params, data = init_cback()

            # Pickle the given parameters and compute a hash
            params_pickled_b64 = []
            params_hashes = []
            params_fns = []
            for param in params:
                # Dump the parameters to a base64-encoded string
                param_pickled_b64 = str(
                    base64.b64encode(
                        pickle.dumps(param, pickle.HIGHEST_PROTOCOL)), 'ascii')
                params_pickled_b64.append(param_pickled_b64)

                # Compute a hash identifying this parameter set
                m = hashlib.sha256()
                for arg in argv:
                    m.update(arg.encode('utf-8'))
                m.update(param_pickled_b64.encode('utf-8'))
                hash_ = m.hexdigest()[:10]
                params_hashes.append(hash_)

                # Compute the filename to store the results of this computation
                params_fns.append(os.path.join(tmpdir, hash_ + ".pkl"))

            # Create a temporary directory. We will store a makefile and all the
            # intermediate results there
            executable = os.path.realpath(argv[0])
            makefile_fn = os.path.join(tmpdir, 'Makefile')
            with open(makefile_fn, 'w') as f:
                # Fancy progress bar, see https://stackoverflow.com/a/35320895
                f.write("""ifneq ($(words $(MAKECMDGOALS)),1) # if no argument was given to make...
.DEFAULT_GOAL = all # set the default goal to all
%:                   # define a last resort default rule
	@$(MAKE) $@ --no-print-directory -rRf $(firstword $(MAKEFILE_LIST)) # recursive make call, 
else
ifndef ECHO
T := $(shell $(MAKE) $(MAKECMDGOALS) --no-print-directory \
	-nrRf $(firstword $(MAKEFILE_LIST)) \
	ECHO="COUNTTHIS" | grep -c "COUNTTHIS")
N := x
C = $(words $N)$(eval N := x $N)
ECHO = echo -ne "\r [`expr $C '*' 100 / $T`%]"
endif

""")
                f.write("all: {}\n\n".format(" ".join(params_fns)))
                for i in range(len(params)):
                    f.write(params_fns[i] + ":\n")
                    f.write(
                        f"\t@$(ECHO)\n\t{executable} child {params_fns[i]} {shlex.quote(params_pickled_b64[i])}\n\n"
                    )

                f.write("endif\n")

            # Execute the makefile
            with env_guard.SingleThreadEnvGuard():
                res = subprocess.run(
                    ["make", "-j", str(n_threads)], cwd=tmpdir)
                if res.returncode != 0:
                    raise RuntimeError(
                        "Error while working on the multiprocessing pool")

            # Call the serialisation routine for each parameter
            for i, param in enumerate(params):
                with open(params_fns[i], 'rb') as f:
                    res = pickle.load(f)
                    serialise_cback(res, data)
            return data

if __name__ == "__main__":
    import sys

    def test_init():
        return [1, 2, 3], "Data"

    def test_runner(arg):
        return [arg, arg * 2]

    def test_serialise(res, data):
        print(res, data)

    run(sys.argv, test_init, test_runner, test_serialise)

