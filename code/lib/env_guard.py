import os


class EnvGuard:
    """
    Class used to temporarily set some environment variables to a certain value.
    In particular, this is used to set the variable OMP_NUM_THREADS to "1" for
    each of the subprocesses using cvxopt.
    """
    def __init__(self, env={}):
        self.env = env
        self.old_env_stack = []

    def __enter__(self):
        # Create a backup of the environment variables and write the desired
        # value.
        old_env = {}
        for key, value in self.env.items():
            if key in os.environ:
                old_env[key] = os.environ[key]
            else:
                old_env[key] = None
            os.environ[key] = str(value)

        # Push the old_env object onto the stack of old_env objects.
        self.old_env_stack.append(old_env)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Get the environment variable backup from the stack.
        old_env = self.old_env_stack.pop()

        # Either delete environment variables that were not originally present
        # or reset them to their original value.
        for key, value in old_env.items():
            if not value is None:
                os.environ[key] = value
            else:
                del os.environ[key]

        return False


class SingleThreadEnvGuard(EnvGuard):
    def __init__(self, env={}):
        super().__init__({**{
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }}, **env)

