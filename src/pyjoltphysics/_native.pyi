"""The compiled pyjoltphysics module over Jolt Physics."""



def jolt_version() -> str:
    """
    Return the version of the Jolt Physics compiled into the module, such as '5.6.0'.
    """

def build_options() -> dict[str, bool | str]:
    """Return the options the module and its Jolt were built with, by name."""

def selftest() -> bool:
    """
    Return True after stepping a one-body Jolt world once on the calling thread.
    """
