from itertools import repeat


def hello_world(n: int) -> str:
    """Print 'hello world' n-times.

    Parameters
    ----------
    n : int
        How many time to return hello world

    Returns
    -------
    str
        str of 'hello world' n-times
    """
    return " ".join(repeat("hello world", n))
