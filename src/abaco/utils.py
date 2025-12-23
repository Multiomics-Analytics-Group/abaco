# these functions are pretty general (file that can be reused across projects)
import argparse
import logging
import os
import sys
from datetime import datetime
from urllib.parse import urlparse, urlunsplit
import pandas as pd

# import yaml


## CHECKS
def assert_path(filepath: str):
    """
    Check that the given filepath is a string and that it exists.

    Parameters
    ----------
    filepath : str
        The filepath or folder path to check.

    Raises
    ------
    TypeError
        If the filepath is not a string.
    FileNotFoundError
        If the filepath does not exist.

    Example
    -------
    >>> assert_path("..")
    >>> assert_path("./tests")
    """
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string: {type(filepath)}")
    if not os.path.exists(os.path.abspath(filepath)):
        raise FileNotFoundError(f"The specified path does not exist: {filepath}")


def create_folder(directory_path: str, is_nested: bool = False) -> bool:
    """
    Create a folder if it doesn't exist.

    Parameters
    ----------
    directory_path : str
        The path of the directory to create.
    is_nested : bool, optional
        Whether to create nested directories (True uses os.makedirs, False uses os.mkdir), by default False.

    Returns
    -------
    bool
        True if the folder was created, False if it already existed.

    Raises
    ------
    TypeError
        If directory_path is not a string.
    ValueError
        If directory_path is an existing file.
    OSError
        If there is an error creating the directory.
    """
    # PRECONDITION CHECK
    if not isinstance(directory_path, str):
        raise TypeError(f"filepath must be a string: {type(directory_path)}")
    abs_path = os.path.abspath(directory_path)

    # make sure it is a folder not a file
    if os.path.isfile(abs_path):
        raise ValueError(
            f"directory_path is an existing file when it should be a folder/foldername: {abs_path}"
        )
    # if folder already exists
    elif os.path.isdir(abs_path):
        return False
    # create the folder(s)
    else:
        try:
            if is_nested:
                # Create the directory and any necessary parent directories
                os.makedirs(directory_path, exist_ok=True)
                return True
            else:
                # Create only the final directory (not nested)
                os.mkdir(directory_path)
                return True
        except OSError as e:
            raise OSError(f"Error creating directory '{directory_path}': {e}") from e


def assert_nonempty_keys(dictionary: dict):
    """
    Check that the keys in a dictionary are not empty strings.

    Parameters
    ----------
    dictionary : dict
        A dictionary (e.g., config file).

    Raises
    ------
    AssertionError
        If dictionary is not a dict or if any key is empty or blank.
    """
    # PRECONDITIONS
    if not isinstance(dictionary, dict):
        raise TypeError(f"dictionary must be a dict, got {type(dictionary)}")

    # MAIN FUNCTION
    for key in dictionary:
        if type(key) is str:
            assert key, f'There is an empty key (e.g., ""): {key, dictionary.keys()}'
            assert (
                key.strip()
            ), f'There is a blank key (e.g., space, " "): {key, dictionary.keys()}'


def assert_nonempty_vals(dictionary: dict):
    """
    Check that the values in a dictionary are not empty strings.

    Parameters
    ----------
    dictionary : dict
        A dictionary (e.g., config file).

    Raises
    ------
    AssertionError
        If dictionary is not a dict or if any value is empty or blank.
    """
    # PRECONDITIONS
    if not isinstance(dictionary, dict):
        raise TypeError(f"dictionary must be a dict, got {type(dictionary)}")

    # MAIN FUNCTION
    for v in dictionary.items():
        if type(v) is str:
            assert v, f'There is an empty key (e.g., ""): {v, dictionary.items()}'
            assert (
                v.strip()
            ), f'There is a blank key (e.g., space, " "): {v, dictionary.items()}'


def normalize_url(host: str, port: int, scheme: str = "http") -> str:
    """
    Normalize the given URL, ensuring it starts with the specified scheme.

    Parameters
    ----------
    host : str
        The host to be normalized.
    port : int
        The port number.
    scheme : str, optional
        The URL scheme (default is "http").

    Returns
    -------
    str
        The normalized URL.

    Raises
    ------
    TypeError
        If host, port, or scheme are not of the correct type, or if URL cannot be normalized.

    Examples
    --------
    >>> normalize_url("localhost", 7474)
    'http://localhost:7474'
    >>> normalize_url("example.com", 80, "bolt")
    'bolt://example.com:80'
    """
    ## PRECONDITIONS
    if not isinstance(host, str):
        raise TypeError(f"host should be a str e.g., 'localhost': {type(host)}")
    if not isinstance(port, int):
        raise TypeError(f"port must be int e.g., '7474': {type(port)}")
    if not isinstance(scheme, str):
        raise TypeError(f"scheme must be str: {type(scheme)}")

    ## MAIN FUNCTION
    if not urlparse(host).netloc:
        host = urlunsplit([scheme, host, "", "", ""])

    # Remove any trailing slashes
    url = host.rstrip("/")

    # Add the port
    url = f"{url}:{str(port)}"

    ## POSTCOND CHECKS
    if not urlparse(url).netloc:
        raise TypeError(f"Unable to normalize url: {url}")

    return url


def get_args(prog_name: str, others: dict = None):
    """
    Initiate argparse.ArgumentParser() and add common arguments.

    Parameters
    ----------
    prog_name : str
        The name of the program.
    others : dict, optional
        Additional keyword arguments for ArgumentParser, by default {}.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.

    Raises
    ------
    TypeError
        If prog_name is not a string or others is not a dict.
    """
    ### PRECONDITIONS
    if not isinstance(prog_name, str):
        raise TypeError(f"prog_name should be a string: {type(prog_name)}")
    if others is None:
        others = {}
    elif not isinstance(others, dict):
        raise TypeError(f"other kwargs must be a dict: {type(others)}")

    ## MAIN FUNCTION
    # init
    parser = argparse.ArgumentParser(prog=prog_name, **others)
    # config file path
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        default="demo/config.yaml",
        help="provide path to config yaml file",
    )
    args = parser.parse_args()
    return args


def get_basename(fname: None | str = None) -> str:
    """
    Get the basename of a given filename, without file extension.

    If no filename is given, returns the basename of the current script.

    Parameters
    ----------
    fname : str or None, optional
        The filename to get basename of, or None (default is None).

    Returns
    -------
    str
        Basename of the given filepath or the current file the function is executed in.
    """
    if fname is not None:
        # PRECONDITION
        assert_path(fname)
        # MAIN FUNCTIONS
        return os.path.splitext(os.path.basename(fname))[0]
    else:
        return os.path.splitext(os.path.basename(sys.argv[0]))[0]


def get_time(incl_time: bool = True, incl_timezone: bool = True) -> str:
    """
    Get current date, time (optional), and timezone (optional) for file naming.

    Parameters
    ----------
    incl_time : bool, optional
        Whether to include timestamp in the string (default is True).
    incl_timezone : bool, optional
        Whether to include the timezone in the string (default is True).

    Returns
    -------
    str
        String including date, timestamp and/or timezone, e.g. 'yyyyMMdd_hhmm_timezone'.

    Raises
    ------
    TypeError
        If incl_time or incl_timezone are not bool.
    AssertionError
        If the output format is not as expected.
    """
    # PRECONDITIONALS
    if not isinstance(incl_time, bool):
        raise TypeError("incl_time must be True or False")
    if not isinstance(incl_timezone, bool):
        raise TypeError("incl_timezone must be True or False")

    # MAIN FUNCTION
    # getting current time and timezone
    the_time = datetime.now()
    timezone = datetime.now().astimezone().tzname()
    # convert date parts to string
    y = str(the_time.year)
    M = str(the_time.month)
    d = str(the_time.day)
    h = str(the_time.hour)
    m = str(the_time.minute)
    s = str(the_time.second)
    # putting date parts into one string
    if incl_time and incl_timezone:
        fname = "_".join([y + M + d, h + m + s, timezone])
    elif incl_time:
        fname = "_".join([y + M + d, h + m + s])
    elif incl_timezone:
        fname = "_".join([y + M + d, timezone])
    else:
        fname = y + M + d

    # POSTCONDITIONALS
    parts = fname.split("_")
    if incl_time and incl_timezone:
        assert len(parts) == 3, f"time and/or timezone inclusion issue: {fname}"
    elif incl_time or incl_timezone:
        assert len(parts) == 2, f"time/timezone inclusion issue: {fname}"
    else:
        assert len(parts) == 1, f"time/timezone inclusion issue: {fname}"

    return fname


def generate_log_filename(folder: str = "logs", suffix: str = "") -> str:
    """
    Create a log file name and path.

    Parameters
    ----------
    folder : str, optional
        Name of the folder to put the log file in (default is "logs").
    suffix : str, optional
        Additional string to add to the log file name (default is "").

    Returns
    -------
    str
        The file path to the log file.
    """
    # PRECONDITIONS
    create_folder(folder)

    # MAIN FUNCTION
    log_filename = get_time(incl_timezone=False) + "_" + suffix + ".log"
    log_filepath = os.path.join(folder, log_filename)

    return log_filepath


def init_log(filename: str, display: bool = False, logger_id: str | None = None):
    """
    Configure a custom Python logger with file and optional stdout handlers.

    Parameters
    ----------
    filename : str
        Filepath to log record file.
    display : bool, optional
        Whether to print the logs to standard output (default is False).
    logger_id : str or None, optional
        An optional identifier for the logger. If None, defaults to 'root'.

    Returns
    -------
    logging.Logger
        Configured logger object.

    Raises
    ------
    TypeError
        If filename is not a string or logger_id is not a string or None.
    """
    # PRECONDITIONS
    if not isinstance(filename, str):
        raise TypeError(f"filename must be a string: {filename}")
    if not (isinstance(logger_id, str) or logger_id is None):
        raise TypeError("logger_id must be a string or None")

    # MAIN FUNCTION
    # init handlers
    file_handler = logging.FileHandler(filename=filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    if display:
        handlers = [file_handler, stdout_handler]
    else:
        handlers = [file_handler]

    # logger configuration
    logging.basicConfig(
        # level=logging.DEBUG,
        format="[%(asctime)s] %(name)s: %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("matplotlib.font_manager").disabled = True

    # instantiate the logger
    logger = logging.getLogger(logger_id)
    logger.setLevel(logging.DEBUG)

    return logger


def get_logger():
    """
    Initialize and return a logger with a log file named after the current script.

    Returns
    -------
    logging.Logger
        Configured logger object.
    """
    # get log suffix, which will be the current script's base file name
    log_suffix = get_basename()
    # generate log file name
    log_file = generate_log_filename(suffix=log_suffix)
    # init logger
    logger = init_log(log_file, display=True)
    # log it
    logger.info(f"Path to log file: {log_file}")

    return logger


# FUNCTIONS FOR CONFIG
# def config_loader(filepath: str) -> dict:
#     """
#     Load a YAML config file as a dictionary.

#     Parameters
#     ----------
#     filepath : str
#         Path to the config file.

#     Returns
#     -------
#     dict
#         Configuration parameters as a dictionary.
#     """
#     # PRECONDITIONS
#     assert_path(filepath)

#     # MAIN FUNCTION
#     with open(filepath, "r") as f:
#         contents = yaml.safe_load(f)

#     # POSTCONDITIONS
#     assert isinstance(contents, dict), "content not returned as a dict"

#     return contents


def df_joiner(
    df_dict: dict[pd.DataFrame],
    on: str,
    how: str = "outer",
) -> pd.DataFrame:
    """
    Join multiple dataframes on a common column.

    Parameters
    ----------
    df_dict : dict of pandas.DataFrame
        Dictionary of dataframes to join.
    on : str, optional
        Column to join on. Defaults to "taxa".
    how : str, optional
        Type of join. Defaults to "outer".

    Returns
    -------
    pandas.DataFrame
        Joined dataframe.
    """

    ## PRECONDITION CHECKS
    if not isinstance(df_dict, dict):
        raise TypeError(f"df_dict must be a dict: {type(df_dict)}")
    if not isinstance(on, str):
        raise TypeError(f"on must be a str: {type(on)}")
    for key, df in df_dict.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df_dict values must be pd.DataFrame: {type(df)}")
        if (on not in df.columns) and (on not in df.index.names):
            raise ValueError(f"Column '{on}' not found in dataframe with key '{key}'")
    if how not in ["left", "right", "outer", "inner"]:
        raise ValueError(f"how must be one of 'left', 'right', 'outer', 'inner': {how}")

    ## MAIN FUNCTION
    # dfs into a list
    df_list = list(df_dict.values())
    # init the merged df with the first one
    df_merged = df_list[0]
    # for all others, merge iteratively
    for df in df_list[1:]:
        df_merged = pd.merge(df_merged, df, on=on, how=how)
    return df_merged
