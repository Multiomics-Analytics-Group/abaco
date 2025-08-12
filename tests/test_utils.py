import pytest
from abaco import utils


def test_get_basename():
    assert utils.get_basename("./tests/test_utils.py") == "test_utils"


@pytest.mark.xfail(raises=FileNotFoundError)
def test_assert_path_nonexistent():
    utils.assert_path("this-is-obvi-a-nonexistent/path/blahdeblah")


@pytest.mark.xfail(raises=TypeError)
def test_assert_path_non_string():
    utils.assert_path(123)


@pytest.mark.xfail(reason="the output of get_time() is dynamic")
def test_get_time():
    assert utils.get_time() == "20231019_101758_CEST"
