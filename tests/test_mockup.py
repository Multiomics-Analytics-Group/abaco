from python_package import hello_world


def test_hello_world_3times():
    expected = "hello world hello world hello world"
    result = hello_world(3)
    assert result == expected
