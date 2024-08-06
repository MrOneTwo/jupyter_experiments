import pytest

@pytest.fixture(scope="function", autouse=True)
def dummy():
    print("...running")
