"""Test Version object and cast_version utility function."""
import pydantic
import pytest
from pydantic import ValidationError

from polus.plugins._plugins.io import Version
from polus.plugins._plugins.utils import cast_version

PYDANTIC_VERSION = pydantic.__version__.split(".", maxsplit=1)[0]

GOOD_VERSIONS = [
    "1.2.3",
    "1.4.7-rc1",
    "4.1.5",
    "12.8.3",
    "10.2.0",
    "2.2.3-dev5",
    "0.3.4",
    "0.2.34-rc23",
]
BAD_VERSIONS = ["02.2.3", "002.2.3", "1.2", "1.0", "1.03.2", "23.3.03", "d.2.4"]

PV = PYDANTIC_VERSION
print(PV)


@pytest.mark.parametrize("ver", GOOD_VERSIONS, ids=GOOD_VERSIONS)
def test_version(ver):
    """Test Version pydantic model."""
    if PV == "1":
        assert isinstance(Version(version=ver), Version)
    assert isinstance(Version(ver), Version)


@pytest.mark.skipif(int(PV) > 1, reason="requires pydantic 1")
@pytest.mark.parametrize("ver", GOOD_VERSIONS, ids=GOOD_VERSIONS)
def test_cast_version(ver):
    """Test cast_version utility function."""
    assert isinstance(cast_version(ver), Version)


@pytest.mark.parametrize("ver", BAD_VERSIONS, ids=BAD_VERSIONS)
def test_bad_version1(ver):
    """Test ValidationError is raised for invalid versions."""
    if PV == "1":
        with pytest.raises(ValidationError):
            assert isinstance(cast_version(ver), Version)
    with pytest.raises(ValidationError):
        assert isinstance(Version(ver), Version)


MAJOR_VERSION_EQUAL = ["2.4.3", "2.98.28", "2.1.2", "2.0.0", "2.4.0"]
MINOR_VERSION_EQUAL = ["1.3.3", "7.3.4", "98.3.12", "23.3.0", "1.3.5"]
PATCH_EQUAL = ["12.2.7", "2.3.7", "1.7.7", "7.7.7", "7.29.7"]


@pytest.mark.parametrize("ver", MAJOR_VERSION_EQUAL, ids=MAJOR_VERSION_EQUAL)
def test_major(ver):
    """Test major version."""
    if PV == "2":
        assert Version(ver).major == 2
    else:
        assert cast_version(ver).major == 2


@pytest.mark.parametrize("ver", MINOR_VERSION_EQUAL, ids=MINOR_VERSION_EQUAL)
def test_minor(ver):
    """Test minor version."""
    if PV == "2":
        assert Version(ver).minor == 3
    else:
        assert cast_version(ver).minor == 3


@pytest.mark.parametrize("ver", PATCH_EQUAL, ids=PATCH_EQUAL)
def test_patch(ver):
    """Test patch version."""
    if PV == "2":
        assert Version(ver).patch == 7
    else:
        assert cast_version(ver).patch == 7


def test_gt1():
    """Test greater than operator."""
    if PV == "2":
        assert Version("1.2.3") > Version("1.2.1")
    else:
        assert cast_version("1.2.3") > cast_version("1.2.1")


def test_gt2():
    """Test greater than operator."""
    if PV == "2":
        assert Version("5.7.3") > Version("5.6.3")
    else:
        assert cast_version("5.7.3") > cast_version("5.6.3")


def test_st1():
    """Test less than operator."""
    if PV == "2":
        assert Version("5.7.3") < Version("5.7.31")
    else:
        assert cast_version("5.7.3") < cast_version("5.7.31")


def test_st2():
    """Test less than operator."""
    if PV == "2":
        assert Version("1.0.2") < Version("2.0.2")
    else:
        assert cast_version("1.0.2") < cast_version("2.0.2")


def test_eq1():
    """Test equality operator."""
    if PV == "2":
        assert Version("1.3.3") == Version("1.3.3")
    else:
        assert Version(version="1.3.3") == cast_version("1.3.3")


def test_eq2():
    """Test equality operator."""
    if PV == "2":
        assert Version("5.4.3") == Version("5.4.3")
    else:
        assert Version(version="5.4.3") == cast_version("5.4.3")


def test_eq3():
    """Test equality operator."""
    if PV == "2":
        assert Version("1.3.3") != Version("1.3.8")
    else:
        assert Version(version="1.3.3") != cast_version("1.3.8")


def test_eq_str1():
    """Test equality with str."""
    if PV == "2":
        assert Version("1.3.3") == "1.3.3"
    else:
        assert Version(version="1.3.3") == "1.3.3"


def test_lt_str1():
    """Test equality with str."""
    if PV == "2":
        assert Version("1.3.3") < "1.5.3"
    else:
        assert Version(version="1.3.3") < "1.5.3"


def test_gt_str1():
    """Test equality with str."""
    if PV == "2":
        assert Version("4.5.10") > "4.5.9"
    else:
        assert Version(version="4.5.10") > "4.5.9"


def test_eq_no_str():
    """Test equality with non-string."""
    if PV == "2":
        with pytest.raises(TypeError):
            assert Version("1.3.3") == 1.3
    else:
        with pytest.raises(TypeError):
            assert Version(version="1.3.3") == 1.3
