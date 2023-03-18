import argparse
import logging
from configparser import ConfigParser
from shutil import rmtree
from typing import List, Optional, Tuple

import nox

nox.options.sessions = ["default"]


_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


def get_packages(group: Optional[str]) -> List[str]:
    cf = ConfigParser()
    cf.read("setup.cfg")
    return [i for i in cf["options.extras_require"].get(group, "").splitlines() if i]


def warn_package_missing(group: str, checks: List[str]) -> List[str]:
    packages = get_packages(group)
    for check in checks:
        for package in packages:
            if package.startswith(check):
                break
        else:
            _logger.warning(
                "Package %s not in setup.cfg [options.extras_require] %s.",
                check,
                group,
            )
            packages.insert(0, check)
    return packages


def install_tools(group: str, expected: Optional[List[str]] = None):
    """Install tools for package."""
    if expected is None:
        expected = []
    return warn_package_missing(group, expected)


def install_package(
    group: Optional[str] = None,
    expected: Optional[List[str]] = None,
    editable: bool = True,
) -> List[str]:
    """Install our package, optionally with tools and section."""
    if expected and not group:
        raise ValueError(
            "Not sure what to do if we want to check packages outside group"
        )

    if not group:
        if editable:
            return ["-e", "."]
        return ["."]

    if not expected:
        expected = []

    packages = warn_package_missing(group, expected)

    packages.insert(0, ".")

    if editable:
        packages.insert(0, "-e")

    return packages


def parse_option(
    session: nox.Session, argname: str, default: Optional[str] = None
) -> Tuple[str, List[str]]:
    """Parse a single option from the nox session positional arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(argname, default=default, dest="value")
    args, remaining = parser.parse_known_args(session.posargs)
    return args.value, remaining


@nox.session(python=False)
def default(session: nox.Session) -> None:
    session.notify("test")


@nox.session(python=False)
def clean(_: nox.Session) -> None:
    rmtree("build", True)
    rmtree("dist", True)


@nox.session
def test(session: nox.Session) -> None:
    session.install(
        *install_package(
            "testing",
            ["pytest", "pytest-cov", "coverage[toml]", "setuptools", "setuptools_scm"],
        )
    )
    session.run("pytest")


@nox.session
def lint(session: nox.Session) -> None:
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")


@nox.session(python=["3.7", "3.8", "3.9", "3.10"])
def test_all_python(session: nox.Session) -> None:
    session.install(*install_package("testing", ["pytest"]))
    session.run("pytest")


@nox.session(python=False)
def docs(session: nox.Session) -> None:
    session.notify("doc(build)")


@nox.session(python=False)
def doctests(session: nox.Session) -> None:
    session.notify("doc(test)")


@nox.session(python=False)
def linkcheck(session: nox.Session) -> None:
    session.notify("doc(linkcheck)")


@nox.session
@nox.parametrize(
    "command",
    [
        nox.param("html", id="build"),
        nox.param("doctest", id="test"),
        nox.param("linkcheck", id="linkcheck"),
    ],
)
def doc(session: nox.Session, command: str) -> None:
    session.install("-r", "docs/requirements.txt")
    session.run(
        "python",
        "-m",
        "sphinx.cmd.build",
        "-b",
        command,
        "-d",
        "docs/_build/doctrees",
        "docs/",
        f"docs/_build/{command}",
        env={"AUTODOCDIR": "api"},
    )


@nox.session
def build(session: nox.Session) -> None:
    session.install(
        *install_package("build", ["build", "setuptools", "setuptools_scm", "wheel"])
    )
    session.run("python", "-m", "build", "--sdist", "--wheel", ".")


@nox.session
def publish(session: nox.Session) -> None:
    repository, _ = parse_option(session, "--repository", "pypi")

    session.install("twine")
    session.run("python3", "-m", "twine", "check", "dist/*")
    session.run(
        "python3", "-m", "twine", "upload", "--repository", repository, "dist/*"
    )
