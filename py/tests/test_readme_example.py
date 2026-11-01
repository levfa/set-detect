"""Checks that the CLI commands in the README example parse."""

import argparse
import importlib
import pathlib as pl
import re
import shlex
import tomllib

import pytest

PY_ROOT = pl.Path(__file__).resolve().parents[1]
README = PY_ROOT.parent / "README.md"


def _script_modules() -> dict[str, str]:
    scripts = tomllib.loads((PY_ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    return {name: target.split(":")[0] for name, target in scripts.items()}


def _readme_commands() -> list[list[str]]:
    blocks = re.findall(r"```[^\n]*\n(.*?)```", README.read_text(), flags=re.DOTALL)
    commands = []
    for block in blocks:
        argv = shlex.split(block.replace("\\\n", " "))
        if argv and argv[0].startswith("set-"):
            commands.append(argv)
    return commands


COMMANDS = _readme_commands()


class _Parsed(Exception):
    def __init__(self, namespace: argparse.Namespace):
        self.namespace = namespace


def test_readme_has_example_commands():
    assert len(COMMANDS) >= 10


@pytest.mark.parametrize("argv", COMMANDS, ids=[" ".join(a[:2]) for a in COMMANDS])
def test_readme_command_parses(argv, monkeypatch):
    modules = _script_modules()
    assert argv[0] in modules, f"unknown script {argv[0]!r}"
    mod = importlib.import_module(modules[argv[0]])

    original = argparse.ArgumentParser.parse_args

    def parse_and_stop(self, args=None, namespace=None):
        raise _Parsed(original(self, args, namespace))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_and_stop)
    with pytest.raises(_Parsed) as exc_info:
        mod.main(argv[1:])
    assert exc_info.value.namespace.command == argv[1]
