import datetime as dt
import json
import pathlib as pl
import re
import subprocess
import typing as tp

_VERSION_RE = re.compile(r"^v(\d+)-\d{8}$")


def next_version_name(runs_root: pl.Path) -> str:
    """Return the next 'vNN-YYYYMMDD' version name for a family's runs directory."""
    max_n = 0
    if runs_root.is_dir():
        for entry in runs_root.iterdir():
            m = _VERSION_RE.match(entry.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    today = dt.date.today().strftime("%Y%m%d")
    return f"v{max_n + 1:02d}-{today}"


def list_versions(runs_root: pl.Path) -> list[str]:
    """List existing version directories under a family's runs directory, sorted."""
    if not runs_root.is_dir():
        return []
    return sorted(entry.name for entry in runs_root.iterdir() if entry.is_dir() and _VERSION_RE.match(entry.name))


def current_version(runs_root: pl.Path) -> str | None:
    """Name of the version currently promoted via the 'current' symlink, or None."""
    current = runs_root / "current"
    if not current.is_symlink():
        return None
    return current.readlink().name


def resolve_current(runs_root: pl.Path) -> pl.Path:
    """Resolve the 'current' promotion symlink; raise a clear SystemExit if missing."""
    current = runs_root / "current"
    if not current.exists():
        raise SystemExit(
            f"no promoted version under {runs_root} (missing '{current}'); "
            "train a run and then promote it with the 'promote <version>' subcommand"
        )
    return current


def promote(runs_root: pl.Path, version: str) -> None:
    """Atomically repoint runs_root/'current' to `version` (a relative symlink)."""
    if not (runs_root / version).is_dir():
        raise SystemExit(f"no such version {version!r} under {runs_root}")
    current = runs_root / "current"
    tmp = runs_root / ".current.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(version, target_is_directory=True)
    tmp.replace(current)


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def write_run_metadata(run_dir: pl.Path, **fields: tp.Any) -> None:
    """Write run.json into run_dir with git provenance, a timestamp, and caller-supplied fields."""
    git_sha = _git("rev-parse", "--short", "HEAD")
    git_status = _git("status", "--porcelain")
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "git_dirty": None if git_status is None else bool(git_status),
        **fields,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
