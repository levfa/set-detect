import datetime as dt
import json

import pytest

from setdetect.model import versioning


def _today() -> str:
    return dt.date.today().strftime("%Y%m%d")


def test_next_version_name_empty_dir(tmp_path):
    assert versioning.next_version_name(tmp_path) == f"v01-{_today()}"


def test_next_version_name_increments_past_existing_max(tmp_path):
    (tmp_path / "v01-20240101").mkdir()
    (tmp_path / "v03-20240102").mkdir()
    assert versioning.next_version_name(tmp_path) == f"v04-{_today()}"


def test_list_versions_filters_and_sorts(tmp_path):
    (tmp_path / "v02-20240102").mkdir()
    (tmp_path / "v01-20240101").mkdir()
    (tmp_path / "not-a-version").mkdir()
    (tmp_path / "v05-20240105").write_text("not a directory")
    assert versioning.list_versions(tmp_path) == ["v01-20240101", "v02-20240102"]


def test_list_versions_missing_dir(tmp_path):
    assert versioning.list_versions(tmp_path / "missing") == []


def test_current_version_none_without_symlink(tmp_path):
    assert versioning.current_version(tmp_path) is None


def test_resolve_current_raises_when_missing(tmp_path):
    with pytest.raises(SystemExit):
        versioning.resolve_current(tmp_path)


def test_promote_raises_for_nonexistent_version(tmp_path):
    with pytest.raises(SystemExit):
        versioning.promote(tmp_path, "v01-20240101")


def test_promote_happy_path_and_repoint(tmp_path):
    (tmp_path / "v01-20240101").mkdir()
    (tmp_path / "v02-20240102").mkdir()

    versioning.promote(tmp_path, "v01-20240101")
    assert versioning.current_version(tmp_path) == "v01-20240101"
    assert versioning.resolve_current(tmp_path).resolve() == (tmp_path / "v01-20240101").resolve()

    versioning.promote(tmp_path, "v02-20240102")
    assert versioning.current_version(tmp_path) == "v02-20240102"
    assert not (tmp_path / ".current.tmp").exists()


def test_write_run_metadata_writes_expected_fields(tmp_path):
    versioning.write_run_metadata(tmp_path, version="v01-20240101", family="test-family")
    data = json.loads((tmp_path / "run.json").read_text())
    assert "created_at" in data
    assert "git_sha" in data
    assert "git_dirty" in data
    assert data["version"] == "v01-20240101"
    assert data["family"] == "test-family"
