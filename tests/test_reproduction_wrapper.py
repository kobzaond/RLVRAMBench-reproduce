import io
import tarfile
from pathlib import Path

import pytest

from reproduce import check_member, normalize_paths, sha256


@pytest.mark.parametrize("name", ["/outside", "../outside", "a/../../outside"])
def test_member_rejects_path_escape(tmp_path, name):
    with pytest.raises(ValueError):
        check_member(tarfile.TarInfo(name), tmp_path)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
@pytest.mark.parametrize("target", ["/outside", "../../outside"])
def test_member_rejects_escaping_links(tmp_path, kind, target):
    member = tarfile.TarInfo("inside/link")
    member.type = kind
    member.linkname = target
    with pytest.raises(ValueError):
        check_member(member, tmp_path)


def test_member_accepts_relative_internal_link(tmp_path):
    member = tarfile.TarInfo("inside/link")
    member.type = tarfile.SYMTYPE
    member.linkname = "../trial.json"
    check_member(member, tmp_path)


def test_member_rejects_device(tmp_path):
    member = tarfile.TarInfo("device")
    member.type = tarfile.CHRTYPE
    with pytest.raises(ValueError):
        check_member(member, tmp_path)


def test_normalization_changes_paths_not_measurements():
    assert normalize_paths(
        "/old/artifact/output/run 3980 -31 588", [Path("/old/artifact")]
    ) == "<ARTIFACT_ROOT>/output/run 3980 -31 588"


def test_digest(tmp_path):
    path = tmp_path / "small"
    path.write_bytes(b"abc")
    assert sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
