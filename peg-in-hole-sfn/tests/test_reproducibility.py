from pathlib import Path

from sfn.reproducibility import package_versions, sha256_file


def test_sha256_file_is_stable(tmp_path: Path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"sfn")
    first = sha256_file(path)
    assert first == sha256_file(path)
    path.write_bytes(b"sfn2")
    assert sha256_file(path) != first


def test_package_versions_reports_missing_package():
    result = package_versions(["definitely-not-an-installed-sfn-package"])
    assert result["definitely-not-an-installed-sfn-package"] is None
