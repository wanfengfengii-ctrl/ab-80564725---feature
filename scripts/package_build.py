#!/usr/bin/env python3
"""Build publishable artifacts (wheel + sdist) for the project.

Prefers the standard ``python -m build`` toolchain. When it is not
installed (for example inside an offline container image), falls back
to a self-contained standard-library builder that emits a valid
PEP 427 wheel and a PEP 625 sdist from pyproject.toml metadata.

Usage: python scripts/package_build.py [--outdir dist]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = "app"
EXTRA_SDIST_PATHS = ("pyproject.toml", "README.md", "tests", "scripts", "verify.py")


def load_project_metadata() -> dict:
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)["project"]


def normalize(name: str) -> str:
    """PEP 503/625 distribution name normalization."""
    return re.sub(r"[-_.]+", "_", name).lower()


def standard_toolchain_available() -> bool:
    import importlib.util

    return (
        importlib.util.find_spec("build") is not None
        and importlib.util.find_spec("setuptools") is not None
    )


def run_standard_build(outdir: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", outdir],
        cwd=ROOT,
    )
    return proc.returncode == 0


def _pkg_info(meta: dict) -> bytes:
    lines = [
        "Metadata-Version: 2.1",
        "Name: %s" % meta["name"],
        "Version: %s" % meta["version"],
    ]
    if meta.get("description"):
        lines.append("Summary: %s" % meta["description"])
    if meta.get("requires-python"):
        lines.append("Requires-Python: %s" % meta["requires-python"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_wheel(outdir: str, meta: dict) -> str:
    dist = normalize(meta["name"])
    version = meta["version"]
    dist_info = "%s-%s.dist-info" % (dist, version)
    wheel_name = "%s-%s-py3-none-any.whl" % (dist, version)
    wheel_path = os.path.join(outdir, wheel_name)

    records: list[tuple[str, str, str]] = []

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:

        def write(arcname: str, data: bytes) -> None:
            zf.writestr(arcname, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            records.append(
                (arcname, "sha256=%s" % digest.rstrip(b"=").decode("ascii"), str(len(data)))
            )

        package_root = os.path.join(ROOT, PACKAGE_DIR)
        for dirpath, _dirnames, filenames in os.walk(package_root):
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                full = os.path.join(dirpath, filename)
                with open(full, "rb") as fh:
                    write(os.path.relpath(full, ROOT), fh.read())

        write(os.path.join(dist_info, "METADATA"), _pkg_info(meta))
        write(
            os.path.join(dist_info, "WHEEL"),
            b"Wheel-Version: 1.0\n"
            b"Generator: package_build.py (stdlib fallback)\n"
            b"Root-Is-Purelib: true\n"
            b"Tag: py3-none-any\n",
        )
        records.append((os.path.join(dist_info, "RECORD"), "", ""))
        record_text = "".join(",".join(row) + "\n" for row in records)
        zf.writestr(os.path.join(dist_info, "RECORD"), record_text)

    return wheel_path


def build_sdist(outdir: str, meta: dict) -> str:
    base = "%s-%s" % (normalize(meta["name"]), meta["version"])
    sdist_path = os.path.join(outdir, base + ".tar.gz")

    with tarfile.open(sdist_path, "w:gz") as tar:
        pkg_info = _pkg_info(meta)
        info = tarfile.TarInfo(base + "/PKG-INFO")
        info.size = len(pkg_info)
        tar.addfile(info, io.BytesIO(pkg_info))

        def add_tree(rel_dir: str) -> None:
            for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, rel_dir)):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for filename in sorted(filenames):
                    if filename.endswith((".pyc", ".pyo")):
                        continue
                    full = os.path.join(dirpath, filename)
                    arcname = base + "/" + os.path.relpath(full, ROOT)
                    tar.add(full, arcname=arcname)

        for rel in EXTRA_SDIST_PATHS:
            full = os.path.join(ROOT, rel)
            if os.path.isdir(full):
                add_tree(rel)
            elif os.path.isfile(full):
                tar.add(full, arcname=base + "/" + rel)

    return sdist_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=os.path.join(ROOT, "dist"))
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    # Start from a clean outdir so stale artifacts cannot mask a bad build.
    for entry in os.listdir(args.outdir):
        if entry.endswith((".whl", ".tar.gz")):
            os.remove(os.path.join(args.outdir, entry))

    if standard_toolchain_available():
        print("[package_build] using standard 'python -m build' toolchain")
        if not run_standard_build(args.outdir):
            print("[package_build] ERROR: standard build failed", file=sys.stderr)
            return 1
    else:
        print("[package_build] 'build'/'setuptools' unavailable; using stdlib fallback")
        meta = load_project_metadata()
        wheel = build_wheel(args.outdir, meta)
        sdist = build_sdist(args.outdir, meta)
        print("[package_build] wrote %s" % wheel)
        print("[package_build] wrote %s" % sdist)

    artifacts = os.listdir(args.outdir)
    wheels = [a for a in artifacts if a.endswith(".whl")]
    sdists = [a for a in artifacts if a.endswith(".tar.gz")]
    if not wheels or not sdists:
        print(
            "[package_build] ERROR: expected a wheel and an sdist, found: %s" % artifacts,
            file=sys.stderr,
        )
        return 1
    print("[package_build] artifacts: %s" % ", ".join(sorted(wheels + sdists)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
