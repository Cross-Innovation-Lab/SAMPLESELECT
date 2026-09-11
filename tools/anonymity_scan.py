"""Fail when identity, credential, or private-infrastructure traces are found."""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path


TEXT_SUFFIXES = {
    ".py", ".toml", ".json", ".yaml", ".yml", ".md", ".txt", ".cfg", ".ini"
}
PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "windows_absolute_path": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    "windows_unc_path": re.compile(r"\\\\[A-Za-z0-9._-]+[\\/]"),
    "private_unix_path": re.compile(
        r"/(?:home|Users|data\d*|mnt|workspace|scratch|cluster|gpfs|lustre|"
        r"private(?:_data)?)/[^\s\"']*",
        re.I,
    ),
    "ssh_endpoint": re.compile(r"\bssh\b[^\n]*\b[A-Za-z0-9._-]+@[A-Za-z0-9.-]+", re.I),
    "ssh_key": re.compile(r"BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "openai_token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{16,}=*", re.I),
    "assigned_secret": re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
        r"password|passwd)\b\s*[:=]\s*[\"']?(?!\$\{|<)[^\s\"']{8,}",
        re.I,
    ),
}

FORBIDDEN_PARTS = {".git", ".env", ".pytest_cache", "__pycache__"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".pem", ".key", ".log"}


def _path_problem(relative: Path) -> str | None:
    lowered_parts = {part.lower() for part in relative.parts}
    if lowered_parts & FORBIDDEN_PARTS:
        return "forbidden_artifact"
    if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
        return "forbidden_artifact"
    if any(part.lower().endswith(".egg-info") for part in relative.parts):
        return "build_metadata"
    return None


def main(root: Path) -> int:
    findings = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        problem = _path_problem(relative)
        if problem:
            findings.append((relative, 0, problem))
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    member_problem = _path_problem(Path(member))
                    if member_problem:
                        findings.append((relative, 0, f"archive_{member_problem}"))
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append((relative, line_number, label))
    if findings:
        for path, line_number, label in findings:
            print(f"{path}:{line_number}: {label}")
        return 1
    print("Anonymity scan passed.")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    raise SystemExit(main(target))
