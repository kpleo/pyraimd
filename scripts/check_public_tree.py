"""Check the staged index or a Git tree before publication (standard library only)."""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import PurePosixPath

ROOT_FILES = {
    '.gitignore', '.python-version', 'pyproject.toml', 'uv.lock',
    'README.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 'LICENSE',
}
ROOT_DIRS = {'.github', '.githooks', 'scripts', 'src', 'tests', 'examples',
             'docs', 'reproducibility'}
PUBLIC_DOCS = {'docs/api.md', 'docs/architecture.md', 'docs/configuration.md',
               'docs/energetic_force_error.md'}
PRIVATE_PARTS = {'manuscript', 'development_reports', 'private',
                 '.env', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache'}
GENERATED_SUFFIXES = {'.tex', '.pdf', '.png', '.jpg', '.jpeg', '.tiff',
                      '.db', '.sqlite', '.pt', '.pth', '.log', '.zip'}
# Reviewed supplement: names, modes, and blob identities, in Git order.
SUPPLEMENT_SHA256 = 'a13963653f61c825bc8da55af6783a3df4e472e9266c2270bf81f7194e2535c8'
TEXT_PATTERNS = {
    'personal or site filesystem path': re.compile(
        rb'/(?:Users|home|data/home|public[0-9]*|lustre[0-9]*)/[^\s"\x27<>]+'),
    'private key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'access token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,}|AKIA[A-Z0-9]{16})'),
    'internal working-document link': re.compile(
        rb'(?:docs[/]development_reports/|docs[/]pyramid_development_plan_|hpc[/]neimeng/)'),
}


def git(*args: str) -> bytes:
    return subprocess.check_output(['git', *args])


def entries(staged: bool, ref: str) -> list[tuple[str, str, str]]:
    result = []
    if staged:
        records = git('ls-files', '--stage', '-z').split(b'\0')
        for record in filter(None, records):
            header, path = record.split(b'\t', 1)
            mode, oid, stage = header.decode().split()
            if stage != '0':
                raise ValueError('resolve merge conflicts before checking publication')
            result.append((mode, oid, path.decode()))
    else:
        for record in filter(None, git('ls-tree', '-rz', '--full-tree', ref).split(b'\0')):
            header, path = record.split(b'\t', 1)
            mode, _kind, oid = header.decode().split()
            result.append((mode, oid, path.decode()))
    return result


def check(records: list[tuple[str, str, str]]) -> list[str]:
    errors = []
    supplement = b''
    # Batch-read blobs without printing matched sensitive content.
    oids = [oid for mode, oid, path in records if mode in {'100644', '100755'}]
    proc = subprocess.run(['git', 'cat-file', '--batch'],
                          input=('\n'.join(oids) + '\n').encode(),
                          stdout=subprocess.PIPE, check=True)
    blobs = {}
    offset = 0
    for oid in oids:
        end = proc.stdout.index(b'\n', offset)
        header = proc.stdout[offset:end].split()
        size = int(header[2])
        offset = end + 1
        blobs[oid] = proc.stdout[offset:offset + size]
        offset += size + 1
    for mode, oid, name in records:
        path = PurePosixPath(name)
        if path.parts[0] == 'reproducibility':
            supplement += f'{mode} blob {oid}\t{name}\n'.encode()
            continue
        if mode not in {'100644', '100755'}:
            errors.append(f'{name}: symlinks and submodules require explicit review')
            continue
        if name not in ROOT_FILES and path.parts[0] not in ROOT_DIRS:
            errors.append(f'{name}: outside the public file allowlist')
        if path.parts[0] == 'docs' and name not in PUBLIC_DOCS:
            errors.append(f'{name}: not an approved user document')
        if PRIVATE_PARTS.intersection(path.parts) or path.suffix.lower() in GENERATED_SUFFIXES:
            errors.append(f'{name}: working material or generated artifact')
        data = blobs[oid]
        if b'\0' in data:
            errors.append(f'{name}: binary content requires explicit review')
            continue
        for label, pattern in TEXT_PATTERNS.items():
            match = pattern.search(data)
            if match:
                line = data[:match.start()].count(b'\n') + 1
                errors.append(f'{name}:{line}: {label}')
    if hashlib.sha256(supplement).hexdigest() != SUPPLEMENT_SHA256:
        errors.append('reproducibility/: differs from the approved frozen supplement')
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--staged', action='store_true')
    group.add_argument('--ref', default='HEAD')
    args = parser.parse_args()
    try:
        errors = check(entries(args.staged, args.ref))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'Public tree check could not complete: {type(exc).__name__}', file=sys.stderr)
        return 2
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print('Public tree check passed; frozen supplement unchanged.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
