"""Download and verify a ModelScope ``per_episode`` dataset export.

The token is read from ``MODELSCOPE_TOKEN`` or an interactive prompt and is
never written to disk.  All output locations are configurable so this script
works from a fresh clone or from a separate data volume.
"""
import argparse
import concurrent.futures
import getpass
import hashlib
import json
import os
import tarfile
from pathlib import Path
def _hub_api_class():
    try:
        from modelscope_hub import HubApi
        return HubApi
    except ImportError:
        try:
            from modelscope.hub.api import HubApi
            return HubApi
        except ImportError as exc:
            raise SystemExit(
                'Install the ModelScope SDK first: python -m pip install modelscope'
            ) from exc


def _as_path(value) -> Path:
    """Normalize SDK download results across ModelScope SDK versions."""
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f'ModelScope returned a missing file: {path}')
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--repo',
        default=os.environ.get('MODELSCOPE_DATASET_REPO', 'XinyaoZhuang/nero_classify_objects'),
        help='ModelScope dataset repository (default: %(default)s)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(os.environ.get('DATA_ROOT', 'data')) / 'modelscope',
        help='Directory for downloaded files and the verified archive',
    )
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--expected-files', type=int, default=None)
    args = parser.parse_args()

    repo = args.repo
    base = args.output_dir.expanduser().resolve() / repo.rsplit('/', 1)[-1]
    base.mkdir(parents=True, exist_ok=True)
    token = os.environ.get('MODELSCOPE_TOKEN') or getpass.getpass('ModelScope access token: ')
    api = _hub_api_class()(token=token)
    files = [
        f for f in api.list_repo_files(repo, 'dataset')
        if f.path.startswith('per_episode/')
        and f.type not in ('tree', 'folder', 'dir')
    ]
    expected_files = args.expected_files
    if expected_files is None:
        expected_files = int(os.environ.get('MODELSCOPE_DATASET_EXPECTED_FILES', '0')) or None
    if expected_files is not None:
        assert len(files) == expected_files, (
            f'Expected {expected_files} validated files, got {len(files)}'
        )

    def fetch(item):
        path = _as_path(api.download_file(
            repo, 'dataset', item.path, local_dir=str(base / 'modelscope_repo')
        ))
        assert path.stat().st_size == item.size, item.path
        return item.path

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, path in enumerate(pool.map(fetch, files), 1):
            if i % 25 == 0 or i == len(files):
                print(f'Downloaded {i}/{len(files)} files', flush=True)

    root = base / 'modelscope_repo' / 'per_episode'
    manifest = json.loads((root / 'file_manifest.json').read_text())
    for item in manifest:
        path = root / item['path']
        assert path.stat().st_size == item['size'], item['path']
        assert _sha256_file(path) == item['sha256'], item['path']
    archive = base / f'{repo.rsplit("/", 1)[-1]}_per_episode.tar'
    with tarfile.open(archive, 'w') as tar:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                tar.add(
                    path,
                    arcname=str(
                        Path(f'{repo.rsplit("/", 1)[-1]}_per_episode')
                        / path.relative_to(root)
                    ),
                )
    digest = _sha256_file(archive)
    (base / 'dataset_archive.sha256').write_text(digest + '\n', encoding='utf-8')
    print('DATASET READY:', archive, 'sha256:', digest, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
