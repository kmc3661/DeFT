"""Create an anonymous source-only ZIP with per-file SHA-256 integrity records."""
import argparse,hashlib,json,re,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'.git','.pytest_cache','__pycache__','build','dist','outputs','data','checkpoints','.cache','.venv'}
SUFFIXES={'.py','.md','.toml','.json','.jsonl','.txt','.cpp','.cc','.h','.cu'}


def release_files(deny_patterns=()):
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if p.is_symlink():raise RuntimeError(f'Symlink not allowed: {rel}')
        if not p.is_file() or any(x in EXCLUDE or x.endswith('.egg-info') for x in rel.parts):continue
        if p.suffix not in SUFFIXES and p.name != '.gitignore' and not p.name.endswith('LICENSE'):continue
        raw=p.read_bytes()
        # Upstream copyright/license names are intentionally preserved.
        patterns=(
            rb'(?<![A-Za-z0-9])/(?:home|Users|SSD[0-9]+)/[A-Za-z0-9_.-]+',
            rb'BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY',
            rb'\bhf_[A-Za-z0-9]{20,}\b',
            rb'\bgh[pousr]_[A-Za-z0-9]{20,}\b',
            *deny_patterns,
        )
        for pattern in patterns:
            if re.search(pattern,raw,re.IGNORECASE):
                raise RuntimeError(f'Private identifier/path/secret marker in {rel}')
        yield rel,raw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deny-pattern',action='append',default=[],
                        help='Local-only identity regex; never saved in the archive')
    args=parser.parse_args()
    files=list(release_files(tuple(p.encode() for p in args.deny_pattern)))
    manifest={str(p):hashlib.sha256(raw).hexdigest() for p,raw in files}
    dest=ROOT/'dist/DeFT_supplement.zip';dest.parent.mkdir(exist_ok=True)
    # A fresh build intentionally replaces only this generated archive.
    contents=files+[(Path('SHA256SUMS.json'),(json.dumps(manifest,indent=2)+'\n').encode())]
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as out:
        for p,raw in contents:
            info=zipfile.ZipInfo('DeFT/'+str(p),date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.create_system=3
            info.external_attr=0o100644<<16
            info.extra=b'';info.comment=b''
            out.writestr(info,raw)
    with zipfile.ZipFile(dest) as check:
        assert check.testzip() is None and check.comment==b''
        for info in check.infolist():
            assert info.date_time==(1980,1,1,0,0,0)
            assert not info.extra and not info.comment
            assert info.external_attr>>16==0o100644
    print(f'{dest.name}: {len(files)} source files; {dest.stat().st_size} bytes')


if __name__=='__main__':main()
