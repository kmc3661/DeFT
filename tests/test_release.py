import importlib.util
import json
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_latency_manifest():
    rows=[json.loads(line) for line in (ROOT/'configs/infovqa_latency_100.jsonl').read_text().splitlines()]
    assert len(rows)==100
    assert len({r['image_group'] for r in rows})==100
    assert len({r['question_id'] for r in rows})==100


def test_release_scan():
    spec=importlib.util.spec_from_file_location('release_builder',ROOT/'tools/build_supplement.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files=list(module.release_files())
    assert any(str(p)=='README.md' for p,_ in files)
    assert any(str(p)=='licenses/lmms-eval-LICENSE' for p,_ in files)
    assert not any(p.suffix in ('.bin','.safetensors','.pt') for p,_ in files)


def test_archive_metadata_and_determinism(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('release_builder',ROOT/'tools/build_supplement.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'README.md').write_text('Anonymous test source.\n')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['build_supplement.py'])
    module.main()
    archive=tmp_path/'dist/DeFT_supplement.zip'
    first=archive.read_bytes()
    module.main()
    assert archive.read_bytes()==first
    with zipfile.ZipFile(archive) as z:
        assert not z.comment
        assert 'DeFT/SHA256SUMS.json' in z.namelist()
        for info in z.infolist():
            assert info.date_time==(1980,1,1,0,0,0)
            assert not info.extra and not info.comment
            assert info.external_attr>>16==0o100644
