"""Package the reviewed DeFT sources for import into a GitHub repository."""
import hashlib
import json
import zipfile
from pathlib import Path

from build_supplement import ROOT, release_files


def replace_once(raw, old, new):
    original = raw.decode('utf-8')
    assert original.count(old) == 1, old
    return original.replace(old, new).encode('utf-8')


def public_text(path, raw):
    if str(path) == 'docs/VALIDATION.md':
        raw = replace_once(raw, '- Supplement builder excludes model weights, datasets, caches, build artifacts,',
                           '- Source packaging excludes model weights, datasets, caches, build artifacts,')
        raw = replace_once(raw, '## Limits / before final submission', '## Validation limits')
        raw = replace_once(
            raw,
            '- Review new-code licensing and conference anonymization rules before public\n'
            '  release or final submission. No GitHub upload or conference submission has\n'
            '  been made by the packaging tool.',
            '- The source package does not contain model weights, benchmark data or every\n'
            '  exploratory analysis. Check the original licenses before redistributing\n'
            '  third-party checkpoints or datasets.',
        )
    elif str(path) == 'THIRD_PARTY_NOTICES.md':
        raw = replace_once(
            raw,
            'Required attribution is not removed\nfor anonymization. Authors should review licensing of their new code and all\n'
            'upstream-derived material before making a public release.',
            'Required upstream attribution is retained. Third-party weights and datasets\n'
            'are subject to their own licenses.',
        )
    return raw


def main():
    files = [(path, public_text(path, raw)) for path, raw in release_files()
             if str(path) != 'docs/ANONYMITY.md']
    manifest = {str(path): hashlib.sha256(raw).hexdigest() for path, raw in files}
    destination = ROOT / 'dist/DeFT_github_source.zip'
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, raw in files + [(Path('SHA256SUMS.json'), (json.dumps(manifest, indent=2) + '\n').encode())]:
            info = zipfile.ZipInfo(str(path), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert all(archive.read(str(name)) == raw for name, raw in files)
    print(destination, len(files), 'source files')


if __name__ == '__main__':
    main()
