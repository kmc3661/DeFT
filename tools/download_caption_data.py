"""Download the pinned public caption parquet snapshots, without model weights."""
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    for repo, revision in (
        ("lmms-lab/NoCaps", "a26b3fe1e0021164ec430c57c48085d58f2fe922"),
        ("lmms-lab/TextCaps", "e9e223338832318a6161d04648871c18a024ce85"),
    ):
        snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                          allow_patterns=["data/*.parquet", "README.md"])
