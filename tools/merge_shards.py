"""Merge disjoint generation shards in caller-provided order, failing on overlap."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("shards",type=Path,nargs="+")
    p.add_argument("--expected",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(line) for f in a.shards for line in f.read_text().splitlines() if line.strip()]
    ids=[(str(r['question_id']),r['prompt']) for r in rows]
    if len(rows)!=a.expected or len(set(ids))!=len(ids):
        p.error("Unexpected row count or duplicate IDs/prompts")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as out:
        for r in rows:out.write(json.dumps(r,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
