"""Build the paper's no-OCR TextVQA prompt file from official val annotations."""
import argparse,json,shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():p.error('Output directory already exists')
    rows=json.loads(a.annotations.read_text())['data']
    if len(rows)!=5000:p.error('Expected 5000 official TextVQA validation questions')
    a.output_dir.mkdir(parents=True)
    shutil.copy2(a.annotations,a.output_dir/'TextVQA_0.5.1_val.json')
    with (a.output_dir/'questions.jsonl').open('x') as f:
        for r in rows:
            out=dict(question_id=str(r['image_id']),source_question_id=r['question_id'],
                image=str(r['image_id'])+'.jpg',
                text=r['question']+'\nAnswer the question using a single word or phrase.',category='default')
            f.write(json.dumps(out,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
