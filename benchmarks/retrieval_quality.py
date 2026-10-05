"""Offline, synthetic diagnostic benchmark. No API calls or private knowledge.
Run: python benchmarks/retrieval_quality.py --baseline 82f1fc4
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.retrieval import HybridRetriever, BM25Index
from core.schemas import EvidenceChunk


def chunk(source, text):
    return EvidenceChunk(source + ':0', source, source, 'Policy', text, 0)


def quality(factory):
    docs = [
        chunk('repair', 'งานซ่อมต้องเสร็จภายใน 24 ชั่วโมง ต้องแจ้งลูกค้าก่อนเข้าพื้นที่'),
        chunk('install', 'งานติดตั้งต้องนัดหมายล่วงหน้า ลูกค้าต้องยืนยันวันติดตั้ง'),
        chunk('p1', 'P1 accident review required'),
        chunk('p2', 'P2 accident disqualifies the team'),
        chunk('repeat', 'Repeat fault is measured within 30 days'),
        chunk('csi', 'CSI survey incomplete use regional average'),
        *[chunk(f'distractor{i}', 'repair SLA') for i in range(40)],
        chunk('selected', 'repair SLA ' + 'other ' * 100),
    ]
    cases = [
        ('ซ่อมเสร็จภายในกี่ชั่วโมง', None, 'repair'),
        ('นัดหมายติดตั้ง', None, 'install'),
        ('P2 accident', None, 'p2'),
        ('P1 review', None, 'p1'),
        ('Repeat fault 30 days', None, 'repeat'),
        ('CSI survey regional average', None, 'csi'),
        ('repair SLA', {'selected'}, 'selected'),
        ('repair SLA', set(), None),
        ('zzzzunknown', None, None),
    ]
    retriever = factory(docs, semantic_enabled=False, model_name='unused')
    rows=[]
    for query, scope, expected in cases:
        results=retriever.search(query, allowed_source_ids=scope)
        actual=results[0].source_id if results else None
        rows.append(dict(query=query, expected=expected, actual=actual, passed=actual==expected))
    return dict(passed=sum(x['passed'] for x in rows), total=len(rows), cases=rows)


def latency(index_class, count=5000, repeats=40):
    # Sparse identifier lookup: realistic benefit of postings, not dense common terms.
    def term(i):
        return 'item' + ''.join(chr(97 + (i // 26**j) % 26) for j in range(4))
    docs=[chunk(f'doc{i}', f'Procedure {term(i)} ข้อมูลขั้นตอนสำหรับงาน {i}') for i in range(count)]
    index=index_class(docs)
    durations=[]
    for i in range(repeats):
        start=time.perf_counter()
        index.search(term((i*107)%count), top_k=6)
        durations.append((time.perf_counter()-start)*1000)
    return dict(chunks=count, queries=repeats, median_ms=round(statistics.median(durations),3), p95_ms=round(sorted(durations)[int(.95*len(durations))-1],3))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline')
    args=parser.parse_args()
    result={'scope':'Synthetic lexical diagnostics; not live LLM or NotebookLM parity', 'current':{'quality':quality(HybridRetriever), 'latency':latency(BM25Index)}}
    if args.baseline:
        code=subprocess.check_output(['git','show',f'{args.baseline}:core/retrieval.py'],cwd=ROOT,text=True)
        baseline=types.ModuleType('baseline_retrieval')
        exec(compile(code,'baseline_retrieval.py','exec'),baseline.__dict__)
        result['baseline_revision']=args.baseline
        result['baseline']={'quality':quality(baseline.HybridRetriever), 'latency':latency(baseline.BM25Index)}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
