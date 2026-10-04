"""Live recognition + validated first-round Supervisor semantics on frozen gold.

No domain Agent or business tool runs here. End-to-end evaluation is separate.
Gold, slices and primary labels are used only for scoring after model decisions.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    return round(values[round((len(values)-1)*fraction)], 3) if values else None


def summarize(rows):
    count = len(rows)
    tp = sum(len(set(row['confirmed']) & set(row['expected_intents'])) for row in rows)
    fp = sum(len(set(row['confirmed']) - set(row['expected_intents'])) for row in rows)
    fn = sum(len(set(row['expected_intents']) - set(row['confirmed'])) for row in rows)
    return {'cases': count, 'semantic_exact_count': sum(row['semantic_exact'] for row in rows),
            'semantic_exact_rate': sum(row['semantic_exact'] for row in rows)/count if count else None,
            'proposal_exact_count': sum(row['proposal_exact'] for row in rows),
            'primary_correct_count': sum(row['primary_correct'] for row in rows),
            'route_correct_count': sum(row['route_correct'] for row in rows),
            'false_positive_labels': fp, 'missed_labels': fn,
            'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None,
            'orchestrate_cases': sum(row['route']=='orchestrate' for row in rows),
            'single_over_split_cases': sum(row['route']=='orchestrate' and len(row['expected_intents'])==1 for row in rows),
            'errors': sum(bool(row['error']) for row in rows),
            'p50_ms': percentile([row['latency_ms'] for row in rows], .5),
            'p95_ms': percentile([row['latency_ms'] for row in rows], .95)}


async def run(args):
    root = args.source_root.resolve()
    for path in (root, root/'backend', root/'evaluation/benchmarks'):
        sys.path.insert(0, str(path))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    from dotenv import load_dotenv
    load_dotenv(root/'.env')
    from core.deepseek_client import load_deepseek_config
    from core.embedding_provider import BGEEmbeddingProvider, BGE_DEFAULT_MODEL, BGE_DEFAULT_REVISION
    from core.intent_embedding import IntentEmbeddingIndex
    from core.intent_pipeline import IntentRecognitionPipeline
    from core.supervisor_context import SupervisorContext
    from agents.agent_registry import AgentRegistration, AgentRegistry
    from agents.supervisor_lead import SupervisorLead
    from evaluate_end_to_end_tasks import GlobalUsageTap
    data = json.loads(args.fixture.read_text(encoding='utf-8'))
    if data['metadata'].get('frozen') is not True or data['metadata'].get('production_evidence') is not False:
        raise ValueError('requires frozen offline gold')
    cases = data['cases'][:args.limit or None]
    config = load_deepseek_config()
    context = SupervisorContext(config['api_key'], base_url=config['base_url'], model=config['model'])
    index = IntentEmbeddingIndex(BGEEmbeddingProvider(os.getenv('INTENT_EMBEDDING_MODEL', BGE_DEFAULT_MODEL),
        revision=os.getenv('INTENT_EMBEDDING_REVISION', BGE_DEFAULT_REVISION),
        device=os.getenv('INTENT_EMBEDDING_DEVICE') or None))
    pipeline = IntentRecognitionPipeline(context, embedding_index=index,
        intent_fusion_alpha=args.alpha, intent_clear_threshold=args.clear_threshold)
    class UnusedAgent:
        async def handle(self, request):
            raise AssertionError('semantic evaluation must never execute a domain Agent')
    registry = AgentRegistry(AgentRegistration(name, name, UnusedAgent(), name)
                             for name in ('subscription', 'billing', 'support'))
    supervisor = SupervisorLead(context, agent_registry=registry)
    paths = sorted((root/'backend').rglob('*.py'))
    hashes = {str(path.relative_to(root)): digest(path) for path in paths}
    fixture_hash = digest(args.fixture)
    tap, rows = GlobalUsageTap(), []
    journal = args.output.with_suffix('.observations.jsonl')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or journal.exists():
        raise ValueError('reports are write-once; choose a new output')
    await index.preload()
    tap.install()
    try:
        for number, case in enumerate(cases, 1):
            started = time.perf_counter()
            row = {'id': case['id'], 'slice': case['slice'], 'message': case['message'],
                   'expected_intents': case['expected_intents'], 'primary_intents': case['primary_intents'],
                   'expected_scope': case.get('expected_scope','in_scope'), 'error': '',
                   'confirmed': [], 'proposed': [], 'primary': '', 'route': '', 'scope': ''}
            try:
                outcome = await pipeline.recognize(case['message'], case_state=case.get('case_state'),
                    history=case.get('history'), context=str(case.get('context',data['metadata']['default_context'])))
                row['recognition'] = outcome.to_dict()
                row['route'], row['scope'], row['status'] = outcome.route, outcome.analysis.scope_status.value if outcome.analysis else '', outcome.status
                row['proposed'] = [item.label.value for item in outcome.analysis.intents] if outcome.analysis else []
                row['confirmed'] = [item.label.value for item in outcome.execution_analysis.intents] if outcome.execution_analysis else []
                row['primary'] = row['confirmed'][0] if row['confirmed'] else ''
                # Baseline routes every ready primary to review; new business routes are final semantics.
                baseline = 'decompose_requests' not in inspect.signature(supervisor._decide).parameters
                compound = outcome.route == 'orchestrate'
                if outcome.status == 'ready' and (baseline or compound):
                    tool_state = {'consultation_only': True, 'frozen_payload': outcome.to_dict()}
                    kwargs = dict(analysis=outcome.execution_analysis, intent_rows=outcome.execution_analysis.intent_rows,
                        stages=[], history=case.get('history'), case_state=case.get('case_state') or {},
                        context=str(case.get('context',data['metadata']['default_context'])), round_index=1,
                        team=registry.prompt_team(), conversation=[], seen_calls=set(), decision_errors=[],
                        retrieval=outcome.retrieval, intent_tool_state=tool_state, frozen_semantics=False,
                        source_analysis=outcome.analysis, intent_confidence=outcome.confidence, intent_review_required=True,
                        handoff_confirmation_intent_ids=())
                    if not baseline:
                        kwargs['decompose_requests'] = True
                    decision = await supervisor._next_decision(case['message'], **kwargs)
                    analysis = decision.analysis
                    fusion = (await supervisor._assess_decomposed_intents(case['message'], analysis, pipeline.fusion_policy,index)
                              if compound else supervisor._assess_reviewed_intents(case['message'],analysis,outcome.confidence,pipeline.fusion_policy))
                    row['supervisor_analysis'], row['fusion'] = analysis.to_dict(), fusion.to_dict()
                    row['proposed'] = [item.label.value for item in analysis.intents]
                    row['confirmed'] = [item.label.value for item in fusion.confirmed]
                    row['scope'] = analysis.scope_status.value
                    if compound:
                        row['primary'] = next((item.label.value for item in analysis.intents if item.intent_id==analysis.primary_intent_id),'')
            except Exception as ex:
                row['error'] = f'{type(ex).__name__}: {str(ex)[:350]}'
            expected = set(row['expected_intents'])
            valid = not row['error'] and row['scope']==row['expected_scope']
            row['semantic_exact'] = valid and set(row['confirmed'])==expected
            row['proposal_exact'] = valid and set(row['proposed'])==expected
            row['primary_correct'] = valid and (row['primary'] in row['primary_intents'] if expected else not row['primary'])
            expected_route = 'orchestrate' if len(expected)>1 else next(iter(expected),'')
            row['route_correct'] = valid and row['route']==expected_route
            row['latency_ms'] = round((time.perf_counter()-started)*1000,3)
            rows.append(row)
            with journal.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(f"[{number}/{len(cases)}] {row['id']} route={row['route']} semantic={row['semantic_exact']} error={row['error'][:85]}",flush=True)
    finally:
        tap.uninstall()
        await context.client.close()
    report = {'meta': {'production_evidence':False,'model':config['model'],'source_root':str(root),
                      'fixture':str(args.fixture),'fixture_sha256':fixture_hash,'source_hashes':hashes,
                      'scope':'live routing and first-round validated semantics; domain execution not_run',
                      'parameters':{'alpha':args.alpha,'clear_threshold':args.clear_threshold,'low_threshold':.4},'repeats':1},
              'summary':summarize(rows),'slices':{key:summarize([row for row in rows if row['slice']==key])
                       for key in sorted({row['slice'] for row in rows})},'usage':tap.summary(),'rows':rows}
    changed = [str(path.relative_to(root)) for path in paths if digest(path)!=hashes[str(path.relative_to(root))]]
    report['meta']['source_changed_during_run'] = changed
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],ensure_ascii=False),flush=True)
    if changed or digest(args.fixture)!=fixture_hash:
        raise RuntimeError('source or fixture changed during evaluation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--fixture',type=Path,default=Path(__file__).resolve().parent/'fixtures/intent_natural_comparison_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--alpha',type=float,default=.1)
    parser.add_argument('--clear-threshold',type=float,default=.7)
    args=parser.parse_args()
    asyncio.run(run(args))


if __name__=='__main__':
    main()
