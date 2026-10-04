"""Offline operating-point search, with a frozen calibration/validation split.

No model calls, no fixture relabeling, and no writes to runtime configuration.
This tunes thresholds on scores; it does not turn scores into probabilities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


GRID_ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.50]
GRID_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def frozen_split(cases):
    calibration, validation = [], []
    groups = {}
    for case in cases:
        route = ('orchestrate' if len(case['expected_intents']) > 1 else
                 case['expected_intents'][0] if case['expected_intents'] else case.get('expected_scope', 'uncertain'))
        groups.setdefault(route, []).append(case['id'])
    for ids in groups.values():
        for index, case_id in enumerate(sorted(ids)):
            (validation if index % 3 == 1 else calibration).append(case_id)
    return {'calibration_ids': sorted(calibration), 'validation_ids': sorted(validation)}


def assess(rows, ids, alpha, threshold):
    correct, wrong, missing, eligible = 0, 0, 0, 0
    for row in rows:
        if row['id'] not in ids:
            continue
        gold = row['expected_intents']
        expected = 'orchestrate' if len(gold) > 1 else gold[0] if gold else ''
        eligible += bool(expected)
        rec = row['recognition']
        accepted = ''
        if rec.get('analysis', {}).get('scope_status') == 'in_scope':
            signal = rec.get('route_fusion') if rec.get('route') == 'orchestrate' else next(iter((rec.get('fusion') or {}).get('decisions', [])), {})
            if signal and signal.get('reason_code') not in {'explicit_negation_conflict', 'background_only_conflict'}:
                active_alpha = 0.0 if signal.get('degraded') or (rec.get('fusion') or {}).get('degraded') else alpha
                score = active_alpha * signal['embedding_score'] + (1 - active_alpha) * signal['tree_score']
                if score >= threshold:
                    accepted = rec.get('route') or ''
        if accepted:
            if accepted == expected and row['expected_scope'] == 'in_scope':
                correct += 1
            else:
                wrong += 1
        elif expected:
            missing += 1
    return {'correct_routes': correct, 'wrong_routes': wrong, 'missed_routes': missing,
            'precision': correct / (correct + wrong) if correct + wrong else 0,
            'recall': correct / eligible if eligible else 0}


def calibrate(rows, split):
    baseline = assess(rows, split['calibration_ids'], .10, .70)
    baseline_validation = assess(rows, split['validation_ids'], .10, .70)
    candidates = []
    for alpha in GRID_ALPHAS:
        for threshold in GRID_THRESHOLDS:
            result = assess(rows, split['calibration_ids'], alpha, threshold)
            candidates.append(dict(alpha=alpha, clear_threshold=threshold, **result))
    safe = [row for row in candidates if row['wrong_routes'] <= baseline['wrong_routes']
            and row['precision'] >= baseline['precision']]
    chosen = max(safe, key=lambda row: (row['correct_routes'], row['precision'],
                                      -abs(row['clear_threshold']-.70), -abs(row['alpha']-.10)))
    validation = assess(rows, split['validation_ids'], chosen['alpha'], chosen['clear_threshold'])
    return {'baseline_calibration': baseline, 'baseline_validation': baseline_validation,
            'candidate': chosen, 'candidate_validation': validation, 'grid': candidates,
            'validation_accepted': validation['wrong_routes'] <= baseline_validation['wrong_routes']
             and validation['precision'] >= baseline_validation['precision']
             and validation['correct_routes'] >= baseline_validation['correct_routes'],
            'boundary': 'route-level offline evidence only; end-to-end and independent business review required'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding='utf-8'))
    sha = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    if not args.report:
        if args.split.exists():
            raise ValueError('split is write-once')
        split = dict(frozen_split(fixture['cases']), fixture_sha256=sha, frozen_before_predictions=True,
                     grid_alphas=GRID_ALPHAS, grid_thresholds=GRID_THRESHOLDS)
        args.split.write_text(json.dumps(split, indent=2), encoding='utf-8')
        print({key: len(split[key]) for key in ('calibration_ids', 'validation_ids')})
        return
    split = json.loads(args.split.read_text(encoding='utf-8'))
    if split['fixture_sha256'] != sha:
        raise ValueError('fixture changed after split freeze')
    report = json.loads(args.report.read_text(encoding='utf-8'))
    if {row['id'] for row in report['rows']} != set(split['calibration_ids'] + split['validation_ids']):
        raise ValueError('prediction coverage differs from frozen split')
    output = calibrate(report['rows'], split)
    output.update(fixture_sha256=sha, predictions_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
                  split_sha256=hashlib.sha256(args.split.read_bytes()).hexdigest(), production_evidence=False)
    if not args.output or args.output.exists():
        raise ValueError('choose a new output')
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in output.items() if key != 'grid'}))


if __name__ == '__main__':
    main()
