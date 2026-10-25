"""训练轨迹的程序边关联证据；不推断因果，不把任务成败直接当作边奖励。"""
from collections import Counter

from src.pg_evidence import project_sample_evidence
from src.procedural_graph import fingerprint


def edge_credit(graph, report, *, train_cases):
    manifest = report.get('manifest', {})
    mechanism = next((m for m in manifest.get('harness_profile', {}).get('mechanisms', [])
                      if m.get('id') == 'procedural-graph-context'), {})
    samples = report.get('samples', [])
    if (manifest.get('split') != 'dev' or not samples
            or mechanism.get('configuration', {}).get('graph_sha256') != graph.fingerprint
            or any(s.get('split') != 'dev' or s.get('case_id') not in train_cases for s in samples)):
        raise ValueError('贡献分析只接受冻结训练 dev 和匹配图身份')
    if len({(s['case_id'], s['repetition']) for s in samples}) != len(samples):
        raise ValueError('贡献分析不能重复计算同一样本')
    rows = {edge.id: dict(edge_id=edge.id, edge_sha256=fingerprint(edge.model_dump(mode='json')),
                         support=0, counter=0, unknown=0, evidence=[], contrasts=[]) for edge in graph.edges}
    sample_refs = []
    for sample in samples:
        sample_ref = f"{sample['case_id']}:{sample['repetition']}"
        sample_refs.append(sample_ref)
        projection = project_sample_evidence(sample)
        for turn in projection['turns']:
            raw_events = sample['turns'][turn['turn_index']].get('result', {}).get('events', [])
            for intervention in turn['interventions']:
                progress = intervention.get('action_progress') or {}
                outcome = next((e for e in raw_events if e.get('type') == 'pg_action_outcome'
                                and e.get('outcome_id') == progress.get('outcome_id')), {})
                guidance = next((e for e in raw_events if e.get('type') == 'pg_guidance'
                                 and e.get('sequence') == intervention['guidance']['sequence']), {})
                review = next((e for e in raw_events if e.get('type') == 'pg_decision_reviewed'
                               and e.get('review_id') == outcome.get('review_id')
                               and e.get('decision_key') == guidance.get('decision_key')), {})
                applicable = {e.get('edge_id'): e.get('status')
                              for e in intervention.get('applicability', {}).get('evaluations', [])}
                considered = set(intervention['retrieval']['retrieved_edge_ids']) | set(applicable)
                alignment = intervention['action_alignment']
                for edge_id in sorted(considered & rows.keys()):
                    classification, reason = 'UNKNOWN', 'INSUFFICIENT_CHAIN'
                    exposed = intervention['guidance']['exposed'] and edge_id in intervention['retrieval']['selected_edge_ids']
                    adopted = alignment['status'] == 'ALIGNED' and edge_id in alignment['matched_edge_ids']
                    known = (guidance.get('graph_sha256') == graph.fingerprint
                             and applicable.get(edge_id) == 'TRUE' and exposed and adopted
                             and review.get('status') == 'ALIGNED'
                             and edge_id in (review.get('matched_edge_ids') or [])
                             and outcome.get('schema_version') == 2)
                    if known:
                        info = outcome.get('information_gain') or {}
                        if (outcome.get('learning_eligible') is True and outcome.get('progress_status') == 'PROGRESSED'
                                and info.get('status') == 'NEW_INFORMATION'):
                            classification, reason = 'SUPPORT', 'ADOPTED_INFORMATION_GAIN'
                        elif outcome.get('progress_status') == 'NO_PROGRESS' and info.get('status') == 'NO_NEW_INFORMATION':
                            classification, reason = 'COUNTER', 'ADOPTED_WITHOUT_INFORMATION'
                        else:
                            reason = 'NO_ATTRIBUTABLE_INFORMATION'
                    elif not exposed:
                        reason = 'NOT_EXPOSED'
                    elif applicable.get(edge_id) != 'TRUE':
                        reason = 'CONDITION_NOT_PROVEN'
                    elif not adopted:
                        reason = 'NOT_ADOPTED'
                    # 只读取同一目标修订中、该行动之后的真实核验；不把总任务成功当作核验事件。
                    later_verified = set()
                    if outcome:
                        for event in raw_events:
                            if (event.get('type') == 'pg_action_outcome' and event.get('schema_version') == 2
                                    and event.get('goal_revision') == outcome.get('goal_revision')
                                    and event.get('sequence', -1) >= outcome.get('sequence', 0)):
                                later_verified.update((event.get('information_gain') or {}).get('newly_verified_objective_ids') or [])
                    evidence = dict(sample_ref=sample_ref, case_id=sample['case_id'], repetition=sample['repetition'],
                        turn_index=turn['turn_index'], intervention_id=intervention['intervention_id'],
                        outcome_id=outcome.get('outcome_id'), guidance_sequence=intervention['guidance']['sequence'],
                        condition=applicable.get(edge_id, 'UNKNOWN'), exposed=exposed, adopted=adopted,
                        information_gain=(outcome.get('information_gain') or {}).get('status', 'UNKNOWN'),
                        later_verified_objective_ids=sorted(later_verified), task_passed=sample['passed'],
                        classification=classification, reason=reason)
                    evidence['evidence_id'] = fingerprint({'graph': graph.fingerprint, 'edge': edge_id, **evidence})
                    rows[edge_id]['evidence'].append(evidence)
                    rows[edge_id][classification.lower()] += 1
    for row in rows.values():
        row['classification'] = ('MIXED' if row['support'] and row['counter'] else
                                 'SUPPORT' if row['support'] else 'COUNTER' if row['counter'] else 'UNKNOWN')
        row['case_count'] = len({e['case_id'] for e in row['evidence']})
        for case_id in sorted(train_cases):
            case_samples = [s for s in samples if s['case_id'] == case_id]
            if not any(s['passed'] for s in case_samples) or not any(not s['passed'] for s in case_samples):
                continue
            def group(passed):
                refs = {f"{s['case_id']}:{s['repetition']}" for s in case_samples if s['passed'] == passed}
                counts = Counter(e['classification'] for e in row['evidence'] if e['sample_ref'] in refs)
                return dict(samples=sorted(refs), counts=dict(counts))
            row['contrasts'].append(dict(case_id=case_id, successful=group(True), failed=group(False)))
    result = dict(schema_version=1, method='trajectory-edge-association-v1', causal_claim=False,
                  graph_id=graph.graph_id, graph_version=graph.version,
                  graph_sha256=graph.fingerprint, report_sha256=fingerprint(report), split='dev',
                  sample_refs=sorted(sample_refs), edges=list(rows.values()))
    return {**result, 'credit_sha256': fingerprint(result)}
