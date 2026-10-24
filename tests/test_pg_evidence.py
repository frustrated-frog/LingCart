from src.pg_evidence import project_sample_evidence


def sample(*, passed=True, aligned=True):
    action = 'search_products' if aligned else 'clarify'
    events = [
        {'type': 'pg_guidance', 'sequence': 1, 'decision_key': 'decision-1',
         'graph_id': 'pg', 'graph_version': 1, 'graph_sha256': 'a' * 64,
         'anchor': 'Start', 'fallback': False, 'node_ids': ['Start', 'search_products'],
         'retrieved_edge_ids': ['start-search'], 'edge_ids': ['start-search'],
         'guidance': '先查询真实目录', 'status': 'READY',
         'state_snapshot': {'goal_revision': 0, 'unfinished_plan_steps': []},
         'subgraph': {'nodes': [{'id': 'Start', 'kind': 'STATE'}, {'id': 'search_products', 'kind': 'ACTION'}],
                      'edges': [{'id': 'start-search', 'source': 'Start', 'target': 'search_products'}]}},
        {'type': 'mechanism_applied', 'sequence': 2, 'decision_key': 'decision-1'},
        {'type': 'evidence_context', 'sequence': 3, 'pg_decision_key': 'decision-1', 'pg_status': 'EXPOSED'},
        {'type': 'adaptive_decision', 'sequence': 4, 'source': 'model',
         'decision': {'action': 'tool' if aligned else 'clarify', 'tool': action if aligned else None, 'args': {'category': '耳机'}}},
        {'type': 'tool_call', 'sequence': 5, 'tool': 'search_products', 'args': {'category': '耳机'}},
        {'type': 'tool_result', 'sequence': 6, 'tool': 'search_products', 'args': {'category': '耳机'},
         'status': 'SUCCEEDED', 'result': {'products': [{'product_id': 'P1'}]}},
        {'type': 'verification', 'sequence': 7, 'all_goals_satisfied': passed},
    ]
    return {'case_id': 'PG13', 'repetition': 1, 'passed': passed,
            'turns': [{'message': '推荐耳机', 'passed': passed,
                       'failures': [] if passed else ['目标未满足'],
                       'result': {'events': events, 'evaluation_environment': {'kind': 'empty_search', 'fired': True}}}]}


def test_projection_links_localization_guidance_action_tool_and_verification():
    projection = project_sample_evidence(sample())
    intervention = projection['turns'][0]['interventions'][0]
    assert intervention['localization']['anchor'] == 'Start'
    assert intervention['retrieval']['selected_edge_ids'] == ['start-search']
    assert intervention['guidance']['exposed'] is True
    assert intervention['action_alignment']['status'] == 'ALIGNED'
    assert intervention['tool']['result']['products'][0]['product_id'] == 'P1'
    assert intervention['verification']['all_goals_satisfied'] is True
    assert projection['summary']['recovered'] is True
    assert projection['summary']['localization_abstentions'] == 0
    assert projection['summary']['shadow_differences'] == 0


def test_projection_classifies_failed_guidance_action_divergence_without_claiming_reasoning():
    projection = project_sample_evidence(sample(passed=False, aligned=False))
    assert projection['summary']['failure_stage'] == 'GUIDANCE_ACTION_DIVERGENCE'
    assert projection['summary']['diverged'] == 1


def test_baseline_failure_is_distinguished_from_missing_pg_evidence():
    projection = project_sample_evidence({'case_id': 'PG13', 'repetition': 1, 'passed': False,
        'turns': [{'message': '推荐耳机', 'passed': False, 'failures': ['失败'], 'result': {'events': []}}]})
    assert projection['summary']['failure_stage'] == 'BASELINE_TASK_FAILURE'


def test_projection_preserves_v2_abstention_and_shadow_difference_without_calling_it_fallback():
    value = sample(passed=False)
    guidance = value['turns'][0]['result']['events'][0]
    guidance.update(anchor=None, fallback=False, abstained=True, node_ids=[], retrieved_edge_ids=[], edge_ids=[],
                    guidance='', status='SKIPPED', reason='LOCALIZATION_ABSTAINED', subgraph=None,
                    localization={'algorithm': 'trajectory-state-v2', 'status': 'ABSTAINED',
                                  'reason': 'NO_CANDIDATE', 'confidence': 0,
                                  'candidates': [], 'excluded_edge_ids': [],
                                  'shadow': {'algorithm': 'action-anchor-v1', 'anchor': 'answer_faq',
                                             'fallback': True, 'node_ids': [], 'edge_ids': []}})
    value['turns'][0]['result']['events'] = [guidance]
    projection = project_sample_evidence(value)
    localization = projection['turns'][0]['interventions'][0]['localization']
    assert localization['abstained'] is True and localization['fallback'] is False
    assert localization['shadow_changed'] is True
    assert projection['summary']['failure_stage'] == 'LOCALIZATION_ABSTAINED'
    assert projection['summary']['localization_abstentions'] == 1
    assert projection['summary']['shadow_fallbacks'] == 1


def test_projection_uses_persisted_contract_review_and_counts_blocked_repeat():
    value = sample(passed=False)
    events = value['turns'][0]['result']['events']
    guidance = events[0]
    guidance.update(
        generation={'mode': 'deterministic-contract-v1', 'model_calls': 0},
        guidance_contract={'contract_id': 'contract-1', 'recommended_actions': ['inspect'],
                           'policy': 'ADVISORY_EXCEPT_EXACT_NO_PROGRESS_REPEAT'},
    )
    value['turns'][0]['result']['events'] = [guidance, events[1], events[2], {
        'type': 'pg_decision_reviewed', 'sequence': 4, 'decision_key': 'decision-1',
        'review_id': 'review-1', 'contract_id': 'contract-1', 'status': 'BLOCKED',
        'blocked': True, 'reason': 'EXACT_NO_PROGRESS_REPEAT', 'actual_action': 'search_products',
        'recommended_actions': ['inspect'], 'matched_edge_ids': [],
    }]

    projection = project_sample_evidence(value)
    intervention = projection['turns'][0]['interventions'][0]
    assert projection['schema_version'] == 4
    assert intervention['guidance']['generation_mode'] == 'deterministic-contract-v1'
    assert intervention['action_alignment']['status'] == 'BLOCKED'
    assert intervention['action_alignment']['blocked'] is True
    assert projection['summary']['blocked_repeats'] == 1
    assert projection['summary']['deterministic_contracts'] == 1
    assert projection['summary']['guidance_model_calls'] == 0
    assert projection['summary']['failure_stage'] == 'NO_PROGRESS_REPEAT_BLOCKED'


def test_projection_links_typed_applicability_review_and_real_progress():
    value = sample()
    events = value['turns'][0]['result']['events']
    guidance = events[0]
    guidance.update(
        applicability={
            'strategy': 'typed-evidence-v1', 'status': 'APPLICABLE', 'reason': 'APPLICABLE_EDGE_FOUND',
            'evaluations': [{'edge_id': 'start-search', 'status': 'TRUE'}],
            'selected_edge_ids': ['start-search'], 'false_edge_ids': [], 'unknown_edge_ids': [],
        },
        projected_paths=[{'path_id': 'path-1', 'edge_ids': ['start-search'],
                          'currently_applicable_prefix': 1}],
    )
    events.insert(4, {
        'type': 'pg_decision_reviewed', 'sequence': 4, 'decision_key': 'decision-1',
        'review_id': 'review-1', 'contract_id': 'contract-1', 'status': 'ALIGNED',
        'blocked': False, 'reason': 'RECOMMENDED_ACTION', 'actual_action': 'search_products',
        'recommended_actions': ['search_products'], 'matched_edge_ids': ['start-search'],
    })
    events.append({
        'type': 'pg_action_outcome', 'sequence': 8, 'decision_key': 'decision-1',
        'review_id': 'review-1', 'outcome_id': 'outcome-1', 'progress_status': 'PROGRESSED',
        'progress_reason': 'VERIFIABLE_STATE_CHANGE', 'signals': ['TRUSTED_OBSERVATION_ADDED'],
        'cost_delta': {'model_calls': 0, 'tool_calls': 1}, 'matched_edge_ids': ['start-search'],
        'before_sha256': 'a' * 64, 'after_sha256': 'b' * 64,
    })

    projection = project_sample_evidence(value)
    intervention = projection['turns'][0]['interventions'][0]
    assert intervention['applicability']['selected_edge_ids'] == ['start-search']
    assert intervention['action_progress']['status'] == 'PROGRESSED'
    assert intervention['action_progress']['signals'] == ['TRUSTED_OBSERVATION_ADDED']
    assert projection['summary']['typed_applicability_interventions'] == 1
    assert projection['summary']['progressed_actions'] == 1
