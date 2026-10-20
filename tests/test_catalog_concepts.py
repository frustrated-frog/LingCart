"""版本化目录概念、无损 codec 与内核映射事务。"""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.adaptive_decision import Constraint
from src.catalog_concepts import (CATALOG_CONCEPT_SCHEMA_REVISION, CatalogConceptError,
                                  CatalogConceptRef, get_catalog_concept_registry)
from src.constraint_bindings import effective_goal, resolve_catalog_concepts
from src.native_protocol import decision_action_surface, decision_functions
from src.request_audit import binding
from src.task_plan import TaskPlan, TaskStep


def concept(concept_id, operator, canonical_value):
    return dict(schema_revision=CATALOG_CONCEPT_SCHEMA_REVISION, concept_id=concept_id,
                operator=operator, canonical_value=canonical_value)


def unbound(description, quote, concept_id, operator, value):
    return dict(description=description, source_quote=quote, field=None, op=operator, value=None,
                catalog_concept=concept(concept_id, operator, value))


def goal():
    return dict(summary='选购键盘', objectives=[dict(id='shopping', description='推荐键盘',
        source_quote='键盘', kind='recommend', order_id=None, question=None, product_ids=[],
        hard_constraints=[])], hard_constraints=[
            dict(description='预算350', source_quote='350元以内', field='price', op='lte', value=350),
            unbound('三模连接', '三模连接', 'keyboard.connection_mode', 'eq', 'tri_mode'),
            unbound('静音轴', '静音轴', 'keyboard.switch_type', 'eq', 'silent'),
        ], soft_preferences=[], uncertainties=[])


def state():
    return dict(request='推荐一款350元以内的键盘，要三模连接，而且必须是静音轴',
                goal_revision=0, goal=goal(), request_audit={'verdict': 'APPROVED'},
                constraint_bindings={}, binding_revision=0, binding_evidence_access={'step-1': ['P1202']})


def plan(specs=None):
    product = dict(product_id='P1202', name='静键机械键盘', category='键盘', price=329,
                   stock=12, summary='三模连接 · 84 键 · 低噪办公',
                   specs=specs or {'连接方式': '三模', '轴体': '静音轴', '键数': 84})
    return TaskPlan(['shopping'], [TaskStep('step-1', 'shopping', 'search_products', {},
        status='SUCCEEDED', result={'products': [product], 'observation': {'digest': 'catalog-row-v1'}})])


def test_registry_identity_and_encoding_are_content_addressed():
    registry = get_catalog_concept_registry()
    reference = CatalogConceptRef.model_validate(concept(
        'keyboard.connection_mode', 'eq', 'tri_mode'))

    resolved = registry.resolve(reference, ('三模', '双模'))

    assert resolved.model_dump() == {
        'field': 'specs.连接方式', 'op': 'eq', 'value': '三模',
        'codec_id': 'keyboard-connection-mode/v1',
        'schema_revision': CATALOG_CONCEPT_SCHEMA_REVISION,
    }
    assert registry.manifest()['fingerprint'] == registry.fingerprint
    assert len(registry.fingerprint) == 64


def test_stale_revision_unknown_value_and_unit_drift_fail_closed():
    registry = get_catalog_concept_registry()
    with pytest.raises(ValidationError, match='schema revision'):
        CatalogConceptRef.model_validate({**concept('keyboard.connection_mode', 'eq', 'tri_mode'),
                                          'schema_revision': 'demo-catalog/product-concepts/v0'})
    with pytest.raises(ValidationError, match='canonical value'):
        CatalogConceptRef.model_validate(concept('keyboard.connection_mode', 'eq', 'bluetooth_only'))

    battery = CatalogConceptRef.model_validate(concept('audio.battery_hours', 'gte', 30))
    with pytest.raises(CatalogConceptError, match='无法无损解释'):
        registry.resolve(battery, ('30 天',))


def test_constraint_keeps_semantic_reference_and_physical_encoding_consistent():
    raw = unbound('三模连接', '三模连接', 'keyboard.connection_mode', 'eq', 'tri_mode')
    assert Constraint.model_validate(raw).field is None

    with pytest.raises(ValidationError, match='operator must equal'):
        Constraint.model_validate({**raw, 'op': 'ne'})
    with pytest.raises(ValidationError, match='trusted concept codec'):
        Constraint.model_validate({**raw, 'field': 'specs.轴体', 'value': '三模'})

    encoded = Constraint.model_validate({**raw, 'field': 'specs.连接方式', 'value': '三模'})
    assert encoded.catalog_concept.canonical_value == 'tri_mode'


def test_registry_proposes_all_concepts_atomically_from_kernel_evidence():
    current = state()
    resolution = resolve_catalog_concepts(current, plan())

    assert resolution['status'] == 'PROPOSED'
    assert {item.constraint_ref for item in resolution['proposals']} == {'global:1', 'global:2'}
    transaction = resolution['transaction']
    assert transaction['source'] == 'catalog_registry' and transaction['status'] == 'PROPOSED'
    assert transaction['registry']['fingerprint'] == get_catalog_concept_registry().fingerprint
    assert len(transaction['transaction_id']) == len(transaction['evidence_sha256']) == 64

    effective = effective_goal({**current, 'constraint_bindings': resolution['bindings']})
    assert effective['hard_constraints'][1]['field'] == 'specs.连接方式'
    assert effective['hard_constraints'][1]['value'] == '三模'
    assert effective['hard_constraints'][2]['field'] == 'specs.轴体'
    assert effective['hard_constraints'][2]['value'] == '静音轴'


def test_registry_never_partially_applies_when_one_concept_lacks_compatible_evidence():
    current = state()
    incomplete = plan({'连接方式': '三模', '键数': 84})

    resolution = resolve_catalog_concepts(current, incomplete)

    assert resolution['status'] == 'UNRESOLVED'
    assert resolution['proposals'] == []
    assert any(item['constraint_ref'] == 'global:2' for item in resolution['unresolved'])
    assert current['constraint_bindings'] == {}


def test_transaction_identity_changes_with_observation_evidence():
    first_state = state()
    first = resolve_catalog_concepts(first_state, plan())['transaction']
    second_plan = deepcopy(plan())
    second_plan.steps[0].result['observation']['digest'] = 'catalog-row-v2'
    second = resolve_catalog_concepts(state(), second_plan)['transaction']

    assert first['after_bindings_sha256'] == second['after_bindings_sha256']
    assert first['evidence_sha256'] != second['evidence_sha256']
    assert first['transaction_id'] != second['transaction_id']


def test_action_surface_reserves_audit_and_exposes_only_terminal_choices():
    current = state()
    current['evidence_access'] = {'step-1': ['P1202']}
    resolution = resolve_catalog_concepts(current, plan())
    transaction = deepcopy(resolution['transaction'])
    transaction.update(status='APPLIED', audit_reference='run:12', applied_binding_revision=1)
    audit = {
        'verdict': 'APPROVED',
        'binding': binding(current['request'], current['goal'], current['goal_revision']),
        'bindings_sha256': transaction['after_bindings_sha256'],
        'binding_revision': 1,
        'reference': 'run:12',
        'binding_transaction_id': transaction['transaction_id'],
    }
    current.update(constraint_bindings=resolution['bindings'], binding_revision=1,
                   binding_tx=transaction, binding_audit=audit)

    surface = decision_action_surface(current, plan(), remaining_model_calls=2)
    declarations = decision_functions(current, plan(), surface=surface)

    assert surface['mode'] == 'recommendation_commit'
    assert {item['function']['name'] for item in declarations} == {'recommend', 'clarify', 'finish'}
    recommendation = next(item['function'] for item in declarations if item['function']['name'] == 'recommend')
    assert recommendation['parameters']['properties']['product_id']['enum'] == ['P1202']
    assert decision_action_surface(current, plan(), remaining_model_calls=3)['mode'] == 'full'

