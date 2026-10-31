from copy import deepcopy
import pytest
from src.adaptive_decision import Constraint, IntentDeclaration
from src.catalog_concepts import get_catalog_concept_registry
from src.semantic_wire import condition_schema, decode_conditions


def predicate():
    return dict(description='至少8个接口', source_quote='至少8个接口', catalog_concept=dict(
        schema_revision=get_catalog_concept_registry().schema_revision, concept_id='hub.port_count', operator='gte', canonical_value=8))


def test_semantic_wire_uses_host_operator_and_keeps_unbound_until_observation():
    data={'hard_constraints': [predicate()]}; original=deepcopy(data)
    result=decode_conditions(data)
    c=Constraint.model_validate(result['hard_constraints'][0])
    assert c.op=='gte' and c.field is None and c.value is None
    assert data==original


def test_wire_schema_has_no_duplicate_encoding_and_preserves_domain_schema():
    original=IntentDeclaration.model_json_schema(); view=condition_schema(original)
    branches=view['$defs']['Constraint']['anyOf']
    assert set(branches[0]['properties'])=={'description','source_quote','catalog_concept'}
    assert set(branches[1]['properties'])=={'description','source_quote','field','op','value'}
    assert 'properties' in original['$defs']['Constraint']


@pytest.mark.parametrize('change', [dict(field='price',op='gte',value=8),dict(field=None,op='gte',value=8)])
def test_conflicting_legacy_encoding_is_not_silently_repaired(change):
    with pytest.raises(ValueError):
        Constraint.model_validate(decode_conditions({**predicate(),**change}))


def test_unknown_concept_rejected_and_pd_unit_compiled():
    p=predicate();p['catalog_concept']['concept_id']='unknown'
    with pytest.raises(ValueError):decode_conditions(p)
    p=predicate();p['catalog_concept'].update(concept_id='hub.pd_power_watts',canonical_value=100)
    c=Constraint.model_validate(decode_conditions(p))
    encoded=get_catalog_concept_registry().encode(c.catalog_concept)
    assert encoded.field=='specs.PD 供电' and encoded.value==100
