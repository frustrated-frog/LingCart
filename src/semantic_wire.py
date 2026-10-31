"""模型条件只表达一个来源；宿主补齐协议字段，持久化验证仍严格。"""
from copy import deepcopy


def condition_schema(schema):
    """为模型提供互斥的概念/原始谓词表示，不修改领域 Schema。"""
    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        props = value.get('properties', {})
        if {'catalog_concept', 'field', 'op', 'value', 'source_quote', 'description'} <= props.keys():
            concept = deepcopy(props['catalog_concept'])
            if 'anyOf' in concept:
                concept = next(item for item in concept['anyOf'] if item.get('type') != 'null')
            common = {key: visit(props[key]) for key in ('description', 'source_quote')}
            semantic = dict(type='object', additionalProperties=False,
                            properties={**common, 'catalog_concept': concept},
                            required=['description', 'source_quote', 'catalog_concept'])
            raw = dict(type='object', additionalProperties=False,
                       properties={**common, **{key: visit(props[key]) for key in ('field', 'op', 'value')}},
                       required=['description', 'source_quote', 'field', 'op', 'value'])
            return {'anyOf': [semantic, raw]}
        return {key: visit(item) for key, item in value.items()}
    return visit(deepcopy(schema))


def decode_conditions(payload):
    """仅补齐新概念表示省略的字段；拒绝由旧表示携带的冲突，不覆盖任何值。"""
    from src.catalog_concepts import CatalogConceptRef, get_catalog_concept_registry
    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: visit(item) for key, item in value.items()}
        if (result.get('catalog_concept') is not None and 'source_quote' in result
                and not {'field', 'op', 'value'} & result.keys()):
            concept = CatalogConceptRef.model_validate(result['catalog_concept'])
            get_catalog_concept_registry().encode(concept)  # 校验版本、操作符和值。
            result.update(op=concept.operator, field=None, value=None)
        return result
    return visit(payload)
