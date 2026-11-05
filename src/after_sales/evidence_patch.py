"""候选模型的单一材料变更协议，编译到已有顺序工具执行边界。"""
from copy import deepcopy
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from src.after_sales.turn_verification import CaseTurnMismatch


class PatchBase(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    source_quote: str = Field(min_length=1, max_length=500)


class Provide(PatchBase):
    operation: Literal['PROVIDE']
    requirement_id: str = Field(min_length=1,max_length=160)


class Defer(PatchBase):
    operation: Literal['DEFER']
    requirement_id: str = Field(min_length=1,max_length=160)


class Retract(PatchBase):
    operation: Literal['RETRACT']
    evidence_id: str = Field(min_length=1,max_length=64)


class Replace(Retract):
    operation: Literal['REPLACE']
    requirement_id: str = Field(min_length=1,max_length=160)


EvidencePatch = Annotated[Provide | Defer | Retract | Replace, Field(discriminator='operation')]
PATCHES = TypeAdapter(Annotated[list[EvidencePatch], Field(min_length=1,max_length=8)])
LEGACY_FIELDS = {'evidence_updates','evidence_text','evidence_readiness_claims','requirement_id',
                 'revoke_evidence_id','clarification_id','case_revision'}
INSTRUCTION = '''Existing cases: material_changes is the ONLY material mutation interface, even for one item.
List operations: PROVIDE{requirement_id,source_quote}, DEFER{requirement_id,source_quote},
RETRACT{evidence_id,source_quote}, REPLACE{evidence_id,requirement_id,source_quote}.
Each entry includes operation. Use exact listed IDs, verbatim current HUMAN quotes. Parent source_quote covers all entries.
REPLACE quote must state both explicit withdrawal/correction and replacement facts. No implicit withdrawal.
One operation per requirement. Host owns case revision, answer binding and execution order.
Empty/omitted material_changes means reassess only. Never emit legacy material fields. No operation grants refund consent.
'''


def enabled():
    from src.runtime import current_runtime
    runtime = current_runtime()
    return bool(runtime and getattr(runtime.mechanisms.get('tool_protocol'), 'material_patch_protocol', False))


def adapt(payload, request):
    """模型协议与内部兼容协议只在此转换；不能自动猜测撤回或材料归属。"""
    if not enabled() or not isinstance(payload, dict):
        return payload
    from src.after_sales.context import current_case_context, normalize_case_continuation
    cases = {c['case_id']: c for c in current_case_context() if c.get('case_id')}
    result = deepcopy(payload)
    for objective in result.get('objectives', []):
        if not isinstance(objective, dict) or objective.get('kind') != 'AFTER_SALES_CONTINUE':
            continue
        fields = sorted(k for k in LEGACY_FIELDS if objective.get(k) not in (None, [], ''))
        if fields:
            raise CaseTurnMismatch('材料变更协议不能混入旧字段', fields={'legacy_fields':fields},
                                   question='请分别说明本轮要提供、延期、撤回或更正的材料。')
        raw = objective.pop('material_changes', [])
        if not raw:
            continue
        try:
            patches = PATCHES.validate_python(raw)
        except ValidationError as error:
            raise CaseTurnMismatch('材料变更字段不符合操作类型', fields={'errors':[
                {'path':list(e['loc']),'message':e['msg']} for e in error.errors(include_input=False)[:8]]},
                question='请分别说明每项材料及要执行的变更。') from error
        normalized, _ = normalize_case_continuation(objective, (request,))
        case = cases.get(normalized.get('case_id'))
        if not case:
            raise CaseTurnMismatch('材料变更缺少唯一的当前案件', fields={}, question='请明确这份材料属于哪个案件或订单。')
        requirements = {r['requirement_id']: r for r in case['requirements']}
        evidence = {e['evidence_id']: e for e in case['evidence'] if e['status']=='ACTIVE'}
        updates = []
        for patch in patches:
            item = patch.model_dump()
            target = item.get('requirement_id')
            old = evidence.get(item.get('evidence_id'))
            if patch.operation in {'RETRACT','REPLACE'}:
                matching = [r['requirement_id'] for r in case['requirements'] if old and r['evidence_type']==old['evidence_type']]
                if len(matching)!=1 or (target and target!=matching[0]):
                    raise CaseTurnMismatch('撤回或替换材料不属于当前要求',fields={'evidence_id':item.get('evidence_id')},
                                           question='请明确要撤回或更正哪条已有材料。')
                target = matching[0]
            issue = ('UNKNOWN_MATERIAL_REQUIREMENT' if target not in requirements else
                     'QUOTE_NOT_CURRENT' if patch.source_quote not in request else
                     'QUOTE_OUTSIDE_OBJECTIVE' if patch.source_quote not in objective.get('source_quote','') else None)
            if issue:
                raise CaseTurnMismatch('材料变更缺少当前原文或合法要求',fields={'requirement_id':target, 'code':issue},
                                       question='请明确材料名称与本轮实际情况。')
            update = dict(requirement_id=target, disposition={'PROVIDE':'PROVIDED','DEFER':'DEFERRED',
                          'REPLACE':'PROVIDED','RETRACT':'RETRACTED'}[patch.operation], source_quote=patch.source_quote)
            if old:
                update['revoke_evidence_id']=old['evidence_id']
            updates.append(update)
        if len({u['requirement_id'] for u in updates})!=len(updates):
            raise CaseTurnMismatch('同一材料要求不能重复变更',fields={},question='同一项材料需要更正时，请明确旧陈述和更正后的情况。')
        objective['evidence_updates']=updates
        from src.runtime import current_runtime
        current_runtime().record('material_patch_decoded', status='PROPOSED', case_id=case['case_id'],
                                 operations=[p.model_dump() for p in patches])
    return result
