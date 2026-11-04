"""案件级目标模式：独立语义解释，持久化控制执行，不授予提交权限。"""
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.errors import AgentError
from src.request_ir import SourceSpan, locate_source_span

Mode = Literal['CONTINUE', 'DRAFT', 'PAUSED', 'CLARIFY']


class CaseGoalScope(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    case_id: str = Field(min_length=1, max_length=64)
    mode: Mode


class CaseGoal(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    case_id: str
    order_id: int
    mode: Mode
    revision: int = Field(ge=1)
    source: SourceSpan
    run_id: str
    question: str | None = None


def enabled(context=None):
    from src.runtime import current_runtime
    context = context or current_runtime()
    return bool(context and getattr(context.mechanisms.get('planning'), 'goal_revision', False))


def current_goals(context):
    if context is None:
        return {}
    rows = context.snapshot.get('case_goals', (context.snapshot.get('conversation') or {}).get('case_goals', {}))
    try:
        if not isinstance(rows, dict): raise ValueError('Expected case map')
        for key, value in rows.items():
            if CaseGoal.model_validate(value).case_id != key: raise ValueError('Case identity mismatch')
    except (ValueError, TypeError) as error:
        raise AgentError('CASE_GOAL_INVALID', '保存的案件目标无法核验，已停止推进', retryable=False) from error
    return deepcopy(rows)


def held_goal(context, case_id):
    goal = current_goals(context).get(case_id)
    return goal if goal and goal['mode'] != 'CONTINUE' else None


def action_held(context, arguments):
    return any(goal['mode'] != 'CONTINUE' and (arguments.get('case_id') == goal['case_id']
               or arguments.get('order_id') == goal['order_id']) for goal in current_goals(context).values())


INSTRUCTION = '''For an existing case, saving a draft, pausing or resuming is AFTER_SALES_CONTINUE, not cancel/confirm.
Explicit material facts and goal changes can coexist in one continuation. Do not drop material_changes to ask about a goal.
If only the target outcome is ambiguous, return CLARIFY plus the independently clear AFTER_SALES_CONTINUE material updates.
Case-specific draft/pause/resume belongs to the case goal, not a global restriction; retain execution_boundaries for explicit
global prohibitions or explicit release of existing global restrictions. Resume never means permission to submit/refund.
'''

SEMANTIC_INSTRUCTION = '''For explicit changes to an EXISTING case's goal, add/replace an EXECUTION_BOUNDARY requirement
with case_goal:{case_id,mode:CONTINUE|DRAFT|PAUSED|CLARIFY}, restriction:null and an exact current HUMAN source_quote.
DRAFT means only save the case/materials; PAUSED means stop active advancement; CONTINUE means resume normal assessment,
not approval. CLARIFY means the desired outcome is unresolved. Preserve an existing case_goal if no new change is stated.
Case-specific mode is not a global restriction. Global prohibitions/retractions remain separate requirements.
Material facts in the same message must still be extracted independently with material_scope; a goal question cannot erase them.
'''


def goal_changes(context, understanding, rows, contract):
    if not enabled(context):
        return {}
    from src.after_sales.turn_verification import CaseTurnMismatch
    objectives = {o.case_id: o for o in understanding.objectives if o.kind == 'AFTER_SALES_CONTINUE' and o.case_id}
    changed = {r['requirement_id'] for r in contract['revisions'] if r['operation'] in {'ADD', 'REPLACE'}}
    changes = {}
    for row in rows:
        if row.requirement_id not in changed: continue
        if row.case_goal is not None:
            scope = row.case_goal
            if scope.case_id not in objectives or scope.case_id in changes:
                raise CaseTurnMismatch('目标变更没有唯一的当前案件归属', fields={}, question='请分别明确要改变哪个案件的处理目标。')
            changes[scope.case_id] = (scope.mode, row.source_quote, None)
        elif row.restriction == 'NO_SUBMISSION' and len(objectives) == 1:
            changes.setdefault(next(iter(objectives)), ('DRAFT', row.source_quote, None))
    question = next((c for c in understanding.commands if c.kind == 'CLARIFY'), None)
    if question:
        for case_id in objectives:
            changes[case_id] = ('CLARIFY', question.source_quote, question.message)
    return changes


def stage_goals(context, understanding, rows, contract, request):
    changes = goal_changes(context, understanding, rows, contract)
    if not changes:
        return
    previous = (context.snapshot.get('conversation') or {}).get('case_goals', {})
    goals = current_goals(context)
    for case_id, (mode, quote, question) in changes.items():
        old = previous.get(case_id)
        source = locate_source_span(quote, request)
        from src.after_sales.context import current_case_context
        case = next(c for c in current_case_context() if c.get('case_id') == case_id)
        new = CaseGoal(case_id=case_id, order_id=case['order_id'], mode=mode, revision=(old or {}).get('revision', 0) + 1,
                       source=source, run_id=context.run_id, question=question).model_dump(mode='json')
        goals[case_id] = new
        context.record('case_goal_revised', case_id=case_id, previous=old, current=new,
                       preserved=['已入账材料', '政策核验与工具账本', '提交前明确确认'],
                       suspended=[] if mode == 'CONTINUE' else ['自主业务查询', '材料追问', '提交确认'],
                       next_steps=['重新核验当前案件', '按当前状态选择下一步'] if mode == 'CONTINUE' else
                       ['保存本轮明确材料', '等待明确目标' if mode == 'CLARIFY' else '保留案件并停止主动推进'])
    context.checkpoint(case_goals=goals)


def goal_message(goal):
    return {'DRAFT': '案件及已核验材料已保存为草稿，未提交、未退款。',
            'PAUSED': '已暂停案件推进，保留已有材料；恢复后会重新核验，提交仍需单独确认。',
            'CLARIFY': goal.get('question') or '材料已保留，请明确接下来希望如何处理该案件。'}[goal['mode']]
