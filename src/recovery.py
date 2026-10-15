"""控制与恢复前先还原已提交账本证据，闭合中断的执行及模型协议。"""
from src.runtime import ToolExecutor


def recover_evidence(context):
    if context.store is None or not context.events:
        return
    ended = {e.get('execution_id') for e in context.events if e['type'] in {'tool_result', 'error'}}
    outcomes = context.store.outcomes(context.run_id)
    for call in list(context.events):
        identity = call.get('execution_id')
        if call['type'] != 'tool_call' or not identity or identity in ended:
            continue
        fingerprint = call.get('action_fingerprint')
        saved = next((r for r in outcomes if r['tool'] == call['tool'] and
                      ToolExecutor.fingerprint(r['tool'], r['args']) == fingerprint and r['status'] == 'SUCCEEDED'), None)
        common = dict(tool=call['tool'], args=call.get('args', {}), action_fingerprint=fingerprint, execution_id=identity)
        if saved:
            context.record('tool_result', **common, status='REUSED', result=saved['result'], reused=True,
                           duration_ms=0, source_ledger=saved['call_key'], recovered=True)
        else:
            context.record('error', **common, code='INTERRUPTED_TOOL_CALL', retryable=False,
                           message='调用在结果保存前中断，未声称业务成功；写操作仍受账本重放策略限制', recovered=True)
        ended.add(identity)
    requests, consumed = {}, set()
    evidence = [e for e in context.events if e['type'] in {'tool_result', 'error', 'approval_required'}]

    def proof(request, status=None):
        kind = {'FAILED': 'error', 'WAITING_APPROVAL': 'approval_required'}.get(status, 'tool_result')
        return next((e for e in evidence if e['sequence'] not in consumed and e['sequence'] > request['sequence']
                     and e.get('tool') == request.get('tool') and e.get('action_fingerprint') == request.get('action_fingerprint')
                     and (status is None or e['type'] == kind) and
                     (status not in {'SUCCEEDED', 'REUSED', 'BUSINESS_ERROR', 'BUSINESS_REJECTED'} or e.get('status') == status)), None)
    for event in list(context.events):
        if event['type'] == 'tool_requested':
            requests[event.get('protocol_id')] = event
        elif event['type'] == 'tool_protocol':
            request = requests.pop(event.get('protocol_id'), None)
            if request:
                found = proof(request, event.get('status'))
                if found:
                    consumed.add(found['sequence'])
    for protocol_id, request in requests.items():
        found = proof(request)
        if found:
            consumed.add(found['sequence'])
            status = {'error': 'FAILED', 'approval_required': 'WAITING_APPROVAL'}.get(found['type'], found.get('status'))
        else:
            status = 'CANCELLED_BY_RUNTIME'
        context.record('tool_protocol', protocol_id=protocol_id, tool=request.get('tool'),
                       tool_call_id=request.get('tool_call_id'), action_fingerprint=request.get('action_fingerprint'),
                       status=status, reason='RECOVERED_PROTOCOL', recovered=True)
