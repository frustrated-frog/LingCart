"""持久会话控制：命令落库、运行边界生效、异步执行和后续任务队列。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import uuid

from sqlalchemy import select

from src import tools
from src.backend.db import AgentRun, Checkpoint, RunCommand, RunControl
from src.errors import AgentError
from src.persistence import RunStore, transaction, utcnow
from src.runtime import BudgetLimits


class ControlSignal(BaseException):
    def __init__(self, kind):
        self.kind = kind


class ControlStore(RunStore):
    @staticmethod
    def command_payload(row):
        return dict(command_id=row.command_id, run_id=row.run_id, kind=row.kind, payload=row.payload,
                    status=row.status, outcome=row.outcome)

    def enqueue(self, session_id, run_id, kind, *, message='', command_id=None, expected_revision=None):
        if kind not in {'STEER', 'CANCEL', 'FOLLOW_UP'}:
            raise AgentError('INVALID_CONTROL', '不支持该运行控制指令', retryable=False)
        if kind != 'CANCEL' and not message.strip():
            raise AgentError('INVALID_CONTROL', '补充指令不能为空', retryable=False)
        if (not isinstance(command_id, str) or not command_id.strip() or len(command_id) > 64
                or type(expected_revision) is not int or expected_revision < 0):
            raise AgentError('INVALID_CONTROL', '必须提供有效的 command_id 和 expected_revision', retryable=False)
        with transaction() as session:
            run = session.get(AgentRun, run_id)
            if run is None or run.session_id != session_id:
                raise AgentError('RUN_NOT_FOUND', '没有找到当前会话的任务', retryable=False)
            existing = session.get(RunCommand, command_id)
            if existing:
                if (existing.session_id != session_id or existing.run_id != run_id or existing.kind != kind
                        or existing.payload.get('message', '') != message
                        or existing.payload.get('expected_revision') != expected_revision):
                    raise AgentError('COMMAND_ID_CONFLICT', '同一指令编号不能用于不同操作', retryable=False)
                return self.command_payload(existing)
            if run.result is not None:
                raise AgentError('RUN_ALREADY_FINISHED', '任务已结束，请发送新的需求', retryable=False)
            if kind != 'CANCEL' and session.scalar(select(RunCommand.command_id).where(
                    RunCommand.run_id == run_id, RunCommand.kind == 'CANCEL')):
                raise AgentError('RUN_CANCELLING', '任务正在停止，不能追加新指令', retryable=False)
            control = session.get(RunControl, run_id)
            if control is None:
                control = RunControl(run_id=run_id, revision=0)
                session.add(control)
            elif control.closed:
                raise AgentError('RUN_FINALIZING', '任务正在保存最终结果，请稍后发送新需求', retryable=True)
            if expected_revision != control.revision:
                raise AgentError('STALE_CONTROL_REVISION', '任务约束版本已变化，请刷新后重试', retryable=False)
            if kind == 'STEER':
                steers = session.scalars(select(RunCommand).where(
                    RunCommand.run_id == run_id, RunCommand.kind == 'STEER')).all()
                if len(steers) >= 4:
                    raise AgentError('STEERING_BUDGET_EXHAUSTED', '本次任务已达到调整次数上限，请结束后创建新任务', retryable=False)
            control.revision += 1
            row = RunCommand(command_id=command_id, run_id=run_id, session_id=session_id, kind=kind,
                             payload={'message': message, 'revision': control.revision, 'expected_revision': expected_revision,
                                      'execution_mode': (session.get(Checkpoint, run_id).payload or {}).get('execution_mode', 'auto')},
                             status='QUEUED', created_at=utcnow())
            session.add(row)
            if kind == 'CANCEL':
                queued = session.scalars(select(RunCommand).where(RunCommand.run_id == run_id,
                                                               RunCommand.kind == 'FOLLOW_UP', RunCommand.status == 'QUEUED')).all()
                for command in queued:
                    command.status = 'CANCELLED'
            session.flush()
            return self.command_payload(row)

    def commands(self, session_id, run_id):
        with tools._session() as session:
            run = session.get(AgentRun, run_id)
            if run is None or run.session_id != session_id:
                raise AgentError('RUN_NOT_FOUND', '没有找到当前会话的任务', retryable=False)
            return [self.command_payload(row) for row in session.scalars(select(RunCommand).where(
                RunCommand.run_id == run_id).order_by(RunCommand.created_at, RunCommand.command_id))]

    def pending_controls(self, run_id):
        with tools._session() as session:
            return [self.command_payload(row) for row in session.scalars(select(RunCommand).where(
                RunCommand.run_id == run_id, RunCommand.kind != 'FOLLOW_UP', RunCommand.status == 'QUEUED')
                .order_by(RunCommand.created_at, RunCommand.command_id))]

    def acknowledge(self, command_id, *, status='APPLIED', outcome=None):
        with transaction() as session:
            row = session.get(RunCommand, command_id)
            if row and row.status == 'QUEUED':
                row.status, row.outcome = status, outcome

    def revision(self, run_id):
        with tools._session() as session:
            row = session.get(RunControl, run_id)
            return row.revision if row else 0

    def close_controls(self, run_id):
        with transaction() as session:
            if session.scalar(select(RunCommand.command_id).where(RunCommand.run_id == run_id,
                    RunCommand.kind != 'FOLLOW_UP', RunCommand.status == 'QUEUED')):
                return False
            row = session.get(RunControl, run_id)
            if row is None:
                row = RunControl(run_id=run_id, revision=0)
                session.add(row)
            row.closed = True
            return True

    def reopen_controls(self, run_id):
        with transaction() as session:
            row = session.get(RunControl, run_id)
            if row:
                row.closed = False

    def next_follow_up(self, session_id):
        with tools._session() as session:
            rows = session.scalars(select(RunCommand).where(RunCommand.session_id == session_id,
                                   RunCommand.kind == 'FOLLOW_UP', RunCommand.status == 'QUEUED')
                                   .order_by(RunCommand.created_at, RunCommand.command_id)).all()
            for row in rows:
                previous = session.get(AgentRun, row.run_id)
                if previous and previous.result is not None:
                    return self.command_payload(row)
            return None


_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='lingcart-run')
_slots = threading.BoundedSemaphore(4)


class SessionController:
    def __init__(self, store=None):
        self.store = store or ControlStore()

    def start(self, session_id, message, *, request_id=None, action_id=None, limits=None, runner=None, resume_id=None, execution_mode='auto'):
        if execution_mode not in {'auto', 'adaptive'}:
            raise AgentError('INVALID_EXECUTION_MODE', '不支持该执行方式', retryable=False)
        if not _slots.acquire(blocking=False):
            raise AgentError('WORKER_CAPACITY_EXHAUSTED', '当前运行任务较多，请稍后重试', retryable=True)
        try:
            claim = self.store.claim(session_id, request_id or uuid.uuid4().hex, message,
                                     (limits or BudgetLimits()).__dict__, resume_id=resume_id, action_id=action_id)
            if 'cached' in claim:
                _slots.release()
                return {'run_id': claim['run_id'], 'session_id': session_id, 'status': claim['cached']['status'],
                        'result': claim['cached'], 'cached': True}
            if not resume_id:
                claim['snapshot']['execution_mode'] = execution_mode
                self.store.save_checkpoint(claim['run_id'], claim['token'], claim['snapshot'])
            else:
                self.store.reopen_controls(claim['run_id'])
            execution_mode = claim['snapshot'].get('execution_mode', 'auto')
            if runner is None and execution_mode == 'adaptive':
                from src.adaptive import run_adaptive
                runner = run_adaptive
            _pool.submit(self._work, claim, runner)
            return {'run_id': claim['run_id'], 'session_id': session_id, 'status': 'RUNNING', 'cached': False}
        except BaseException:
            _slots.release()
            raise

    def _work(self, claim, runner):
        from src.durable import execute_claim
        try:
            execute_claim(claim, store=self.store, runner=runner)
        except BaseException:
            # execute_claim 已保存 INTERRUPTED，错误不发送给外部提供方或其它会话。
            pass
        finally:
            _slots.release()
        self.drain(claim['session_id'], runner=runner)

    def drain(self, session_id, *, runner=None):
        last_handle = None
        while True:
            command = self.store.next_follow_up(session_id)
            if command is None:
                return last_handle
            try:
                handle = self.start(session_id, command['payload']['message'], request_id=command['command_id'], runner=runner,
                                    execution_mode=command['payload'].get('execution_mode', 'auto'))
            except AgentError as error:
                # 其它运行占用会话时保留排队；只有同一command已被claim才可确认。
                if error.code == 'RUN_IN_PROGRESS':
                    with tools._session() as session:
                        claimed = session.scalar(select(AgentRun).where(AgentRun.session_id == session_id,
                                                          AgentRun.request_id == command['command_id']))
                        if claimed:
                            self.store.acknowledge(command['command_id'], outcome={'run_id': claimed.run_id})
                return last_handle
            self.store.acknowledge(command['command_id'], outcome={'run_id': handle['run_id']})
            if not handle.get('cached'):
                return handle
            last_handle = handle


def apply_controls(context):
    """检查点先保存已应用ID，再确认命令；崩溃后可幂等补齐确认。"""
    if context.store is None:
        return
    store = ControlStore()
    commands = store.pending_controls(context.run_id)
    processed = list(context.snapshot.get('applied_commands', []))
    signal = None
    for command in commands:
        identity = command['command_id']
        if identity in processed:
            store.acknowledge(identity)
            continue
        processed.append(identity)
        context.snapshot['applied_commands'] = processed
        if command['kind'] == 'STEER':
            context.snapshot.setdefault('steering', []).append(command['payload']['message'])
            context.snapshot['goal_revision'] = command['payload']['revision']
            context.models.clear()
            context.model_cursor = 0
            context.approval = None
            context.snapshot.pop('plan_definition', None)
            context.snapshot.pop('plan', None)
            signal = 'STEER'
        elif command['kind'] == 'CANCEL':
            context.snapshot['cancel_requested'] = True
            signal = 'CANCEL'
        context.record('control_applied', kind=command['kind'], command_id=identity,
                       goal_revision=context.snapshot.get('goal_revision', 0))
        store.acknowledge(identity)
    if context.snapshot.get('cancel_requested'):
        raise ControlSignal('CANCEL')
    if signal:
        raise ControlSignal(signal)
