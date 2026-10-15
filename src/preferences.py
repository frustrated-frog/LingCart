"""明确确认的跨会话偏好；画像编号是本地演示的随机访问凭据。"""
from datetime import UTC, datetime
import uuid

from sqlalchemy import select
from src import tools
from src.backend.db import Base, Conversation
from sqlalchemy import ForeignKey, Integer, JSON, String, DateTime, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from src.errors import AgentError
from src.persistence import transaction

ALLOWED_FEATURES = {'主动降噪', '通话降噪', '降噪', '静音', '三模', '双模', '有线', '无线', '蓝牙',
                    '头戴式', '开放式', '入耳式', '4K', '60Hz', '便携', '通勤', '办公', '防水',
                    '红轴', '茶轴', '双屏', 'HDMI', '运动', '视频会议', '麦克风', '空间音频'}


class PreferenceProfile(Base):
    __tablename__ = 'preference_profiles'
    profile_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class ProfileBinding(Base):
    __tablename__ = 'profile_bindings'
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile_id: Mapped[str] = mapped_column(ForeignKey('preference_profiles.profile_id'), index=True)


class Preference(Base):
    __tablename__ = 'preferences'
    __table_args__ = (UniqueConstraint('profile_id', 'key'),)
    preference_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile_id: Mapped[str] = mapped_column(ForeignKey('preference_profiles.profile_id'), index=True)
    key: Mapped[str] = mapped_column(String(64))
    value: Mapped[dict] = mapped_column(JSON)
    version: Mapped[int] = mapped_column(Integer, default=1)
    source: Mapped[str] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(DateTime)


def ensure_tables():
    Base.metadata.create_all(tools._engine, tables=[PreferenceProfile.__table__, ProfileBinding.__table__, Preference.__table__])


class PreferenceStore:
    def create(self):
        ensure_tables()
        identity = uuid.uuid4().hex
        with transaction() as session:
            session.add(PreferenceProfile(profile_id=identity, created_at=datetime.now(UTC).replace(tzinfo=None)))
        return identity

    def _profile(self, session, profile_id):
        if not session.get(PreferenceProfile, profile_id):
            raise AgentError('PROFILE_NOT_FOUND', '没有找到该偏好档案', retryable=False)

    def bind(self, session_id, profile_id):
        ensure_tables()
        with transaction() as session:
            self._profile(session, profile_id)
            row = session.get(ProfileBinding, session_id)
            if row and row.profile_id != profile_id:
                raise AgentError('PROFILE_BINDING_CONFLICT', '当前对话已绑定其它偏好档案，请创建新对话', retryable=False)
            if not row:
                session.add(ProfileBinding(session_id=session_id, profile_id=profile_id))

    def list(self, profile_id, query=''):
        ensure_tables()
        with tools._session() as session:
            self._profile(session, profile_id)
            rows = session.scalars(select(Preference).where(Preference.profile_id == profile_id).order_by(Preference.key)).all()
            return [dict(preference_id=r.preference_id, key=r.key, value=r.value, version=r.version,
                         source=r.source, updated_at=r.updated_at.replace(tzinfo=UTC).isoformat()) for r in rows
                    if not query or query in r.key or query in str(r.value)]

    def remember(self, profile_id, key, value, *, confirmed=False):
        ensure_tables()
        if confirmed is not True:
            raise AgentError('PREFERENCE_CONFIRMATION_REQUIRED', '保存偏好需要明确确认', retryable=False)
        valid = ((key == 'max_price' and type(value) in (int, float) and 0 < value <= 1000000) or
                 (key == 'category' and value in ('耳机', '键盘', '扩展坞')) or
                 (key == 'features' and isinstance(value, list) and len(value) <= 6 and
                  all(isinstance(v, str) and v in ALLOWED_FEATURES for v in value)))
        if not valid:
            raise AgentError('INVALID_PREFERENCE', '只支持预算、品类和选购特征偏好，值无效', retryable=False)
        with transaction() as session:
            self._profile(session, profile_id)
            row = session.scalar(select(Preference).where(Preference.profile_id == profile_id, Preference.key == key))
            if row is None:
                row = Preference(preference_id=uuid.uuid4().hex, profile_id=profile_id, key=key, version=0)
                session.add(row)
            row.value, row.source, row.version = {'value': value}, 'USER_CONFIRMED', row.version + 1
            row.updated_at = datetime.now(UTC).replace(tzinfo=None)
        return self.list(profile_id)

    def forget(self, profile_id, key):
        ensure_tables()
        with transaction() as session:
            self._profile(session, profile_id)
            row = session.scalar(select(Preference).where(Preference.profile_id == profile_id, Preference.key == key))
            if row:
                session.delete(row)

    def for_session(self, session_id):
        ensure_tables()
        with tools._session() as session:
            binding = session.get(ProfileBinding, session_id)
            profile_id = binding.profile_id if binding else None
        return self.list(profile_id) if profile_id else []
