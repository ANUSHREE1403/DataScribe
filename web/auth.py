from datetime import datetime
import os

from passlib.context import CryptContext
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

from utils.config import settings

# ---------------------------------------------------------------------------
# Database configuration
# Try the configured URL first; fall back to local SQLite if it fails.
# ---------------------------------------------------------------------------
_FALLBACK_SQLITE = "sqlite:///./datascribe.db"
DATABASE_URL = os.getenv("DATABASE_URL", settings.database_url)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def _make_engine(url: str):
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True, pool_size=2, max_overflow=0)


try:
    engine = _make_engine(DATABASE_URL)
    # Quick connectivity test (no tables yet, just a connection)
    with engine.connect() as _conn:
        pass
    print(f"Database connected: {DATABASE_URL[:40]}...")
except Exception as db_err:
    print(f"WARNING: Could not connect to database ({db_err}). Falling back to local SQLite.")
    DATABASE_URL = _FALLBACK_SQLITE
    engine = _make_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    analyses = relationship("AnalysisJob", back_populates="user")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(64), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    dataset_name = Column(String(255), nullable=False)
    target_column = Column(String(255), nullable=True)
    model_choice = Column(String(50), nullable=True)
    accuracy = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="analyses")


# Create tables (safe: only creates if they don't already exist)
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"WARNING: Could not create tables: {e}")


# Password hashing
# Use pbkdf2_sha256 only (stable on current dependency set, no bcrypt backend issues).
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
)


def get_db():
    """FastAPI dependency to get a DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_password_hash(password: str) -> str:
    """Hash the password using the configured scheme."""
    return pwd_context.hash(password or "")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Preferred path for current hashes.
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        # Backward-compatibility: older deployments may still have bcrypt hashes.
        # Newer bcrypt wheels enforce 72-byte limit; truncate for legacy verify only.
        if isinstance(hashed_password, str) and hashed_password.startswith("$2"):
            try:
                import bcrypt

                candidate = (plain_password or "").encode("utf-8")[:72]
                return bcrypt.checkpw(candidate, hashed_password.encode("utf-8"))
            except Exception:
                return False
        return False


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, email: str, password: str, full_name: str | None = None):
    existing = get_user_by_email(db, email)
    if existing:
        raise ValueError("Email is already registered")

    user = User(
        email=email,
        full_name=full_name,
        hashed_password=get_password_hash(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_analysis_job(
    db: Session,
    *,
    user_id: int,
    job_id: str,
    dataset_name: str,
    target_column: str | None,
    model_choice: str | None,
    accuracy: float | None,
):
    job = AnalysisJob(
        user_id=user_id,
        job_id=job_id,
        dataset_name=dataset_name,
        target_column=target_column,
        model_choice=model_choice,
        accuracy=accuracy,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_current_user(request, db: Session):
    """Return the currently logged-in user based on session, or None."""
    user_id = request.session.get("user_id") if hasattr(request, "session") else None
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()
