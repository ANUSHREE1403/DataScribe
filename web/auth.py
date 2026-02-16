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

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", settings.database_url)

# Render/Railway often give URLs starting with postgres://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(DATABASE_URL)

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


# Create tables
Base.metadata.create_all(bind=engine)


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_db():
    """FastAPI dependency to get a DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


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

