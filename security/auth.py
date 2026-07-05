"""
JWT Authentication for Document Portal.

How it works:
    1. User registers with email + password → password is hashed with bcrypt
       and stored in the users table in PostgreSQL.
    2. User logs in → we verify the hash → if valid, issue a JWT access token
       (signed with SECRET_KEY, expires in 30 minutes by default).
    3. Protected endpoints receive the token in the Authorization header:
           Authorization: Bearer <token>
    4. The get_current_user() dependency decodes the JWT, validates it,
       and returns the user. If the token is missing/expired/invalid → 401.

Why JWT (not sessions)?
    JWTs are stateless — the server doesn't need to store tokens in a database.
    The token itself carries the user's identity (sub = user email).
    This makes horizontal scaling easy: any container can validate any token
    without sharing session state.

Environment variables:
    JWT_SECRET_KEY     — MUST be set in production (long random string)
    JWT_ALGORITHM      — default HS256
    JWT_EXPIRE_MINUTES — default 30

Tables:
    users — id, email (unique), hashed_password, is_active, created_at
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import Column, Boolean, DateTime, Integer, String
from sqlalchemy.exc import IntegrityError

from db.models import Base, SessionLocal, engine
from logger import GLOBAL_LOGGER as log


# ── Config ────────────────────────────────────────────────────────────────────

SECRET_KEY     = os.getenv("JWT_SECRET_KEY", "change-me-in-production-use-a-long-random-string")
ALGORITHM      = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))

pwd_context    = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme  = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── ORM Model ─────────────────────────────────────────────────────────────────

class User(Base):
    """
    Stores registered users. Each user can own multiple chat sessions.
    passwords are NEVER stored in plaintext — only bcrypt hashes.
    """
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    email           = Column(String(256), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime(timezone=True),
                             default=lambda: datetime.now(timezone.utc))


def create_user_table() -> None:
    """Create the users table if it doesn't exist (called at startup)."""
    Base.metadata.create_all(bind=engine, tables=[User.__table__])


# ── Pydantic schemas (request/response shapes) ────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = EXPIRE_MINUTES * 60   # seconds

class UserOut(BaseModel):
    email: str
    is_active: bool


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """bcrypt-hash a plaintext password."""
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the bcrypt hash."""
    return pwd_context.verify(plain, hashed)


# ── JWT helpers ───────────────────────────────────────────────────────────────

def create_access_token(email: str) -> str:
    """
    Create a signed JWT containing the user's email as the subject (sub).
    Expires in EXPIRE_MINUTES minutes.
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=EXPIRE_MINUTES)
    payload = {"sub": email, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> str:
    """
    Decode and validate a JWT. Returns the email (sub) on success.
    Raises HTTPException 401 if invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return email
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid or has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependency ────────────────────────────────────────────────────────

def get_current_user(token: str = Depends(oauth2_scheme)) -> UserOut:
    """
    FastAPI dependency — add to any endpoint to require authentication.

    Usage in endpoints:
        @app.get("/protected")
        async def protected(user: UserOut = Depends(get_current_user)):
            return {"email": user.email}

    The client must send:
        Authorization: Bearer <access_token>
    """
    email = decode_token(token)
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == email).first()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found or inactive")
        return UserOut(email=user.email, is_active=user.is_active)


# ── Service functions ─────────────────────────────────────────────────────────

def register_user(email: str, password: str) -> UserOut:
    """
    Create a new user. Raises 409 if email already exists.
    Password is bcrypt-hashed before storage — plaintext is never saved.
    """
    with SessionLocal() as db:
        try:
            user = User(email=email, hashed_password=hash_password(password))
            db.add(user)
            db.commit()
            log.info("New user registered", email=email)
            return UserOut(email=email, is_active=True)
        except IntegrityError:
            db.rollback()
            raise HTTPException(status_code=409, detail="Email already registered")

def login_user(email: str, password: str) -> TokenResponse:
    """
    Verify credentials and return a JWT access token.
    Raises 401 if email not found or password wrong.
    Deliberately uses the same error message for both cases
    to avoid leaking which emails are registered (security best practice).
    """
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == email).first()
        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Account is disabled")
        token = create_access_token(email)
        log.info("User logged in", email=email)
        return TokenResponse(access_token=token)
