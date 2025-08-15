from datetime import datetime, timedelta
from typing import Optional

from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

# --- NEW: Import the CRUD functions and the User model ---
from . import crud, models
from .config import JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

# --- Password Hashing (Unchanged) ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# --- JWT Token Handling (Unchanged) ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

# --- Pydantic User Schemas (previously in security.py, now more formal) ---
class UserBase(BaseModel):
    username: str
    full_name: Optional[str] = None

class User(UserBase):
    id: int
    disabled: bool

    class Config:
        orm_mode = True # This allows Pydantic to read data from ORM objects

# --- REFACTORED: User "Database" Logic ---
# The FAKE_USERS_DB dictionary is GONE.
# This function now takes a database session and uses the CRUD utility.
def get_user(db: Session, username: str) -> Optional[models.User]:
    """
    Retrieves a user from the database.
    """
    return crud.get_user_by_username(db, username=username)