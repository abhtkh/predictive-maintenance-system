from sqlalchemy.orm import Session
from pydantic import BaseModel

# Import our database models and security functions
from . import models, security

# --- Pydantic Schemas for data validation ---

class UserCreate(BaseModel):
    username: str
    password: str
    email: str
    role: models.UserRole
    receives_email_alerts: bool = False

# --- CRUD Functions for the User Model ---

def get_user_by_username(db: Session, username: str):
    """
    Reads a single user from the database by their username.
    """
    return db.query(models.User).filter(models.User.username == username).first()

def get_alert_recipients(db: Session) -> list[models.User]:
    return db.query(models.User).filter(models.User.receives_email_alerts == True, models.User.disabled == False).all()

def create_user(db: Session, user: UserCreate):
    """
    Creates a new user in the database.
    This function hashes the password before storing it.
    """
    hashed_password = security.get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        email=user.email,
        role=user.role,
        receives_email_alerts=user.receives_email_alerts,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user: models.User, details: dict):
    """
    Updates a user's details from a dictionary of provided fields.
    """
    for key, value in details.items():
        if value is not None:
            setattr(user, key, value)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def update_user_password(db: Session, user: models.User, new_password: str):
    user.hashed_password = security.get_password_hash(new_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user