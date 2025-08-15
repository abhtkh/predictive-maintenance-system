from sqlalchemy import Column, Integer, String, Boolean, Enum as SQLAlchemyEnum
from .database import Base
import enum

# --- NEW: Define an Enum for our user roles for consistency ---
class UserRole(str, enum.Enum):
    operator = "operator"
    engineer = "engineer"
    manager = "manager"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    disabled = Column(Boolean, default=False)

    # --- NEW FIELDS FOR ROLE-BASED ALERTING ---
    email = Column(String, unique=True, index=True, nullable=True) # Emails must be unique
    role = Column(SQLAlchemyEnum(UserRole), default=UserRole.operator, nullable=False)
    receives_email_alerts = Column(Boolean, default=False, nullable=False)