from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, PatientProfile
from app.schemas import UserCreate, UserLogin, Token
from app.services.auth import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new patient or doctor account in the platform database."""
    # Check if email is already taken
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account with this email already exists."
        )
    
    # Hash password & construct User
    hashed_pwd = hash_password(user_data.password)
    new_user = User(
        email=user_data.email,
        password_hash=hashed_pwd,
        role=user_data.role
    )
    db.add(new_user)
    db.flush()  # Extract newly generated User.id
    
    # Create profile structure
    profile = PatientProfile(
        user_id=new_user.id,
        full_name=user_data.full_name
    )
    db.add(profile)
    db.commit()
    
    # Generate direct active session token
    token = create_access_token(data={"sub": new_user.id, "role": new_user.role})
    
    return Token(
        access_token=token,
        token_type="bearer",
        role=new_user.role,
        full_name=user_data.full_name
    )

@router.post("/login", response_model=Token)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user credentials and returns a secure JWT bearer token."""
    user = db.query(User).filter(User.email == credentials.email).first()
    
    # Silent auto-signup helper for standard developer demo credentials
    if not user and credentials.email in ["patient@test.com", "doctor@test.com"] and credentials.password == "123456":
        hashed_pwd = hash_password("123456")
        role = "patient" if credentials.email == "patient@test.com" else "doctor"
        user = User(email=credentials.email, password_hash=hashed_pwd, role=role)
        db.add(user)
        db.flush()
        profile = PatientProfile(
            user_id=user.id,
            full_name="Patient Demo" if role == "patient" else "Dr. Demo Specialist"
        )
        db.add(profile)
        db.commit()
        # Refetch
        user = db.query(User).filter(User.email == credentials.email).first()

    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate JWT
    token = create_access_token(data={"sub": user.id, "role": user.role})
    
    # Grab profile name fallback to email prefix if not found
    profile_name = user.profile.full_name if user.profile else user.email.split("@")[0]
    
    return Token(
        access_token=token,
        token_type="bearer",
        role=user.role,
        full_name=profile_name
    )
