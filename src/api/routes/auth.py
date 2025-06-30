#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认证API路由

提供用户认证、授权、注册等功能。
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from jose import JWTError, jwt

from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)
router = APIRouter()

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT认证
security = HTTPBearer()

# JWT配置
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


# 请求模型
class UserLogin(BaseModel):
    """用户登录请求"""
    username: str = Field(..., description="用户名或邮箱")
    password: str = Field(..., min_length=6, description="密码")


class UserRegister(BaseModel):
    """用户注册请求"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱地址")
    password: str = Field(..., min_length=6, description="密码")
    full_name: Optional[str] = Field(None, max_length=100, description="全名")


class PasswordChange(BaseModel):
    """密码修改请求"""
    old_password: str = Field(..., description="旧密码")
    new_password: str = Field(..., min_length=6, description="新密码")


class PasswordReset(BaseModel):
    """密码重置请求"""
    email: EmailStr = Field(..., description="邮箱地址")


# 响应模型
class Token(BaseModel):
    """令牌响应"""
    access_token: str = Field(description="访问令牌")
    refresh_token: str = Field(description="刷新令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(description="过期时间（秒）")


class UserInfo(BaseModel):
    """用户信息"""
    id: str = Field(description="用户ID")
    username: str = Field(description="用户名")
    email: str = Field(description="邮箱")
    full_name: Optional[str] = Field(description="全名")
    is_active: bool = Field(description="是否激活")
    is_superuser: bool = Field(description="是否超级用户")
    created_at: datetime = Field(description="创建时间")
    last_login: Optional[datetime] = Field(description="最后登录时间")


# 工具函数
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """获取密码哈希"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """创建刷新令牌"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> dict:
    """验证令牌"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != token_type:
            raise JWTError("Invalid token type")
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )


# 依赖函数
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """获取当前用户"""
    token = credentials.credentials
    payload = verify_token(token)
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 这里应该从数据库获取用户信息
    # 暂时返回模拟数据
    user = {
        "id": user_id,
        "username": "admin",
        "email": "admin@climate-system.com",
        "full_name": "系统管理员",
        "is_active": True,
        "is_superuser": True,
        "created_at": datetime.now(),
        "last_login": datetime.now()
    }
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    """获取当前活跃用户"""
    if not current_user.get("is_active"):
        raise HTTPException(status_code=400, detail="用户账户已被禁用")
    return current_user


# API路由
@router.post("/login", response_model=Token, summary="用户登录")
async def login(user_login: UserLogin):
    """用户登录"""
    try:
        # 这里应该从数据库验证用户
        # 暂时使用硬编码的管理员账户
        if user_login.username == "admin" and user_login.password == "admin123":
            user_id = "admin-user-id"
            
            # 创建令牌
            access_token = create_access_token(data={"sub": user_id})
            refresh_token = create_refresh_token(data={"sub": user_id})
            
            logger.info(f"用户登录成功: {user_login.username}")
            
            return Token(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )
        else:
            logger.warning(f"用户登录失败: {user_login.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"登录过程中发生错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登录服务暂时不可用"
        )


@router.post("/register", response_model=UserInfo, summary="用户注册")
async def register(user_register: UserRegister):
    """用户注册"""
    try:
        # 检查用户名和邮箱是否已存在
        # 这里应该查询数据库
        
        # 创建用户
        hashed_password = get_password_hash(user_register.password)
        
        # 这里应该保存到数据库
        user_info = UserInfo(
            id="new-user-id",
            username=user_register.username,
            email=user_register.email,
            full_name=user_register.full_name,
            is_active=True,
            is_superuser=False,
            created_at=datetime.now(),
            last_login=None
        )
        
        logger.info(f"新用户注册: {user_register.username}")
        
        return user_info
        
    except Exception as e:
        logger.error(f"用户注册失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注册服务暂时不可用"
        )


@router.post("/refresh", response_model=Token, summary="刷新令牌")
async def refresh_token(refresh_token: str):
    """刷新访问令牌"""
    try:
        payload = verify_token(refresh_token, "refresh")
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的刷新令牌"
            )
        
        # 创建新的访问令牌
        new_access_token = create_access_token(data={"sub": user_id})
        new_refresh_token = create_refresh_token(data={"sub": user_id})
        
        return Token(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"令牌刷新失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="令牌刷新服务暂时不可用"
        )


@router.get("/me", response_model=UserInfo, summary="获取当前用户信息")
async def get_me(current_user: dict = Depends(get_current_active_user)):
    """获取当前用户信息"""
    return UserInfo(**current_user)


@router.put("/me", response_model=UserInfo, summary="更新用户信息")
async def update_me(
    user_update: dict,
    current_user: dict = Depends(get_current_active_user)
):
    """更新当前用户信息"""
    try:
        # 这里应该更新数据库中的用户信息
        updated_user = current_user.copy()
        updated_user.update(user_update)
        
        logger.info(f"用户信息更新: {current_user['username']}")
        
        return UserInfo(**updated_user)
        
    except Exception as e:
        logger.error(f"用户信息更新失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="用户信息更新失败"
        )


@router.post("/change-password", summary="修改密码")
async def change_password(
    password_change: PasswordChange,
    current_user: dict = Depends(get_current_active_user)
):
    """修改用户密码"""
    try:
        # 验证旧密码
        # 这里应该从数据库获取用户的密码哈希进行验证
        
        # 更新密码
        new_password_hash = get_password_hash(password_change.new_password)
        
        # 这里应该更新数据库中的密码
        
        logger.info(f"用户密码修改: {current_user['username']}")
        
        return {"message": "密码修改成功"}
        
    except Exception as e:
        logger.error(f"密码修改失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="密码修改失败"
        )


@router.post("/reset-password", summary="重置密码")
async def reset_password(password_reset: PasswordReset):
    """重置用户密码"""
    try:
        # 这里应该实现密码重置逻辑
        # 1. 验证邮箱是否存在
        # 2. 生成重置令牌
        # 3. 发送重置邮件
        
        logger.info(f"密码重置请求: {password_reset.email}")
        
        return {"message": "密码重置邮件已发送"}
        
    except Exception as e:
        logger.error(f"密码重置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="密码重置服务暂时不可用"
        )


@router.post("/logout", summary="用户登出")
async def logout(current_user: dict = Depends(get_current_active_user)):
    """用户登出"""
    try:
        # 这里可以实现令牌黑名单机制
        
        logger.info(f"用户登出: {current_user['username']}")
        
        return {"message": "登出成功"}
        
    except Exception as e:
        logger.error(f"用户登出失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登出服务暂时不可用"
        )