"""API依赖项

定义FastAPI的依赖注入，包括认证、权限、数据库会话等。
"""

from typing import Optional, Generator, Annotated
from fastapi import Depends, HTTPException, status, Request, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from redis import Redis
from influxdb_client import InfluxDBClient
import jwt
from datetime import datetime, timedelta
import hashlib
import time

from ..utils.database import DatabaseManager, get_db_session, get_redis_client, get_influxdb_client
from ..utils.config import get_settings
from ..utils.logger import logger
from .models import PaginationParams


# ==================== 配置和数据库依赖 ====================

def get_database_manager() -> DatabaseManager:
    """获取数据库管理器"""
    return DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """获取PostgreSQL数据库会话"""
    with get_db_session() as session:
        yield session


def get_redis() -> Generator[Redis, None, None]:
    """获取Redis客户端"""
    with get_redis_client() as client:
        yield client


def get_influxdb() -> Generator[InfluxDBClient, None, None]:
    """获取InfluxDB客户端"""
    with get_influxdb_client() as client:
        yield client


# ==================== 认证和授权 ====================

security = HTTPBearer(auto_error=False)


class AuthenticationError(HTTPException):
    """认证错误"""
    def __init__(self, detail: str = "认证失败"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(HTTPException):
    """授权错误"""
    def __init__(self, detail: str = "权限不足"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


class User:
    """用户模型"""
    def __init__(self, user_id: str, username: str, roles: list = None, permissions: list = None):
        self.user_id = user_id
        self.username = username
        self.roles = roles or []
        self.permissions = permissions or []
        
    def has_role(self, role: str) -> bool:
        """检查用户是否具有指定角色"""
        return role in self.roles
        
    def has_permission(self, permission: str) -> bool:
        """检查用户是否具有指定权限"""
        return permission in self.permissions


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    settings = get_settings()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=settings.jwt_expire_hours)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """验证令牌"""
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("令牌已过期")
    except jwt.JWTError:
        raise AuthenticationError("无效的令牌")


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    redis: Redis = Depends(get_redis)
) -> Optional[User]:
    """获取当前用户（可选认证）"""
    if not credentials:
        return None
        
    try:
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")
        username = payload.get("username")
        
        if not user_id:
            return None
            
        # 从Redis缓存中获取用户信息
        user_key = f"user:{user_id}"
        user_data = redis.hgetall(user_key)
        
        if user_data:
            roles = user_data.get("roles", "").split(",") if user_data.get("roles") else []
            permissions = user_data.get("permissions", "").split(",") if user_data.get("permissions") else []
            return User(user_id, username, roles, permissions)
        
        # 如果缓存中没有，创建基础用户
        return User(user_id, username)
        
    except AuthenticationError:
        return None


def require_authentication(
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """要求用户认证"""
    if not current_user:
        raise AuthenticationError("需要认证")
    return current_user


def require_role(required_role: str):
    """要求特定角色"""
    def role_checker(current_user: User = Depends(require_authentication)) -> User:
        if not current_user.has_role(required_role):
            raise AuthorizationError(f"需要角色: {required_role}")
        return current_user
    return role_checker


def require_permission(required_permission: str):
    """要求特定权限"""
    def permission_checker(current_user: User = Depends(require_authentication)) -> User:
        if not current_user.has_permission(required_permission):
            raise AuthorizationError(f"需要权限: {required_permission}")
        return current_user
    return permission_checker


# ==================== 请求限制和缓存 ====================

class RateLimiter:
    """请求频率限制器"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    def __call__(self, request: Request, redis: Redis = Depends(get_redis)):
        # 获取客户端IP
        client_ip = request.client.host
        current_time = int(time.time())
        window_start = current_time - self.window_seconds
        
        # Redis键
        key = f"rate_limit:{client_ip}"
        
        # 清理过期的请求记录
        redis.zremrangebyscore(key, 0, window_start)
        
        # 检查当前窗口内的请求数
        current_requests = redis.zcard(key)
        
        if current_requests >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="请求过于频繁，请稍后再试"
            )
        
        # 记录当前请求
        redis.zadd(key, {str(current_time): current_time})
        redis.expire(key, self.window_seconds)
        
        return True


# 预定义的限制器
rate_limit_strict = RateLimiter(max_requests=10, window_seconds=60)  # 每分钟10次
rate_limit_normal = RateLimiter(max_requests=100, window_seconds=60)  # 每分钟100次
rate_limit_loose = RateLimiter(max_requests=1000, window_seconds=60)  # 每分钟1000次


def cache_key_generator(*args, **kwargs) -> str:
    """生成缓存键"""
    # 将参数转换为字符串并生成哈希
    key_data = f"{args}_{kwargs}"
    return hashlib.md5(key_data.encode()).hexdigest()


def cached_response(expire_seconds: int = 300):
    """缓存响应装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            redis = kwargs.get('redis')
            if not redis:
                return await func(*args, **kwargs)
            
            # 生成缓存键
            cache_key = f"api_cache:{func.__name__}:{cache_key_generator(*args, **kwargs)}"
            
            # 尝试从缓存获取
            cached_result = redis.get(cache_key)
            if cached_result:
                logger.info(f"缓存命中: {cache_key}")
                return cached_result
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            redis.setex(cache_key, expire_seconds, result)
            logger.info(f"缓存设置: {cache_key}")
            
            return result
        return wrapper
    return decorator


# ==================== 分页和查询参数 ====================

def get_pagination_params(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(20, ge=1, le=100, description="每页大小")
) -> PaginationParams:
    """获取分页参数"""
    return PaginationParams(page=page, size=size)


def get_common_query_params(
    q: Optional[str] = Query(None, description="搜索关键词"),
    sort_by: Optional[str] = Query(None, description="排序字段"),
    sort_order: Optional[str] = Query("asc", regex="^(asc|desc)$", description="排序方向"),
    filter_by: Optional[str] = Query(None, description="过滤字段"),
    filter_value: Optional[str] = Query(None, description="过滤值")
) -> dict:
    """获取通用查询参数"""
    return {
        "q": q,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "filter_by": filter_by,
        "filter_value": filter_value
    }


# ==================== 请求验证 ====================

def validate_file_upload(request: Request):
    """验证文件上传请求"""
    content_type = request.headers.get("content-type", "")
    
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请求必须是multipart/form-data格式"
        )
    
    # 检查文件大小限制
    content_length = request.headers.get("content-length")
    if content_length:
        size = int(content_length)
        max_size = get_settings().max_file_size  # 从配置获取最大文件大小
        if size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"文件大小超过限制 ({max_size} bytes)"
            )
    
    return True


def validate_coordinates(
    latitude: float = Query(..., ge=-90, le=90, description="纬度"),
    longitude: float = Query(..., ge=-180, le=180, description="经度")
) -> dict:
    """验证坐标参数"""
    return {"latitude": latitude, "longitude": longitude}


def validate_date_range(
    start_date: Optional[str] = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="结束日期 (YYYY-MM-DD)")
) -> dict:
    """验证日期范围参数"""
    from datetime import datetime
    
    result = {}
    
    if start_date:
        try:
            result["start_date"] = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="开始日期格式错误，应为 YYYY-MM-DD"
            )
    
    if end_date:
        try:
            result["end_date"] = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="结束日期格式错误，应为 YYYY-MM-DD"
            )
    
    # 验证日期范围
    if "start_date" in result and "end_date" in result:
        if result["end_date"] < result["start_date"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="结束日期不能早于开始日期"
            )
    
    return result


# ==================== 健康检查依赖 ====================

def check_database_health(
    db: Session = Depends(get_db),
    redis: Redis = Depends(get_redis),
    influxdb: InfluxDBClient = Depends(get_influxdb)
) -> dict:
    """检查数据库健康状态"""
    health_status = {
        "postgresql": False,
        "redis": False,
        "influxdb": False
    }
    
    # 检查PostgreSQL
    try:
        db.execute("SELECT 1")
        health_status["postgresql"] = True
    except Exception as e:
        logger.error(f"PostgreSQL健康检查失败: {e}")
    
    # 检查Redis
    try:
        redis.ping()
        health_status["redis"] = True
    except Exception as e:
        logger.error(f"Redis健康检查失败: {e}")
    
    # 检查InfluxDB
    try:
        influxdb.ping()
        health_status["influxdb"] = True
    except Exception as e:
        logger.error(f"InfluxDB健康检查失败: {e}")
    
    return health_status


# ==================== 类型注解别名 ====================

# 常用依赖的类型注解
DBSession = Annotated[Session, Depends(get_db)]
RedisClient = Annotated[Redis, Depends(get_redis)]
InfluxDBClient = Annotated[InfluxDBClient, Depends(get_influxdb)]
CurrentUser = Annotated[Optional[User], Depends(get_current_user)]
AuthenticatedUser = Annotated[User, Depends(require_authentication)]
PaginationDep = Annotated[PaginationParams, Depends(get_pagination_params)]
CommonQueryDep = Annotated[dict, Depends(get_common_query_params)]

# 权限依赖
AdminUser = Annotated[User, Depends(require_role("admin"))]
DataAnalystUser = Annotated[User, Depends(require_role("data_analyst"))]
ModelDeveloperUser = Annotated[User, Depends(require_role("model_developer"))]

# 限流依赖
StrictRateLimit = Annotated[bool, Depends(rate_limit_strict)]
NormalRateLimit = Annotated[bool, Depends(rate_limit_normal)]
LooseRateLimit = Annotated[bool, Depends(rate_limit_loose)]