#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API中间件模块

提供各种中间件功能，包括日志记录、安全控制、限流等。
"""

import time
import uuid
from typing import Callable
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """日志记录中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 记录请求信息
        logger.info(
            f"请求开始 - {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent")
            }
        )
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录响应信息
            logger.info(
                f"请求完成 - {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": round(process_time, 4)
                }
            )
            
            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            
            return response
            
        except Exception as e:
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录错误
            logger.error(
                f"请求失败 - {request.method} {request.url.path} - {str(e)}",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time": round(process_time, 4)
                },
                exc_info=True
            )
            
            # 返回错误响应
            return JSONResponse(
                status_code=500,
                content={
                    "error": "内部服务器错误",
                    "request_id": request_id
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Process-Time": str(round(process_time, 4))
                }
            )


class SecurityMiddleware(BaseHTTPMiddleware):
    """安全中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 检查请求头大小
        if len(str(request.headers)) > 8192:  # 8KB限制
            return JSONResponse(
                status_code=413,
                content={"error": "请求头过大"}
            )
        
        # 检查Content-Length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"error": "请求体过大"}
            )
        
        # 处理请求
        response = await call_next(request)
        
        # 添加安全响应头
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """限流中间件"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls  # 允许的调用次数
        self.period = period  # 时间窗口（秒）
        self.clients = defaultdict(list)  # 客户端请求记录
    
    def _get_client_id(self, request: Request) -> str:
        """获取客户端标识"""
        # 优先使用认证用户ID
        if hasattr(request.state, 'user_id'):
            return f"user:{request.state.user_id}"
        
        # 使用IP地址
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    def _is_rate_limited(self, client_id: str) -> bool:
        """检查是否超出限流"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.period)
        
        # 清理过期记录
        self.clients[client_id] = [
            timestamp for timestamp in self.clients[client_id]
            if timestamp > cutoff
        ]
        
        # 检查是否超出限制
        if len(self.clients[client_id]) >= self.calls:
            return True
        
        # 记录当前请求
        self.clients[client_id].append(now)
        return False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 跳过健康检查和静态文件
        if request.url.path in ["/health", "/"] or request.url.path.startswith("/static"):
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        
        if self._is_rate_limited(client_id):
            logger.warning(
                f"限流触发 - 客户端: {client_id}",
                extra={
                    "client_id": client_id,
                    "path": request.url.path,
                    "method": request.method
                }
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "请求过于频繁",
                    "message": f"每{self.period}秒最多允许{self.calls}次请求"
                },
                headers={
                    "Retry-After": str(self.period)
                }
            )
        
        return await call_next(request)


class CacheMiddleware(BaseHTTPMiddleware):
    """缓存中间件"""
    
    def __init__(self, app, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl  # 缓存TTL（秒）
        self.cache = {}  # 简单内存缓存
    
    def _get_cache_key(self, request: Request) -> str:
        """生成缓存键"""
        return f"{request.method}:{request.url.path}:{request.url.query}"
    
    def _is_cacheable(self, request: Request, response: Response) -> bool:
        """判断是否可缓存"""
        # 只缓存GET请求
        if request.method != "GET":
            return False
        
        # 只缓存成功响应
        if response.status_code != 200:
            return False
        
        # 跳过某些路径
        skip_paths = ["/health", "/auth", "/system"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return False
        
        return True
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        cache_key = self._get_cache_key(request)
        
        # 检查缓存
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"缓存命中: {cache_key}")
                return Response(
                    content=cached_data["content"],
                    status_code=cached_data["status_code"],
                    headers=dict(cached_data["headers"]),
                    media_type=cached_data["media_type"]
                )
            else:
                # 缓存过期，删除
                del self.cache[cache_key]
        
        # 处理请求
        response = await call_next(request)
        
        # 缓存响应
        if self._is_cacheable(request, response):
            # 读取响应内容
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # 存储到缓存
            self.cache[cache_key] = ({
                "content": body,
                "status_code": response.status_code,
                "headers": response.headers,
                "media_type": response.media_type
            }, time.time())
            
            logger.debug(f"缓存存储: {cache_key}")
            
            # 重新创建响应
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        return response


__all__ = [
    "LoggingMiddleware",
    "SecurityMiddleware", 
    "RateLimitMiddleware",
    "CacheMiddleware"
]