#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气候数据分析系统 - 主应用程序入口

这是气候数据分析系统的主要入口点，负责启动FastAPI应用程序，
初始化数据库连接，配置路由和中间件。

作者: 气候数据分析团队
创建时间: 2024
版本: 1.0.0
"""

import asyncio
import sys
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from src.utils.config import settings
from src.utils.logger import get_logger
from src.data_processing.data_storage import DataStorage
from src.api.routes import api_router
from src.api.middleware import (
    LoggingMiddleware,
    SecurityMiddleware,
    RateLimitMiddleware
)

# 初始化日志
logger = get_logger(__name__)

# 全局数据存储实例
data_storage: DataStorage = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理"""
    global data_storage
    
    # 启动时初始化
    logger.info("🚀 启动气候数据分析系统...")
    
    try:
        # 初始化数据存储
        data_storage = DataStorage()
        await data_storage.initialize()
        
        # 将数据存储实例添加到应用状态
        app.state.data_storage = data_storage
        
        logger.info("✅ 系统初始化完成")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ 系统初始化失败: {e}")
        raise
    
    finally:
        # 关闭时清理资源
        logger.info("🔄 正在关闭系统...")
        
        if data_storage:
            await data_storage.close()
        
        logger.info("✅ 系统已安全关闭")


# 创建FastAPI应用实例
app = FastAPI(
    title=settings.APP_NAME,
    description="基于AI的气候数据分析与预测系统",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    debug=settings.DEBUG
)

# 添加中间件
# 1. 安全中间件
app.add_middleware(SecurityMiddleware)

# 2. 信任主机中间件
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", settings.HOST]
    )

# 3. CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Gzip压缩中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 5. 限流中间件
app.add_middleware(RateLimitMiddleware)

# 6. 日志中间件
app.add_middleware(LoggingMiddleware)

# 挂载静态文件
static_dir = project_root / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 注册API路由
app.include_router(api_router, prefix="/api")


@app.get("/", tags=["根路径"])
async def root():
    """根路径 - 返回系统信息"""
    return {
        "message": "欢迎使用气候数据分析系统",
        "version": settings.VERSION,
        "docs": "/docs",
        "status": "运行中"
    }


@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查端点"""
    try:
        # 检查数据库连接
        if hasattr(app.state, 'data_storage') and app.state.data_storage:
            # 这里可以添加更详细的健康检查逻辑
            db_status = "healthy"
        else:
            db_status = "unhealthy"
        
        return {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "version": settings.VERSION,
            "database": db_status,
            "environment": "development" if settings.DEBUG else "production"
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail="服务不可用")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "message": "系统遇到了一个意外错误，请稍后重试",
            "request_id": getattr(request.state, 'request_id', None)
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": getattr(request.state, 'request_id', None)
        }
    )


def main():
    """主函数 - 启动应用程序"""
    logger.info(f"🌍 启动气候数据分析系统 v{settings.VERSION}")
    logger.info(f"📍 运行环境: {'开发' if settings.DEBUG else '生产'}")
    logger.info(f"🌐 服务地址: http://{settings.HOST}:{settings.PORT}")
    logger.info(f"📚 API文档: http://{settings.HOST}:{settings.PORT}/docs")
    
    # 启动服务器
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level="debug" if settings.DEBUG else "info",
        access_log=True,
        use_colors=True
    )


if __name__ == "__main__":
    main()