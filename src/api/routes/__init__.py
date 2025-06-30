#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API路由模块初始化

这个模块负责组织和导出所有的API路由。
"""

from fastapi import APIRouter

from .data import router as data_router
from .models import router as models_router
from .predictions import router as predictions_router
from .auth import router as auth_router
from .system import router as system_router

# 创建主API路由器
api_router = APIRouter()

# 注册子路由
api_router.include_router(
    auth_router,
    prefix="/auth",
    tags=["认证"]
)

api_router.include_router(
    data_router,
    prefix="/data",
    tags=["数据管理"]
)

api_router.include_router(
    models_router,
    prefix="/models",
    tags=["模型管理"]
)

api_router.include_router(
    predictions_router,
    prefix="/predictions",
    tags=["预测任务"]
)

api_router.include_router(
    system_router,
    prefix="/system",
    tags=["系统管理"]
)

__all__ = ["api_router"]