#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web应用主模块

提供气候数据分析和预测的Web界面。
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import zipfile
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()

# Web框架
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, session
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# 数据处理
import pandas as pd
import numpy as np
import xarray as xr

# 项目模块
from ..data_processing.data_collector import DataCollector
from ..data_processing.data_storage import DataStorage
from ..data_processing.data_processor import DataProcessor
from ..ml.model_manager import ModelManager
from ..ml.prediction_engine import PredictionEngine
from ..visualization.charts import ChartGenerator, ChartConfig, ChartType
from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.natural_language_query import convert_nl_to_sql
import mysql.connector

logger = get_logger(__name__)
config = get_config()

# 创建Flask应用
app = Flask(__name__)
app.secret_key = getattr(config, 'SECRET_KEY', 'climate-research-secret-key-2024')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = str(getattr(config, 'DATA_ROOT_PATH', Path('data')) / 'uploads')

# 启用CORS
CORS(app)

# 登录管理
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '请先登录以访问此页面。'

# 创建必要的目录
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# 全局变量
data_collector = None
data_storage = None
data_processor = None
model_manager = None
prediction_engine = None
chart_generator = None


class User(UserMixin):
    """用户类"""
    def __init__(self, user_id, username, password_hash):
        self.id = user_id
        self.username = username
        self.password_hash = password_hash


# 简单的用户存储（生产环境应使用数据库）
users = {
    'admin': User('1', 'admin', generate_password_hash('admin123')),
    'researcher': User('2', 'researcher', generate_password_hash('research123'))
}


@login_manager.user_loader
def load_user(user_id):
    """加载用户"""
    for user in users.values():
        if user.id == user_id:
            return user
    return None


def initialize_app():
    """初始化应用"""
    global data_collector, data_storage, data_processor, model_manager, prediction_engine, chart_generator
    
    try:
        logger.info("初始化Web应用组件")
        
        # 初始化各个组件
        data_storage = DataStorage()
        data_processor = DataProcessor()
        model_manager = ModelManager()
        prediction_engine = PredictionEngine()
        chart_generator = ChartGenerator()
        
        logger.info("Web应用组件初始化完成")
        
    except Exception as e:
        logger.error(f"初始化应用失败: {e}")


# 在应用上下文中初始化组件
with app.app_context():
    initialize_app()


# 路由定义
@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users.get(username)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('用户名或密码错误')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """登出"""
    logout_user()
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    """仪表板"""
    try:
        # 获取系统统计信息
        stats = {
            'data_records': 0,
            'models': 0,
            'predictions': 0,
            'charts': 0
        }
        
        if data_storage:
            storage_stats = data_storage.get_storage_stats()
            stats['data_records'] = storage_stats.get('database_records', 0)
        
        if model_manager:
            models = model_manager.list_models()
            stats['models'] = len(models)
        
        if prediction_engine:
            tasks = prediction_engine.list_tasks()
            stats['predictions'] = len(tasks)
        
        if chart_generator:
            chart_stats = chart_generator.get_chart_statistics()
            stats['charts'] = chart_stats.get('total_charts', 0)
        
        return render_template('dashboard.html', stats=stats)
        
    except Exception as e:
        logger.error(f"加载仪表板失败: {e}")
        flash('加载仪表板失败')
        return render_template('dashboard.html', stats={})


@app.route('/data')
@login_required
def data_management():
    """数据管理页面"""
    return render_template('data_management.html')


@app.route('/climate-data')
@login_required
def climate_data_page():
    """气候数据探索页面"""
    return render_template('climate_data.html')


@app.route('/climate/data', methods=['GET'])
def get_climate_data():
    """获取气候数据，支持筛选"""
    try:
        # 获取筛选参数
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        location = request.args.get('location')
        variable = request.args.get('variable')

        # 构建基础查询
        query = "SELECT * FROM climate_data WHERE 1=1"
        params = []

        if start_date:
            query += " AND start_time >= %s"
            params.append(start_date)
        if end_date:
            query += " AND end_time <= %s"
            params.append(end_date)
        if location:
            # 简单的位置查询，可能需要更复杂的地理空间查询
            query += " AND location LIKE %s"
            params.append(f"%{location}%")
        if variable:
            query += " AND data_type = %s"
            params.append(variable)

        query += " ORDER BY start_time DESC LIMIT 1000" # 限制返回结果数量

        connection = data_storage.get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query, tuple(params))
        records = cursor.fetchall()
        cursor.close()
        connection.close()

        return jsonify({'data': records})

    except Exception as e:
        logger.error(f"获取气候数据失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/download', methods=['GET'])
def download_data():
    """下载筛选后的数据为CSV"""
    try:
        # (与get_climate_data类似的筛选逻辑)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        location = request.args.get('location')
        variable = request.args.get('variable')

        query = "SELECT * FROM climate_data WHERE 1=1"
        params = []

        if start_date:
            query += " AND start_time >= %s"
            params.append(start_date)
        if end_date:
            query += " AND end_time <= %s"
            params.append(end_date)
        if location:
            query += " AND location LIKE %s"
            params.append(f"%{location}%")
        if variable:
            query += " AND data_type = %s"
            params.append(variable)

        connection = data_storage.get_connection()
        df = pd.read_sql(query, connection, params=params)
        connection.close()

        # 创建CSV内存文件
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='climate_data.csv'
        )

    except Exception as e:
        logger.error(f"下载数据失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/natural_language_query', methods=['POST'])
@login_required
def natural_language_query():
    """自然语言查询"""
    try:
        data = request.get_json()
        nl_query = data.get('query')
        api_key = os.environ.get("DASHSCOPE_API_KEY")

        if not nl_query:
            return jsonify({"error": "Query is required"}), 400

        if not api_key:
            return jsonify({"error": "DASHSCOPE_API_KEY is not set"}), 500

        # This is a simplified schema. In a real application, you would fetch this dynamically.
        table_schema = "CREATE TABLE climate_data_records (id INT, source VARCHAR(255), data_type VARCHAR(255), location VARCHAR(255), start_time DATETIME, end_time DATETIME)"

        sql_query = convert_nl_to_sql(nl_query, api_key, table_schema)

        # Execute the SQL query
        db_config = {
            'host': os.environ.get('DB_HOST', 'localhost'),
            'user': os.environ.get('DB_USER', 'root'),
            'password': os.environ.get('DB_PASSWORD', ''),
            'database': os.environ.get('DB_NAME', 'climate_data')
        }
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql_query)
        result = cursor.fetchall()
        cursor.close()
        conn.close()

        return jsonify({"data": result})

    except Exception as e:
        logger.error(f"自然语言查询失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/data/collect', methods=['POST'])
@login_required
def collect_data():
    """收集数据API"""
    try:
        data = request.get_json()
        
        source = data.get('source', 'noaa')
        start_date = datetime.fromisoformat(data.get('start_date'))
        end_date = datetime.fromisoformat(data.get('end_date'))
        variables = data.get('variables', ['temperature'])
        location = data.get('location')
        
        # 异步收集数据
        async def collect_async():
            async with DataCollector() as collector:
                if source == 'noaa':
                    result = await collector.collect_noaa_data(
                        start_date=start_date,
                        end_date=end_date,
                        stations=[location] if location else None,
                        variables=variables
                    )
                elif source == 'satellite':
                    result = await collector.collect_satellite_data(
                        start_date=start_date,
                        end_date=end_date,
                        region=location
                    )
                elif source == 'reanalysis':
                    result = await collector.collect_reanalysis_data(
                        start_date=start_date,
                        end_date=end_date,
                        variables=variables,
                        region=location
                    )
                else:
                    raise ValueError(f"不支持的数据源: {source}")
                
                # 保存数据
                if data_storage and result is not None:
                    record_id = await data_storage.save_climate_data_record(
                        source=source,
                        data_type='time_series',
                        location=location or 'global',
                        start_time=start_date,
                        end_time=end_date,
                        variables=variables,
                        file_path=None,
                        metadata={'collection_time': datetime.now().isoformat()}
                    )
                    
                    # 保存DataFrame
                    if isinstance(result, pd.DataFrame):
                        file_path = f"data_{record_id}.parquet"
                        await data_storage.save_dataframe(result, file_path)
                        await data_storage.update_record_file_path(record_id, file_path)
                
                return result
        
        # 运行异步任务
        try:
            result = asyncio.run(collect_async())
        except RuntimeError:
            # 如果已经在事件循环中，使用 asyncio.create_task
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, collect_async())
                    result = future.result()
            else:
                result = loop.run_until_complete(collect_async())
        
        if result is not None:
            return jsonify({
                'success': True,
                'message': '数据收集成功',
                'data_shape': result.shape if hasattr(result, 'shape') else None
            })
        else:
            return jsonify({
                'success': False,
                'message': '数据收集失败'
            }), 400
            
    except Exception as e:
        logger.error(f"数据收集失败: {e}")
        return jsonify({
            'success': False,
            'message': f'数据收集失败: {str(e)}'
        }), 500


@app.route('/api/data/upload', methods=['POST'])
@login_required
def upload_data():
    """上传数据文件API"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '没有选择文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '没有选择文件'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 尝试读取文件
            try:
                if filename.endswith('.csv'):
                    data = pd.read_csv(filepath)
                elif filename.endswith('.json'):
                    data = pd.read_json(filepath)
                elif filename.endswith('.parquet'):
                    data = pd.read_parquet(filepath)
                elif filename.endswith('.nc'):
                    data = xr.open_dataset(filepath)
                else:
                    return jsonify({
                        'success': False,
                        'message': '不支持的文件格式'
                    }), 400
                
                # 保存到数据存储
                if data_storage:
                    # 创建异步函数来保存记录
                    async def save_record():
                        return await data_storage.save_climate_data_record(
                            source='upload',
                            data_type='file',
                            location='unknown',
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            variables=[],
                            file_path=filepath,
                            metadata={
                                'filename': filename,
                                'upload_time': datetime.now().isoformat(),
                                'user': current_user.username
                            }
                        )

                    # 运行异步函数
                    try:
                        record_id = asyncio.run(save_record())
                    except RuntimeError:
                        # 如果已经在事件循环中，使用线程池执行
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, save_record())
                                record_id = future.result()
                        else:
                            record_id = loop.run_until_complete(save_record())
                
                return jsonify({
                    'success': True,
                    'message': '文件上传成功',
                    'filename': filename,
                    'data_shape': data.shape if hasattr(data, 'shape') else None
                })
                
            except Exception as e:
                os.remove(filepath)  # 删除无效文件
                return jsonify({
                    'success': False,
                    'message': f'文件读取失败: {str(e)}'
                }), 400
                
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return jsonify({
            'success': False,
            'message': f'文件上传失败: {str(e)}'
        }), 500


@app.route('/api/data/list')
@login_required
def list_data():
    """列出数据记录API"""
    try:
        if not data_storage:
            return jsonify({'success': False, 'message': '数据存储未初始化'}), 500
        
        # 获取查询参数
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        source = request.args.get('source')
        data_type = request.args.get('data_type')
        
        # 搜索数据记录
        records = data_storage.search_data_records(
            source=source,
            data_type=data_type,
            limit=per_page,
            offset=(page - 1) * per_page
        )
        
        return jsonify({
            'success': True,
            'records': records,
            'page': page,
            'per_page': per_page
        })
        
    except Exception as e:
        logger.error(f"列出数据记录失败: {e}")
        return jsonify({
            'success': False,
            'message': f'列出数据记录失败: {str(e)}'
        }), 500


@app.route('/models')
@login_required
def model_management():
    """模型管理页面"""
    return render_template('model_management.html')


@app.route('/api/models/train', methods=['POST'])
@login_required
def train_model():
    """训练模型API"""
    try:
        data = request.get_json()
        
        model_name = data.get('model_name')
        model_type = data.get('model_type', 'linear_regression')
        data_record_id = data.get('data_record_id')
        target_variable = data.get('target_variable')
        feature_variables = data.get('feature_variables', [])
        
        if not all([model_name, data_record_id, target_variable]):
            return jsonify({
                'success': False,
                'message': '缺少必要参数'
            }), 400
        
        # 加载训练数据
        if data_storage:
            train_data = data_storage.load_dataframe(f"data_{data_record_id}.parquet")
        else:
            return jsonify({
                'success': False,
                'message': '数据存储未初始化'
            }), 500
        
        # 训练模型
        if model_manager:
            model_id = model_manager.train_model(
                model_name=model_name,
                model_type=model_type,
                train_data=train_data,
                target_variable=target_variable,
                feature_variables=feature_variables
            )
            
            return jsonify({
                'success': True,
                'message': '模型训练成功',
                'model_id': model_id
            })
        else:
            return jsonify({
                'success': False,
                'message': '模型管理器未初始化'
            }), 500
            
    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        return jsonify({
            'success': False,
            'message': f'模型训练失败: {str(e)}'
        }), 500


@app.route('/api/models/list')
@login_required
def list_models():
    """列出模型API"""
    try:
        if not model_manager:
            return jsonify({'success': False, 'message': '模型管理器未初始化'}), 500
        
        models = model_manager.list_models()
        
        return jsonify({
            'success': True,
            'models': models
        })
        
    except Exception as e:
        logger.error(f"列出模型失败: {e}")
        return jsonify({
            'success': False,
            'message': f'列出模型失败: {str(e)}'
        }), 500


@app.route('/predictions')
@login_required
def prediction_management():
    """预测管理页面"""
    return render_template('prediction_management.html')


@app.route('/api/predictions/create', methods=['POST'])
@login_required
def create_prediction():
    """创建预测任务API"""
    try:
        data = request.get_json()
        
        model_id = data.get('model_id')
        prediction_type = data.get('prediction_type', 'time_series')
        input_data_id = data.get('input_data_id')
        forecast_horizon = data.get('forecast_horizon', 30)
        
        if not all([model_id, input_data_id]):
            return jsonify({
                'success': False,
                'message': '缺少必要参数'
            }), 400
        
        # 加载输入数据
        if data_storage:
            input_data = data_storage.load_dataframe(f"data_{input_data_id}.parquet")
        else:
            return jsonify({
                'success': False,
                'message': '数据存储未初始化'
            }), 500
        
        # 创建预测任务
        if prediction_engine:
            task_id = prediction_engine.create_prediction_task(
                model_id=model_id,
                prediction_type=prediction_type,
                input_data=input_data,
                forecast_horizon=forecast_horizon
            )
            
            return jsonify({
                'success': True,
                'message': '预测任务创建成功',
                'task_id': task_id
            })
        else:
            return jsonify({
                'success': False,
                'message': '预测引擎未初始化'
            }), 500
            
    except Exception as e:
        logger.error(f"创建预测任务失败: {e}")
        return jsonify({
            'success': False,
            'message': f'创建预测任务失败: {str(e)}'
        }), 500


@app.route('/api/predictions/run/<task_id>', methods=['POST'])
@login_required
def run_prediction(task_id):
    """运行预测任务API"""
    try:
        if not prediction_engine:
            return jsonify({'success': False, 'message': '预测引擎未初始化'}), 500
        
        # 异步运行预测
        async def run_async():
            return await prediction_engine.run_prediction_task(task_id)
        
        try:
            result = asyncio.run(run_async())
        except RuntimeError:
            # 如果已经在事件循环中，使用线程池执行
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, run_async())
                    result = future.result()
            else:
                result = loop.run_until_complete(run_async())
        
        return jsonify({
            'success': True,
            'message': '预测任务运行成功',
            'result': result.to_dict() if hasattr(result, 'to_dict') else str(result)
        })
        
    except Exception as e:
        logger.error(f"运行预测任务失败: {e}")
        return jsonify({
            'success': False,
            'message': f'运行预测任务失败: {str(e)}'
        }), 500


@app.route('/api/predictions/list')
@login_required
def list_predictions():
    """列出预测任务API"""
    try:
        if not prediction_engine:
            return jsonify({'success': False, 'message': '预测引擎未初始化'}), 500
        
        tasks = prediction_engine.list_tasks()
        
        return jsonify({
            'success': True,
            'tasks': tasks
        })
        
    except Exception as e:
        logger.error(f"列出预测任务失败: {e}")
        return jsonify({
            'success': False,
            'message': f'列出预测任务失败: {str(e)}'
        }), 500


@app.route('/visualization')
@login_required
def visualization():
    """可视化页面"""
    return render_template('visualization.html')


@app.route('/api/charts/create', methods=['POST'])
@login_required
def create_chart():
    """创建图表API"""
    try:
        data = request.get_json()
        
        chart_type = data.get('chart_type', 'time_series')
        data_record_id = data.get('data_record_id')
        title = data.get('title', '气候数据图表')
        interactive = data.get('interactive', False)
        
        if not data_record_id:
            return jsonify({
                'success': False,
                'message': '缺少数据记录ID'
            }), 400
        
        # 加载数据
        if data_storage:
            chart_data = data_storage.load_dataframe(f"data_{data_record_id}.parquet")
        else:
            return jsonify({
                'success': False,
                'message': '数据存储未初始化'
            }), 500
        
        # 创建图表配置
        config = ChartConfig(
            title=title,
            chart_type=chart_type,
            interactive=interactive
        )
        
        # 生成图表
        if chart_generator:
            if chart_type == 'time_series':
                chart_path = chart_generator.create_time_series_chart(chart_data, config)
            elif chart_type == 'correlation':
                chart_path = chart_generator.create_correlation_matrix(chart_data, config)
            elif chart_type == 'distribution':
                variable = data.get('variable')
                if not variable:
                    numeric_cols = chart_data.select_dtypes(include=[np.number]).columns
                    variable = numeric_cols[0] if len(numeric_cols) > 0 else None
                
                if variable:
                    chart_path = chart_generator.create_distribution_chart(chart_data, variable, config)
                else:
                    return jsonify({
                        'success': False,
                        'message': '没有找到数值变量'
                    }), 400
            else:
                chart_path = chart_generator.create_comparison_chart(chart_data, config, chart_type)
            
            # 返回图表路径
            chart_url = f"/api/charts/view/{Path(chart_path).name}"
            
            return jsonify({
                'success': True,
                'message': '图表创建成功',
                'chart_url': chart_url,
                'chart_path': chart_path
            })
        else:
            return jsonify({
                'success': False,
                'message': '图表生成器未初始化'
            }), 500
            
    except Exception as e:
        logger.error(f"创建图表失败: {e}")
        return jsonify({
            'success': False,
            'message': f'创建图表失败: {str(e)}'
        }), 500


@app.route('/api/charts/view/<filename>')
@login_required
def view_chart(filename):
    """查看图表API"""
    try:
        if chart_generator:
            chart_path = chart_generator.output_path / filename
            if chart_path.exists():
                return send_file(str(chart_path))
        
        return jsonify({
            'success': False,
            'message': '图表文件不存在'
        }), 404
        
    except Exception as e:
        logger.error(f"查看图表失败: {e}")
        return jsonify({
            'success': False,
            'message': f'查看图表失败: {str(e)}'
        }), 500


@app.route('/api/export/data/<record_id>')
@login_required
def export_data(record_id):
    """导出数据API"""
    try:
        if not data_storage:
            return jsonify({'success': False, 'message': '数据存储未初始化'}), 500
        
        # 加载数据
        data = data_storage.load_dataframe(f"data_{record_id}.parquet")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f'climate_data_{record_id}.csv',
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"导出数据失败: {e}")
        return jsonify({
            'success': False,
            'message': f'导出数据失败: {str(e)}'
        }), 500


@app.route('/api/export/model/<model_id>')
@login_required
def export_model(model_id):
    """导出模型API"""
    try:
        if not model_manager:
            return jsonify({'success': False, 'message': '模型管理器未初始化'}), 500
        
        # 获取模型信息
        model_info = model_manager.get_model_info(model_id)
        if not model_info:
            return jsonify({'success': False, 'message': '模型不存在'}), 404
        
        # 创建模型导出包
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            with zipfile.ZipFile(f.name, 'w') as zf:
                # 添加模型文件
                model_path = Path(model_info['file_path'])
                if model_path.exists():
                    zf.write(model_path, model_path.name)
                
                # 添加模型信息
                info_json = json.dumps(model_info, indent=2, default=str)
                zf.writestr('model_info.json', info_json)
            
            temp_path = f.name
        
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f'model_{model_id}.zip',
            mimetype='application/zip'
        )
        
    except Exception as e:
        logger.error(f"导出模型失败: {e}")
        return jsonify({
            'success': False,
            'message': f'导出模型失败: {str(e)}'
        }), 500


@app.route('/api/system/status')
@login_required
def system_status():
    """系统状态API"""
    try:
        status = {
            'data_collector': data_collector is not None,
            'data_storage': data_storage is not None,
            'data_processor': data_processor is not None,
            'model_manager': model_manager is not None,
            'prediction_engine': prediction_engine is not None,
            'chart_generator': chart_generator is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        # 获取详细统计
        if data_storage:
            storage_stats = data_storage.get_storage_stats()
            status['storage_stats'] = storage_stats
        
        if chart_generator:
            chart_stats = chart_generator.get_chart_statistics()
            status['chart_stats'] = chart_stats
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        return jsonify({
            'success': False,
            'message': f'获取系统状态失败: {str(e)}'
        }), 500


@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return render_template('error.html', error_code=404, error_message='页面未找到'), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return render_template('error.html', error_code=500, error_message='服务器内部错误'), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)