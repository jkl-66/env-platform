#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量导入全球极端气候事件数据集
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
import json
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 尝试导入项目模块
try:
    from src.data_processing.data_storage import DataStorage
    from src.utils.logger import get_logger
    from src.utils.config import get_settings
    USE_PROJECT_MODULES = True
except ImportError as e:
    print(f"警告: 无法导入项目模块 ({e})，将使用简化版本")
    USE_PROJECT_MODULES = False
    
    # 简化的日志记录器
    import logging
    def get_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    # 简化的设置
    class SimpleSettings:
        def __init__(self):
            self.database_url = "sqlite:///climate_data.db"
    
    def get_settings():
        return SimpleSettings()

logger = get_logger(__name__)
settings = get_settings()

class ClimateDataBatchImporter:
    """气候数据批量导入器"""
    
    def __init__(self):
        if USE_PROJECT_MODULES:
            self.data_storage = DataStorage()
        else:
            self.data_storage = None
            # 创建输出目录
            self.output_dir = Path('outputs') / 'climate_batch_import'
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建本地数据目录
            self.local_data_dir = Path('data') / 'climate_netcdf'
            self.local_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.import_results = {
            'total_datasets': 0,
            'successful_imports': 0,
            'failed_imports': 0,
            'skipped_imports': 0,
            'details': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
    
    async def initialize(self):
        """初始化数据存储"""
        if USE_PROJECT_MODULES and self.data_storage:
            await self.data_storage.initialize()
            logger.info("数据存储初始化完成")
        else:
            logger.info("使用简化模式，跳过数据存储初始化")
    
    def find_netcdf_files(self, dataset_path: str) -> List[str]:
        """查找数据集目录中的NetCDF文件"""
        logger.info(f"搜索NetCDF文件: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            logger.warning(f"数据集路径不存在: {dataset_path}")
            return []
        
        if not os.path.isdir(dataset_path):
            logger.warning(f"路径不是目录: {dataset_path}")
            return []
        
        netcdf_files = []
        try:
            for item in os.listdir(dataset_path):
                if item.endswith('.nc'):
                    file_path = os.path.join(dataset_path, item)
                    if os.path.isfile(file_path):
                        netcdf_files.append(file_path)
            
            logger.info(f"找到 {len(netcdf_files)} 个NetCDF文件")
            return netcdf_files
            
        except Exception as e:
            logger.error(f"搜索NetCDF文件时发生错误: {e}")
            return []
    
    def copy_file_to_local(self, source_path: str) -> str:
        """复制文件到本地目录"""
        try:
            source_file = Path(source_path)
            local_file = self.local_data_dir / source_file.name
            
            # 如果本地文件已存在且大小相同，跳过复制
            if local_file.exists():
                if local_file.stat().st_size == source_file.stat().st_size:
                    logger.info(f"本地文件已存在，跳过复制: {local_file.name}")
                    return str(local_file)
            
            # 复制文件
            shutil.copy2(source_path, local_file)
            logger.info(f"文件复制成功: {local_file.name}")
            return str(local_file)
            
        except Exception as e:
            logger.error(f"文件复制失败: {e}")
            return None
    
    def load_netcdf_file(self, file_path: str) -> pd.DataFrame:
        """加载NetCDF文件"""
        logger.info(f"正在加载NetCDF文件: {os.path.basename(file_path)}")
        
        try:
            import xarray as xr
            
            # 打开NetCDF文件
            ds = xr.open_dataset(file_path)
            
            logger.info(f"NetCDF文件信息:")
            logger.info(f"  维度: {dict(ds.dims)}")
            logger.info(f"  变量: {list(ds.data_vars)}")
            logger.info(f"  坐标: {list(ds.coords)}")
            
            # 转换为DataFrame
            df = ds.to_dataframe().reset_index()
            
            # 移除NaN值
            df = df.dropna()
            
            # 添加文件信息
            df['source_file'] = os.path.basename(file_path)
            df['dataset_type'] = self._extract_dataset_type(file_path)
            df['year'] = self._extract_year(file_path)
            
            ds.close()
            
            logger.info(f"NetCDF文件加载完成: {os.path.basename(file_path)}")
            logger.info(f"数据形状: {df.shape}")
            
            return df
            
        except ImportError:
            logger.error("需要安装xarray库来处理NetCDF文件")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"加载NetCDF文件时发生错误: {e}")
            return pd.DataFrame()
    
    def _extract_dataset_type(self, filename: str) -> str:
        """从文件名提取数据集类型"""
        if 'hot-dry-windy' in filename:
            return 'hot-dry-windy'
        elif 'hot-dry' in filename:
            return 'hot-dry'
        elif 'hot-wet' in filename:
            return 'hot-wet'
        elif 'wet-windy' in filename:
            return 'wet-windy'
        else:
            return 'unknown'
    
    def _extract_year(self, filename: str) -> int:
        """从文件名提取年份"""
        import re
        match = re.search(r'_(\d{4})_', filename)
        if match:
            return int(match.group(1))
        return None
    
    async def import_dataset(self, dataset_path: str, dataset_name: str) -> Dict[str, Any]:
        """导入单个数据集"""
        logger.info(f"开始导入数据集: {dataset_name}")
        
        result = {
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'status': 'failed',
            'files_processed': 0,
            'total_records': 0,
            'error_message': None,
            'record_ids': [],
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        try:
            # 查找NetCDF文件
            netcdf_files = self.find_netcdf_files(dataset_path)
            
            if not netcdf_files:
                result['status'] = 'skipped'
                result['error_message'] = '未找到NetCDF文件'
                logger.warning(f"数据集 {dataset_name} 中未找到NetCDF文件")
                return result
            
            # 限制处理文件数量（测试时只处理前5个文件）
            max_files = 5
            if len(netcdf_files) > max_files:
                logger.info(f"数据集包含 {len(netcdf_files)} 个文件，限制处理前 {max_files} 个")
                netcdf_files = netcdf_files[:max_files]
            
            all_data = []
            processed_files = 0
            
            for file_path in netcdf_files:
                try:
                    # 复制文件到本地
                    local_file = self.copy_file_to_local(file_path)
                    if not local_file:
                        continue
                    
                    # 加载NetCDF文件
                    data = self.load_netcdf_file(local_file)
                    
                    if not data.empty:
                        all_data.append(data)
                        processed_files += 1
                        logger.info(f"成功处理文件: {os.path.basename(file_path)} ({data.shape[0]} 条记录)")
                    else:
                        logger.warning(f"文件数据为空: {os.path.basename(file_path)}")
                        
                except Exception as e:
                    logger.error(f"处理文件 {os.path.basename(file_path)} 时发生错误: {e}")
                    continue
            
            if not all_data:
                result['status'] = 'skipped'
                result['error_message'] = '所有文件处理失败或数据为空'
                logger.warning(f"数据集 {dataset_name} 中所有文件处理失败")
                return result
            
            # 合并所有数据
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"数据集 {dataset_name} 合并完成，总记录数: {combined_data.shape[0]}")
            
            # 保存数据
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if USE_PROJECT_MODULES and self.data_storage:
                # 使用完整的数据存储系统
                filename = f"climate_batch_{dataset_name}_{timestamp}"
                
                # 保存为parquet格式
                saved_file_path = self.data_storage.save_dataframe(
                    combined_data, 
                    filename, 
                    data_category="processed", 
                    format="parquet"
                )
                
                # 创建数据记录
                record_id = await self.data_storage.save_data_record(
                    source=f"Global Climate NetCDF Batch - {dataset_name}",
                    data_type="extreme_events",
                    location="Global",
                    start_time=datetime(1951, 1, 1),
                    end_time=datetime(2022, 12, 31),
                    file_path=saved_file_path,
                    file_format="parquet",
                    file_size=os.path.getsize(saved_file_path),
                    variables=list(combined_data.columns),
                    data_metadata={
                        "description": f"全球0.5°极端气候事件数据集 - {dataset_name}",
                        "resolution": "0.5度",
                        "time_range": "1951-2022",
                        "data_type": "extreme_climate_events",
                        "event_type": dataset_name,
                        "files_processed": processed_files,
                        "total_files": len(netcdf_files),
                        "data_shape": list(combined_data.shape)
                    }
                )
                
                result['record_ids'].append(record_id)
                logger.info(f"数据集 {dataset_name} 已导入数据库，记录ID: {record_id}")
            else:
                # 简化模式：直接保存到本地文件
                filename = f"climate_batch_{dataset_name}_{timestamp}.parquet"
                saved_file_path = self.output_dir / filename
                
                combined_data.to_parquet(saved_file_path)
                
                # 保存元数据
                metadata = {
                    "source": f"Global Climate NetCDF Batch - {dataset_name}",
                    "data_type": "extreme_events",
                    "location": "Global",
                    "time_range": "1951-2022",
                    "file_path": str(saved_file_path),
                    "file_format": "parquet",
                    "file_size": os.path.getsize(saved_file_path),
                    "variables": list(combined_data.columns),
                    "description": f"全球0.5°极端气候事件数据集 - {dataset_name}",
                    "resolution": "0.5度",
                    "event_type": dataset_name,
                    "files_processed": processed_files,
                    "total_files": len(netcdf_files),
                    "data_shape": list(combined_data.shape)
                }
                
                metadata_path = self.output_dir / f"metadata_{dataset_name}_{timestamp}.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                logger.info(f"数据集 {dataset_name} 已保存到本地: {saved_file_path}")
                logger.info(f"元数据已保存: {metadata_path}")
            
            result['status'] = 'success'
            result['files_processed'] = processed_files
            result['total_records'] = combined_data.shape[0]
            
        except Exception as e:
            result['error_message'] = str(e)
            logger.error(f"导入数据集 {dataset_name} 时发生错误: {e}")
        
        finally:
            end_time = datetime.now()
            result['processing_time'] = (end_time - start_time).total_seconds()
        
        return result
    
    async def import_all_datasets(self, dataset_paths: Dict[str, str]):
        """导入所有数据集"""
        logger.info(f"开始批量导入 {len(dataset_paths)} 个数据集")
        
        self.import_results['total_datasets'] = len(dataset_paths)
        
        for dataset_name, dataset_path in dataset_paths.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"处理数据集: {dataset_name}")
            
            result = await self.import_dataset(dataset_path, dataset_name)
            self.import_results['details'].append(result)
            
            if result['status'] == 'success':
                self.import_results['successful_imports'] += 1
                logger.info(f"✅ 数据集 {dataset_name} 导入成功")
            elif result['status'] == 'skipped':
                self.import_results['skipped_imports'] += 1
                logger.warning(f"⚠️ 数据集 {dataset_name} 被跳过: {result['error_message']}")
            else:
                self.import_results['failed_imports'] += 1
                logger.error(f"❌ 数据集 {dataset_name} 导入失败: {result['error_message']}")
        
        self.import_results['end_time'] = datetime.now().isoformat()
        
        # 保存导入结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"batch_import_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.import_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n{'='*80}")
        logger.info("批量导入结果摘要")
        logger.info(f"{'='*80}")
        logger.info(f"📊 导入统计:")
        logger.info(f"  总数据集数量: {self.import_results['total_datasets']}")
        logger.info(f"  成功导入: {self.import_results['successful_imports']}")
        logger.info(f"  跳过: {self.import_results['skipped_imports']}")
        logger.info(f"  失败: {self.import_results['failed_imports']}")
        logger.info(f"")
        logger.info(f"📋 详细结果:")
        for detail in self.import_results['details']:
            status_icon = "✅" if detail['status'] == 'success' else "⚠️" if detail['status'] == 'skipped' else "❌"
            logger.info(f"  {status_icon} {detail['dataset_name']}: {detail['status']} ({detail['files_processed']} 文件, {detail['total_records']} 记录)")
        logger.info(f"{'='*80}")
        logger.info(f"✅ 批量导入完成!")
        logger.info(f"📋 导入结果已保存: {results_file}")

async def main():
    """主函数"""
    # 数据集路径
    dataset_paths = {
        "hot-dry": r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-dry_[tasmax_p_95_all_7_1]-[pr_p_5_+0_0_1]_1_1~12",
        "hot-dry-windy": r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-dry-windy_[tasmax_p_95_all_7_1]-[pr_p_5_+0_0_1]-[sfcwind_p_95_+0.5_0_1]_1_1~12",
        "hot-wet": r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-wet_[tasmax_p_95_all_7_1]-[pr_p_95_+1_0_1]_1_1~12",
        "wet-windy": r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_wet-windy_[pr_p_95_+1_0_1]-[sfcwind_p_95_+0.5_0_1]_1_1~12"
    }
    
    print("🚀 开始批量导入全球极端气候事件数据集...")
    
    # 创建导入器并运行导入
    importer = ClimateDataBatchImporter()
    
    try:
        await importer.initialize()
        await importer.import_all_datasets(dataset_paths)
        
    except Exception as e:
        print(f"❌ 批量导入失败: {e}")
        logger.error(f"批量导入失败: {e}", exc_info=True)
    finally:
        if USE_PROJECT_MODULES and importer.data_storage:
            await importer.data_storage.close()

if __name__ == "__main__":
    asyncio.run(main())