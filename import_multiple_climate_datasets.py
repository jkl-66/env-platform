#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多路径全球极端气候事件数据集导入脚本

该脚本用于导入多个路径下的全球0.5°逐年复合极端气候事件数据集（1951-2022年），
包括hot-dry、hot-dry-windy、hot-wet、wet-windy等不同类型的极端气候事件数据。

使用方法:
    python import_multiple_climate_datasets.py
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 尝试导入项目模块，如果失败则使用简化版本
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

class MultipleClimateDatasetImporter:
    """多路径全球极端气候事件数据集导入器"""
    
    def __init__(self, dataset_paths: List[str]):
        self.dataset_paths = [Path(path) for path in dataset_paths]
        self.imported_records = []
        
        if USE_PROJECT_MODULES:
            self.data_storage = DataStorage()
        else:
            self.data_storage = None
            # 创建输出目录
            self.output_dir = Path('outputs') / 'climate_data_import'
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """初始化数据存储"""
        if USE_PROJECT_MODULES and self.data_storage:
            await self.data_storage.initialize()
            logger.info("数据存储初始化完成")
        else:
            logger.info("使用简化模式，跳过数据存储初始化")
    
    def identify_dataset_type(self, path: Path) -> str:
        """根据路径识别数据集类型"""
        path_str = str(path).lower()
        
        if 'hot-dry-windy' in path_str:
            return 'hot-dry-windy'
        elif 'hot-dry' in path_str:
            return 'hot-dry'
        elif 'hot-wet' in path_str:
            return 'hot-wet'
        elif 'wet-windy' in path_str:
            return 'wet-windy'
        else:
            return 'unknown'
    
    def load_dataset_from_path(self, dataset_path: Path) -> pd.DataFrame:
        """从指定路径加载数据集"""
        logger.info(f"正在加载数据集: {dataset_path}")
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
        
        # 如果是目录，查找数据文件
        if dataset_path.is_dir():
            # 支持多种文件格式
            extensions = ['*.csv', '*.xlsx', '*.xls', '*.nc', '*.txt', '*.dat', '*.json', '*.tif', '*.tiff', '*.hdf', '*.h5']
            data_files = []
            
            try:
                # 列出目录中的所有文件以便调试
                all_files = list(dataset_path.iterdir())
                logger.info(f"目录 {dataset_path.name} 中包含 {len(all_files)} 个项目")
                
                # 显示前10个文件/目录的信息
                for i, item in enumerate(all_files[:10]):
                    if item.is_file():
                        logger.info(f"  文件: {item.name} (大小: {item.stat().st_size} 字节, 扩展名: {item.suffix})")
                    else:
                        logger.info(f"  目录: {item.name}/")
                
                if len(all_files) > 10:
                    logger.info(f"  ... 还有 {len(all_files) - 10} 个项目")
                
                # 搜索支持的文件格式
                for ext in extensions:
                    found_files = list(dataset_path.glob(ext))
                    if found_files:
                        logger.info(f"找到 {len(found_files)} 个 {ext} 文件")
                        data_files.extend(found_files)
                    
                    # 也搜索子目录
                    found_files_recursive = list(dataset_path.rglob(ext))
                    if found_files_recursive:
                        logger.info(f"在子目录中找到 {len(found_files_recursive)} 个 {ext} 文件")
                        data_files.extend(found_files_recursive)
                
                # 去重
                data_files = list(set(data_files))
                logger.info(f"总共找到 {len(data_files)} 个支持的数据文件")
                
            except Exception as e:
                logger.error(f"搜索数据文件时发生错误: {e}")
                return pd.DataFrame()
            
            if not data_files:
                logger.warning(f"在目录 {dataset_path} 中未找到支持的数据文件")
                logger.info(f"支持的文件格式: {', '.join(extensions)}")
                return pd.DataFrame()  # 返回空DataFrame
            
            # 如果有多个文件，尝试合并
            if len(data_files) == 1:
                file_path = data_files[0]
            else:
                logger.info(f"找到 {len(data_files)} 个数据文件，将尝试合并")
                # 加载所有文件并合并
                dfs = []
                for file_path in data_files[:10]:  # 限制最多处理10个文件
                    try:
                        df = self._load_single_file(file_path)
                        if not df.empty:
                            df['source_file'] = file_path.name
                            dfs.append(df)
                    except Exception as e:
                        logger.warning(f"无法加载文件 {file_path}: {e}")
                
                if dfs:
                    data = pd.concat(dfs, ignore_index=True)
                    logger.info(f"成功合并 {len(dfs)} 个文件，总数据形状: {data.shape}")
                    return data
                else:
                    logger.warning(f"无法从目录 {dataset_path} 加载任何数据")
                    return pd.DataFrame()
        else:
            file_path = dataset_path
        
        # 加载单个文件
        return self._load_single_file(file_path)
    
    def _load_single_file(self, file_path: Path) -> pd.DataFrame:
        """加载单个数据文件"""
        try:
            # 根据文件扩展名选择加载方式
            if file_path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.nc':
                try:
                    import xarray as xr
                    ds = xr.open_dataset(file_path)
                    data = ds.to_dataframe().reset_index()
                except ImportError:
                    logger.warning("需要安装xarray库来处理NetCDF文件")
                    return pd.DataFrame()
            elif file_path.suffix.lower() in ['.tif', '.tiff']:
                try:
                    # 处理GeoTIFF文件
                    import rasterio
                    import rasterio.features
                    import rasterio.warp
                    
                    with rasterio.open(file_path) as src:
                        # 读取数据
                        data_array = src.read(1)  # 读取第一个波段
                        
                        # 获取地理坐标
                        height, width = data_array.shape
                        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                        
                        # 转换像素坐标到地理坐标
                        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                        
                        # 创建DataFrame
                        data = pd.DataFrame({
                            'longitude': np.array(xs).flatten(),
                            'latitude': np.array(ys).flatten(),
                            'value': data_array.flatten()
                        })
                        
                        # 移除无效值
                        if hasattr(src, 'nodata') and src.nodata is not None:
                            data = data[data['value'] != src.nodata]
                        
                        # 添加文件信息
                        data['file_name'] = file_path.stem
                        
                        logger.info(f"GeoTIFF文件加载完成: {file_path.name}, 形状: {data.shape}")
                        logger.info(f"坐标范围: 经度 [{data['longitude'].min():.3f}, {data['longitude'].max():.3f}], 纬度 [{data['latitude'].min():.3f}, {data['latitude'].max():.3f}]")
                        logger.info(f"数值范围: [{data['value'].min():.3f}, {data['value'].max():.3f}]")
                        
                except ImportError:
                    logger.warning("需要安装rasterio库来处理GeoTIFF文件")
                    logger.info("请运行: pip install rasterio")
                    return pd.DataFrame()
            elif file_path.suffix.lower() in ['.hdf', '.h5']:
                try:
                    import h5py
                    # 这里需要根据具体的HDF5文件结构来处理
                    # 暂时返回空DataFrame，需要用户提供具体的数据结构信息
                    logger.warning(f"HDF5文件需要特定的处理逻辑: {file_path.name}")
                    return pd.DataFrame()
                except ImportError:
                    logger.warning("需要安装h5py库来处理HDF5文件")
                    return pd.DataFrame()
            elif file_path.suffix.lower() in ['.txt', '.dat']:
                # 尝试以制表符或逗号分隔的文本文件
                try:
                    data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
                except:
                    data = pd.read_csv(file_path, sep=',', encoding='utf-8')
            elif file_path.suffix.lower() == '.json':
                data = pd.read_json(file_path)
            else:
                logger.warning(f"不支持的文件格式: {file_path.suffix}")
                return pd.DataFrame()
            
            logger.info(f"文件加载完成: {file_path.name}, 形状: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"加载文件 {file_path} 时发生错误: {e}")
            return pd.DataFrame()
    
    async def import_single_dataset(self, dataset_path: Path) -> str:
        """导入单个数据集到数据库"""
        dataset_type = self.identify_dataset_type(dataset_path)
        logger.info(f"正在导入数据集类型: {dataset_type}")
        
        # 加载数据
        data = self.load_dataset_from_path(dataset_path)
        
        if data.empty:
            logger.warning(f"数据集为空，跳过导入: {dataset_path}")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if USE_PROJECT_MODULES and self.data_storage:
            # 使用完整的数据存储系统
            filename = f"climate_extreme_events_{dataset_type}_{timestamp}"
            
            # 保存为parquet格式以提高性能
            file_path = self.data_storage.save_dataframe(
                data, 
                filename, 
                data_category="processed", 
                format="parquet"
            )
            
            # 创建数据记录
            record_id = await self.data_storage.save_data_record(
                source=f"Global Climate Dataset - {dataset_type}",
                data_type="extreme_events",
                location="Global",
                start_time=datetime(1951, 1, 1),
                end_time=datetime(2022, 12, 31),
                file_path=file_path,
                file_format="parquet",
                file_size=os.path.getsize(file_path),
                variables=list(data.columns),
                data_metadata={
                    "description": f"全球0.5°逐年复合极端气候事件数据集（1951-2022年）- {dataset_type}",
                    "resolution": "0.5度",
                    "temporal_coverage": "1951-2022",
                    "data_type": "extreme_climate_events",
                    "event_type": dataset_type,
                    "original_path": str(dataset_path)
                }
            )
            
            logger.info(f"数据集已导入数据库，记录ID: {record_id}")
            return record_id
        else:
            # 简化模式：直接保存到本地文件
            filename = f"climate_extreme_events_{dataset_type}_{timestamp}.parquet"
            file_path = self.output_dir / filename
            
            data.to_parquet(file_path)
            
            # 保存元数据
            metadata = {
                "source": f"Global Climate Dataset - {dataset_type}",
                "data_type": "extreme_events",
                "location": "Global",
                "start_time": "1951-01-01",
                "end_time": "2022-12-31",
                "file_path": str(file_path),
                "file_format": "parquet",
                "file_size": os.path.getsize(file_path),
                "variables": list(data.columns),
                "description": f"全球0.5°逐年复合极端气候事件数据集（1951-2022年）- {dataset_type}",
                "resolution": "0.5度",
                "temporal_coverage": "1951-2022",
                "event_type": dataset_type,
                "original_path": str(dataset_path),
                "data_shape": data.shape
            }
            
            metadata_path = self.output_dir / f"metadata_{dataset_type}_{timestamp}.json"
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"数据集已保存到本地文件: {file_path}")
            logger.info(f"元数据已保存: {metadata_path}")
            return f"local_file_{dataset_type}_{timestamp}"
    
    async def import_all_datasets(self) -> List[str]:
        """导入所有数据集"""
        logger.info(f"开始导入 {len(self.dataset_paths)} 个数据集...")
        
        imported_records = []
        
        for i, dataset_path in enumerate(self.dataset_paths, 1):
            logger.info(f"\n处理数据集 {i}/{len(self.dataset_paths)}: {dataset_path}")
            
            try:
                record_id = await self.import_single_dataset(dataset_path)
                if record_id:
                    imported_records.append({
                        'path': str(dataset_path),
                        'type': self.identify_dataset_type(dataset_path),
                        'record_id': record_id,
                        'status': 'success'
                    })
                else:
                    imported_records.append({
                        'path': str(dataset_path),
                        'type': self.identify_dataset_type(dataset_path),
                        'record_id': None,
                        'status': 'skipped'
                    })
            except Exception as e:
                logger.error(f"导入数据集失败 {dataset_path}: {e}")
                imported_records.append({
                    'path': str(dataset_path),
                    'type': self.identify_dataset_type(dataset_path),
                    'record_id': None,
                    'status': 'failed',
                    'error': str(e)
                })
        
        self.imported_records = imported_records
        return imported_records
    
    def print_import_summary(self):
        """打印导入摘要"""
        print("\n" + "="*80)
        print("多路径气候数据集导入结果摘要")
        print("="*80)
        
        success_count = sum(1 for record in self.imported_records if record['status'] == 'success')
        skipped_count = sum(1 for record in self.imported_records if record['status'] == 'skipped')
        failed_count = sum(1 for record in self.imported_records if record['status'] == 'failed')
        
        print(f"\n📊 导入统计:")
        print(f"  总数据集数量: {len(self.imported_records)}")
        print(f"  成功导入: {success_count}")
        print(f"  跳过: {skipped_count}")
        print(f"  失败: {failed_count}")
        
        print(f"\n📋 详细结果:")
        for record in self.imported_records:
            status_icon = "✅" if record['status'] == 'success' else "⚠️" if record['status'] == 'skipped' else "❌"
            print(f"  {status_icon} {record['type']}: {record['status']}")
            if record['record_id']:
                print(f"      记录ID: {record['record_id']}")
            if record.get('error'):
                print(f"      错误: {record['error']}")
        
        print("="*80)
    
    async def run_import(self):
        """运行完整导入流程"""
        try:
            # 初始化
            await self.initialize()
            
            # 导入所有数据集
            imported_records = await self.import_all_datasets()
            
            # 打印摘要
            self.print_import_summary()
            
            return imported_records
            
        except Exception as e:
            logger.error(f"导入过程中发生错误: {e}", exc_info=True)
            raise
        finally:
            if USE_PROJECT_MODULES and self.data_storage:
                await self.data_storage.close()

async def main():
    """主函数"""
    # 用户指定的4个数据集路径
    dataset_paths = [
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-dry_[tasmax_p_95_all_7_1]-[pr_p_5_+0_0_1]_1_1~12",
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-dry-windy_[tasmax_p_95_all_7_1]-[pr_p_5_+0_0_1]-[sfcwind_p_95_+0.5_0_1]_1_1~12",
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-wet_[tasmax_p_95_all_7_1]-[pr_p_95_+1_0_1]_1_1~12",
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_wet-windy_[pr_p_95_+1_0_1]-[sfcwind_p_95_+0.5_0_1]_1_1~12"
    ]
    
    print("🚀 开始多路径全球极端气候事件数据集导入...")
    print(f"📂 将处理 {len(dataset_paths)} 个数据集路径")
    
    # 检查路径是否存在
    existing_paths = []
    for path in dataset_paths:
        if Path(path).exists():
            existing_paths.append(path)
            print(f"✅ 路径存在: {Path(path).name}")
        else:
            print(f"❌ 路径不存在: {path}")
    
    if not existing_paths:
        print("\n❌ 错误: 所有指定的数据集路径都不存在")
        print("请检查路径是否正确，或将数据集文件放置在指定位置。")
        return
    
    if len(existing_paths) < len(dataset_paths):
        print(f"\n⚠️ 警告: 只有 {len(existing_paths)}/{len(dataset_paths)} 个路径存在，将只处理存在的路径")
    
    # 创建导入器并运行导入
    importer = MultipleClimateDatasetImporter(existing_paths)
    
    try:
        results = await importer.run_import()
        
        print("\n✅ 导入完成!")
        
        # 保存导入结果
        import json
        results_file = Path('outputs') / 'climate_data_import' / f'import_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"📋 导入结果已保存: {results_file}")
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        logger.error(f"导入失败: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())