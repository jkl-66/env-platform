"""
将GRIB数据导入系统并注册到数据库的脚本。

该脚本提供了一个命令行工具，用于将单个GRIB文件的数据和元数据导入到平台的数据存储系统中。
它会从GRIB文件中提取元数据，并将其保存在PostgreSQL数据库的 `climate_data_records` 表中，
同时将原始文件复制到数据存储的 `raw` 目录中以便将来进行处理。

使用方法:
    python scripts/import_grib_data.py /path/to/your/data.grib

"""

import asyncio
import argparse
import sys
from pathlib import Path
import uuid
import shutil

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.grib_processor import GRIBProcessor
from src.data_processing.data_storage import DataStorage, ClimateDataRecord
from src.utils.logger import setup_logger, get_logger
from src.utils.config import get_settings

async def import_grib_data(file_path: Path):
    """
    处理单个GRIB文件并将其元数据导入数据库。

    Args:
        file_path (Path): 要导入的GRIB文件的路径。
    """
    logger = get_logger(__name__)
    logger.info(f"开始处理GRIB文件: {file_path}")

    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        return

    grib_processor = GRIBProcessor()
    data_storage = DataStorage()
    await data_storage.initialize()

    try:
        # 1. 获取GRIB文件信息
        logger.info("正在提取GRIB元数据...")
        grib_info = grib_processor.get_grib_info(file_path)

        # 2. 将原始文件复制到数据存储区
        raw_data_path = data_storage.raw_data_path / file_path.name
        if not raw_data_path.exists() or not file_path.samefile(raw_data_path):
            shutil.copy(file_path, raw_data_path)
            logger.info(f"原始文件已复制到: {raw_data_path}")
        else:
            logger.info(f"文件已存在于目标位置: {raw_data_path}，跳过复制。")

        # 3. 创建数据库记录
        logger.info("正在创建数据库记录...")
        record = ClimateDataRecord(
            id=uuid.uuid4(),
            source=grib_info.get('attributes', {}).get('GRIB_centreDescription', 'Unknown'),
            data_type='GRIB',
            location='Global',  # 可以根据需要从元数据中提取更具体的位置
            latitude=grib_info.get('spatial_extent', {}).get('latitude', {}).get('min'),
            longitude=grib_info.get('spatial_extent', {}).get('longitude', {}).get('min'),
            start_time=grib_info.get('time_range', {}).get('start'),
            end_time=grib_info.get('time_range', {}).get('end'),
            file_path=str(raw_data_path),
            file_format=file_path.suffix.lower(),
            file_size=grib_info.get('file_size'),
            variables=grib_info.get('variables'),
            data_metadata=grib_info.get('attributes'),
            quality_score=1.0  # 初始质量评分，可以后续进行评估
        )

        # 4. 保存记录到数据库
        record_id = await data_storage.save_data_record(
            source=grib_info.get('attributes', {}).get('GRIB_centreDescription', 'Unknown'),
            data_type='GRIB',
            location='Global',
            start_time=grib_info.get('time_range', {}).get('start'),
            end_time=grib_info.get('time_range', {}).get('end'),
            file_path=str(raw_data_path),
            file_format=file_path.suffix.lower(),
            file_size=grib_info.get('file_size'),
            variables=grib_info.get('variables'),
            data_metadata=grib_info.get('attributes')
        )
        if record_id:
            logger.info(f"成功将数据记录 {record_id} 保存到数据库。")
        else:
            logger.error("保存数据记录失败，未返回记录ID。")

    except Exception as e:
        logger.error(f"处理文件 {file_path} 时发生错误: {e}", exc_info=True)
    finally:
        await data_storage.close()
        logger.info("数据存储连接已关闭。")

def main():
    """主函数，用于解析命令行参数并启动导入过程。"""
    setup_logger()
    parser = argparse.ArgumentParser(description="GRIB数据导入工具")
    parser.add_argument("file_path", type=str, help="要导入的GRIB文件的路径")
    args = parser.parse_args()

    file_to_import = Path(args.file_path)
    
    asyncio.run(import_grib_data(file_to_import))

if __name__ == "__main__":
    main()