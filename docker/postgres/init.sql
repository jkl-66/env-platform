-- 气候数据分析系统数据库初始化脚本
-- 创建时间: 2024
-- 描述: 初始化PostgreSQL数据库，创建必要的表和索引

-- 设置时区
SET timezone = 'UTC';

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

-- 数据源表
CREATE TABLE IF NOT EXISTS data_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL, -- 'noaa', 'ecmwf', 'satellite', 'station'
    description TEXT,
    api_endpoint VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 气象站表
CREATE TABLE IF NOT EXISTS weather_stations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    station_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    country VARCHAR(100),
    state VARCHAR(100),
    latitude DECIMAL(10, 7) NOT NULL,
    longitude DECIMAL(10, 7) NOT NULL,
    elevation DECIMAL(8, 2),
    data_source_id UUID REFERENCES data_sources(id),
    is_active BOOLEAN DEFAULT true,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 数据集表
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    data_source_id UUID REFERENCES data_sources(id),
    dataset_type VARCHAR(50) NOT NULL, -- 'timeseries', 'gridded', 'satellite'
    variables TEXT[], -- 存储变量名数组
    temporal_resolution VARCHAR(20), -- 'hourly', 'daily', 'monthly', 'yearly'
    spatial_resolution VARCHAR(50),
    start_date DATE,
    end_date DATE,
    file_format VARCHAR(20), -- 'netcdf', 'csv', 'json', 'grib'
    file_path TEXT,
    file_size BIGINT,
    checksum VARCHAR(64),
    metadata JSONB,
    is_processed BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 模型表
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'lstm', 'prophet', 'diffusion', 'gan'
    description TEXT,
    version VARCHAR(20) DEFAULT '1.0.0',
    framework VARCHAR(50), -- 'tensorflow', 'pytorch', 'sklearn'
    model_path TEXT,
    config JSONB,
    training_dataset_id UUID REFERENCES datasets(id),
    performance_metrics JSONB,
    is_trained BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 预测任务表
CREATE TABLE IF NOT EXISTS prediction_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    model_id UUID REFERENCES models(id),
    input_dataset_id UUID REFERENCES datasets(id),
    output_dataset_id UUID REFERENCES datasets(id),
    task_type VARCHAR(50) NOT NULL, -- 'forecast', 'anomaly_detection', 'classification'
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    progress INTEGER DEFAULT 0,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    config JSONB,
    results JSONB,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 系统日志表
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    level VARCHAR(20) NOT NULL, -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    message TEXT NOT NULL,
    module VARCHAR(100),
    function_name VARCHAR(100),
    user_id UUID REFERENCES users(id),
    request_id VARCHAR(100),
    extra_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 数据质量检查表
CREATE TABLE IF NOT EXISTS data_quality_checks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID REFERENCES datasets(id),
    check_type VARCHAR(50) NOT NULL, -- 'completeness', 'accuracy', 'consistency', 'validity'
    check_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'passed', 'failed', 'warning'
    score DECIMAL(5, 4), -- 0.0000 to 1.0000
    details JSONB,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
-- 用户表索引
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- 数据源表索引
CREATE INDEX IF NOT EXISTS idx_data_sources_type ON data_sources(type);
CREATE INDEX IF NOT EXISTS idx_data_sources_name ON data_sources(name);

-- 气象站表索引
CREATE INDEX IF NOT EXISTS idx_weather_stations_station_id ON weather_stations(station_id);
CREATE INDEX IF NOT EXISTS idx_weather_stations_location ON weather_stations(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_weather_stations_country ON weather_stations(country);
CREATE INDEX IF NOT EXISTS idx_weather_stations_data_source ON weather_stations(data_source_id);

-- 数据集表索引
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);
CREATE INDEX IF NOT EXISTS idx_datasets_type ON datasets(dataset_type);
CREATE INDEX IF NOT EXISTS idx_datasets_source ON datasets(data_source_id);
CREATE INDEX IF NOT EXISTS idx_datasets_dates ON datasets(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at);

-- 模型表索引
CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type);
CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
CREATE INDEX IF NOT EXISTS idx_models_created_by ON models(created_by);
CREATE INDEX IF NOT EXISTS idx_models_is_active ON models(is_active);

-- 预测任务表索引
CREATE INDEX IF NOT EXISTS idx_prediction_tasks_status ON prediction_tasks(status);
CREATE INDEX IF NOT EXISTS idx_prediction_tasks_model ON prediction_tasks(model_id);
CREATE INDEX IF NOT EXISTS idx_prediction_tasks_created_by ON prediction_tasks(created_by);
CREATE INDEX IF NOT EXISTS idx_prediction_tasks_created_at ON prediction_tasks(created_at);

-- 系统日志表索引
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);
CREATE INDEX IF NOT EXISTS idx_system_logs_module ON system_logs(module);
CREATE INDEX IF NOT EXISTS idx_system_logs_created_at ON system_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_system_logs_user_id ON system_logs(user_id);

-- 数据质量检查表索引
CREATE INDEX IF NOT EXISTS idx_data_quality_dataset ON data_quality_checks(dataset_id);
CREATE INDEX IF NOT EXISTS idx_data_quality_type ON data_quality_checks(check_type);
CREATE INDEX IF NOT EXISTS idx_data_quality_status ON data_quality_checks(status);

-- 插入初始数据
-- 插入默认管理员用户
INSERT INTO users (username, email, password_hash, full_name, is_superuser) 
VALUES (
    'admin', 
    'admin@climate-system.com', 
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6hsxq5/Qe2', -- 密码: admin123
    '系统管理员', 
    true
) ON CONFLICT (username) DO NOTHING;

-- 插入数据源
INSERT INTO data_sources (name, type, description, api_endpoint, config) VALUES
('NOAA Climate Data', 'noaa', 'NOAA国家气候数据中心', 'https://www.ncdc.noaa.gov/cdo-web/api/v2/', '{"rate_limit": 1000, "timeout": 30}'),
('ECMWF ERA5', 'ecmwf', 'ECMWF ERA5再分析数据', 'https://cds.climate.copernicus.eu/api/v2/', '{"rate_limit": 100, "timeout": 300}'),
('卫星遥感数据', 'satellite', '多源卫星遥感气候数据', null, '{"sources": ["MODIS", "Landsat", "Sentinel"]}'),
('地面观测站', 'station', '地面气象观测站数据', null, '{"update_frequency": "hourly"}')
ON CONFLICT DO NOTHING;

-- 创建更新时间触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为所有有updated_at字段的表创建触发器
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_data_sources_updated_at BEFORE UPDATE ON data_sources FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_weather_stations_updated_at BEFORE UPDATE ON weather_stations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON datasets FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_prediction_tasks_updated_at BEFORE UPDATE ON prediction_tasks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 创建视图
-- 活跃数据集视图
CREATE OR REPLACE VIEW active_datasets AS
SELECT 
    d.*,
    ds.name as source_name,
    ds.type as source_type
FROM datasets d
JOIN data_sources ds ON d.data_source_id = ds.id
WHERE ds.is_active = true;

-- 模型性能统计视图
CREATE OR REPLACE VIEW model_performance_stats AS
SELECT 
    m.id,
    m.name,
    m.model_type,
    m.is_trained,
    COUNT(pt.id) as total_predictions,
    COUNT(CASE WHEN pt.status = 'completed' THEN 1 END) as successful_predictions,
    COUNT(CASE WHEN pt.status = 'failed' THEN 1 END) as failed_predictions,
    AVG(EXTRACT(EPOCH FROM (pt.end_time - pt.start_time))/60) as avg_runtime_minutes
FROM models m
LEFT JOIN prediction_tasks pt ON m.id = pt.model_id
GROUP BY m.id, m.name, m.model_type, m.is_trained;

-- 数据质量汇总视图
CREATE OR REPLACE VIEW data_quality_summary AS
SELECT 
    d.id as dataset_id,
    d.name as dataset_name,
    COUNT(dqc.id) as total_checks,
    COUNT(CASE WHEN dqc.status = 'passed' THEN 1 END) as passed_checks,
    COUNT(CASE WHEN dqc.status = 'failed' THEN 1 END) as failed_checks,
    COUNT(CASE WHEN dqc.status = 'warning' THEN 1 END) as warning_checks,
    ROUND(AVG(dqc.score), 4) as avg_quality_score
FROM datasets d
LEFT JOIN data_quality_checks dqc ON d.id = dqc.dataset_id
GROUP BY d.id, d.name;

-- 授予权限
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- 完成初始化
INSERT INTO system_logs (level, message, module) 
VALUES ('INFO', '数据库初始化完成', 'database_init');