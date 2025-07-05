# DashScope 环境图像生成器 - 专业 Prompt 查看指南

本指南详细介绍如何查看和分析 DashScope 环境图像生成器生成的专业 prompt，包括新增的默认值对比、偏差分析和多用户群体支持功能。

## 📋 目录

1. [新功能概览](#新功能概览)
2. [查看方式概览](#查看方式概览)
3. [实时查看 Prompt](#实时查看-prompt)
4. [历史记录查看](#历史记录查看)
5. [详细分析功能](#详细分析功能)
6. [批量生成对比](#批量生成对比)
7. [保存和导出](#保存和导出)

## 🆕 新功能概览

### 默认值对比系统
- **环境数据默认值**: 每种环境数据都有当前世界的平均值作为参考
- **偏差分析**: 自动计算输入数据与默认值的偏差比例
- **主次控制**: 根据偏差程度自动调整图像元素的主次关系

### 多用户群体支持
- **general**: 普通用户 - 现实主义风格，专业氛围
- **educators**: 教育工作者 - 教育风格，清晰明了
- **parents**: 家长 - 温和风格，关怀氛围
- **students**: 学生 - 卡通风格，生动有趣

### 图像内容限制
- **无人物**: 确保生成的图像不包含任何人类形象
- **无文字**: 避免图像中出现文字、标签或标识
- **环境焦点**: 专注于环境和生态系统的表现

## 🔍 查看方式概览

系统提供了多种方式来查看生成的专业 prompt：

### 方式一：在图像生成过程中查看

当使用 `environmental_warning_demo.py` 生成图像时，系统会自动显示生成的 prompt：

```bash
python environmental_warning_demo.py
```

生成结果中会显示：
```
📝 生成的专业 Prompt (前150字符):
   Environmental warning scene showing severe environmental degradation, industrial emissions and carbon pollution, smoggy air and poor visibility...
```

### 方式二：专用 Prompt 查看器

使用专门的 prompt 查看工具，无需生成图像：

```bash
python view_prompt_demo.py
```

## 🤖 实时查看 Prompt

### 使用专用查看器

运行 `view_prompt_demo.py` 后，选择以下选项：

1. **使用预设数据生成 prompt** - 快速查看不同污染程度的 prompt
2. **自定义环境数据生成 prompt** - 根据你的数据生成定制 prompt
3. **批量生成多个场景的 prompt** - 对比不同场景的 prompt 差异

### 预设场景示例

系统提供了5个预设场景：

- **轻度空气污染**: AQI 120, 碳排放 300 吨
- **中度工业污染**: 碳排放 800 吨, AQI 180, 水污染指数 60
- **严重环境危机**: 多重污染叠加
- **海洋塑料污染**: 塑料废物 2000 吨
- **噪音污染**: 噪音 95 分贝

## 📚 历史记录查看

### 查看已生成的报告

1. 运行 `view_prompt_demo.py`
2. 选择 "3. 查看历史生成报告中的 prompt"
3. 系统会列出最近的10个报告文件
4. 选择要查看的报告

### 报告文件位置

历史报告保存在：
```
outputs/environmental_images/environmental_report_YYYYMMDD_HHMMSS.json
```

### 手动查看报告文件

你也可以直接打开 JSON 报告文件，查找 `professional_prompt` 字段：

```json
{
  "professional_prompt": "Environmental warning scene showing severe environmental degradation...",
  "environmental_data": {
    "carbon_emission": 2200,
    "air_quality_index": 280
  },
  "analysis": {
    "overall_severity": "critical"
  }
}
```

## 📈 详细分析功能

专用查看器提供了详细的 prompt 分析：

### 内容统计
- 字符数
- 单词数
- 句子数

### 关键词分析
- **环境关键词**: pollution, toxic, emissions, smog 等
- **视觉关键词**: dramatic, lighting, photography, realistic 等

### 场景分析
- 环境数据详情
- 严重程度评估
- 关键问题因素
- 目标受众适配

## 🔄 批量生成对比

使用批量生成功能可以对比不同污染程度的 prompt 差异：

1. 运行 `view_prompt_demo.py`
2. 选择 "4. 批量生成多个场景的 prompt"
3. 系统会生成4个不同严重程度的场景
4. 对比 prompt 的变化趋势

## 💾 保存和导出

### 自动保存

每次生成图像时，完整的 prompt 会自动保存在报告文件中。

### 手动保存

在专用查看器中，查看 prompt 后可以选择保存到文件：

```
💾 是否保存此 prompt 到文件？(y/N): y
✅ Prompt 已保存到: outputs/prompts/prompt_场景名称_20240101_120000.txt
```

### 保存格式

保存的文件包含：
- 场景信息
- 生成时间
- 环境数据
- 完整的专业 prompt
- 分析结果

## 🎯 实用技巧

### 1. 快速查看最新 Prompt

```bash
# 运行快速演示，查看生成的 prompt
python environmental_warning_demo.py
# 选择 "2. 快速演示"
```

### 2. 对比不同受众的 Prompt

使用相同的环境数据，但选择不同的目标受众（general, educators, parents, students），观察 prompt 的差异。

### 3. 分析 Prompt 质量

关注以下要素：
- 是否包含具体的环境数据体现
- 视觉描述是否生动具体
- 是否适合目标受众
- 是否包含专业摄影术语

### 4. 自定义数据测试

输入极端数值（如 AQI 500+），观察 prompt 如何体现严重程度。

## ⚠️ 注意事项

1. **API 调用**: 每次生成 prompt 都会调用 Qwen 模型，消耗 API 额度
2. **网络连接**: 需要稳定的网络连接访问 DashScope 服务
3. **API Key**: 确保 `.env` 文件中正确配置了 `DASHSCOPE_API_KEY`
4. **文件权限**: 确保有写入 `outputs` 目录的权限

## 🔧 故障排除

### Prompt 生成失败

如果 Qwen 模型调用失败，系统会使用备用方法生成基础 prompt。检查：
- API Key 是否正确
- 网络连接是否正常
- DashScope 服务是否可用

### 历史记录为空

如果没有历史记录，说明还没有生成过图像。先运行一次完整的图像生成流程。

### 文件保存失败

检查 `outputs` 目录的写入权限，或手动创建目录。

## 📞 获取帮助

如果遇到问题，可以：
1. 查看控制台的错误信息
2. 检查 `logs/app.log` 日志文件
3. 确认环境配置是否正确

---

通过这些方式，你可以全面了解和分析系统生成的专业 prompt，优化环境数据输入，获得更好的图像生成效果。