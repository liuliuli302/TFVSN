#!/bin/bash
# 视频摘要 Pipeline 执行脚本
# filepath: /root/TFVSN/run_pipeline.sh

# 确保脚本在错误时停止
set -e

# 默认参数
CONFIG_PATH="config_pipeline.json"
DATA_DIR="/root/TFVSN/dataset"
OUTPUT_DIR="/root/TFVSN/output"

# 帮助信息
print_help() {
    echo "视频摘要 Pipeline 执行脚本"
    echo
    echo "使用方法:"
    echo "  ./run_pipeline.sh [选项]"
    echo
    echo "选项:"
    echo "  -c, --config FILE    指定配置文件(默认: config_pipeline.json)"
    echo "  -d, --data DIR       指定数据目录(默认: /root/TFVSN/dataset)"
    echo "  -o, --output DIR     指定输出目录(默认: /root/TFVSN/output)"
    echo "  -v, --video FILE     指定视频文件(可选)"
    echo "  --skip-video         跳过视频处理步骤"
    echo "  --skip-samples       跳过样本构建步骤"
    echo "  --skip-llm           跳过LLM分析步骤"
    echo "  --skip-cleaning      跳过数据清理步骤"
    echo "  --skip-eval          跳过评估步骤"
    echo "  -h, --help           显示帮助信息"
    echo
}

# 解析命令行参数
SKIP_FLAGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -d|--data)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -v|--video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --skip-video)
            SKIP_FLAGS="$SKIP_FLAGS --skip_video"
            shift
            ;;
        --skip-samples)
            SKIP_FLAGS="$SKIP_FLAGS --skip_samples"
            shift
            ;;
        --skip-llm)
            SKIP_FLAGS="$SKIP_FLAGS --skip_llm"
            shift
            ;;
        --skip-cleaning)
            SKIP_FLAGS="$SKIP_FLAGS --skip_cleaning"
            shift
            ;;
        --skip-eval)
            SKIP_FLAGS="$SKIP_FLAGS --skip_eval"
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "错误: 未知选项 $1"
            print_help
            exit 1
            ;;
    esac
done

# 检查配置文件是否存在
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "错误: 配置文件 '$CONFIG_PATH' 不存在"
    exit 1
fi

# 构建执行命令
CMD="python main.py --config $CONFIG_PATH --data_dir $DATA_DIR --output_dir $OUTPUT_DIR"

# 添加视频路径(如果提供)
if [[ -n "$VIDEO_PATH" ]]; then
    CMD="$CMD --video_path $VIDEO_PATH"
fi

# 添加跳过标志
if [[ -n "$SKIP_FLAGS" ]]; then
    CMD="$CMD $SKIP_FLAGS"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 打印执行信息
echo "===================================="
echo "视频摘要 Pipeline 开始执行"
echo "===================================="
echo "配置文件: $CONFIG_PATH"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
if [[ -n "$VIDEO_PATH" ]]; then
    echo "视频文件: $VIDEO_PATH"
fi
echo "跳过标志: $SKIP_FLAGS"
echo "===================================="

# 执行pipeline
echo "执行命令: $CMD"
echo "===================================="
eval "$CMD"

# 检查执行结果
if [[ $? -eq 0 ]]; then
    echo "===================================="
    echo "视频摘要 Pipeline 执行成功!"
    echo "结果已保存到: $OUTPUT_DIR"
    echo "===================================="
else
    echo "===================================="
    echo "视频摘要 Pipeline 执行失败!"
    echo "===================================="
    exit 1
fi
