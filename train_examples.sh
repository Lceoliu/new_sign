#!/bin/bash

# Uni-Sign 训练配置示例脚本
# 使用方法: bash train_examples.sh [config_name]

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Uni-Sign 训练配置示例 ===${NC}"

# 检查参数
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}可用的配置选项:${NC}"
    echo "  basic     - 基础训练 (仅 TensorBoard)"
    echo "  wandb     - 启用 wandb 日志"
    echo "  debug     - 调试模式"
    echo "  inference - 在线推理示例"
    echo ""
    echo "使用方法: bash $0 [config_name]"
    exit 1
fi

CONFIG=$1

case $CONFIG in
    "basic")
        echo -e "${GREEN}启动基础训练配置...${NC}"
        python fine_tuning.py \
            --dataset CSL-Daily \
            --task SLT \
            --epochs 50 \
            --batch_size 8 \
            --learning_rate 1e-4 \
            --output_dir ./outputs/basic_training \
            --seed 42
        ;;

    "wandb")
        echo -e "${GREEN}启动 wandb 增强训练配置...${NC}"
        python fine_tuning.py \
            --dataset CSL-Daily \
            --task SLT \
            --epochs 50 \
            --batch_size 8 \
            --learning_rate 1e-4 \
            --output_dir ./outputs/wandb_training \
            --use_wandb \
            --wandb_project uni-sign-experiments \
            --wandb_run_name pose-only-v1 \
            --seed 42
        ;;

    "debug")
        echo -e "${YELLOW}启动调试模式...${NC}"
        python fine_tuning.py \
            --dataset CSL-Daily \
            --task SLT \
            --epochs 2 \
            --batch_size 2 \
            --learning_rate 1e-4 \
            --output_dir ./outputs/debug \
            --seed 42
        ;;

    "inference")
        echo -e "${GREEN}在线推理示例...${NC}"
        if [ ! -f "./outputs/best_checkpoint.pth" ]; then
            echo -e "${RED}错误: 未找到模型检查点文件 ./outputs/best_checkpoint.pth${NC}"
            echo "请先完成训练或指定正确的检查点路径"
            exit 1
        fi

        if [ ! -f "./demo_video.mp4" ]; then
            echo -e "${RED}错误: 未找到演示视频文件 ./demo_video.mp4${NC}"
            echo "请提供一个手语视频文件"
            exit 1
        fi

        python demo/online_inference.py \
            --online_video ./demo_video.mp4 \
            --finetune ./outputs/best_checkpoint.pth
        ;;

    *)
        echo -e "${RED}错误: 未知的配置选项 '$CONFIG'${NC}"
        echo "请使用: basic, wandb, debug, 或 inference"
        exit 1
        ;;
esac

echo -e "${GREEN}配置 '$CONFIG' 执行完成!${NC}"