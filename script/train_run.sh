#!/bin/bash

# ==============================================================================
# 8 GPUs 분산 학습 실행 스크립트 (PyTorch DDP)
# 실행 방법: 터미널에서 `bash train_run.sh` 또는 `./train_run.sh` 입력
# ==============================================================================

# 1. 로그(Log) 파일 저장을 위한 폴더 및 파일명 설정
LOG_DIR=./logs
mkdir -p "$LOG_DIR" # logs 폴더가 없으면 생성합니다. (-p 옵션: 에러 무시 및 상위 폴더 자동 생성)

# 현재 날짜와 시간(년월일_시분초)을 활용하여 덮어씌워지지 않는 고유한 로그 파일명을 생성합니다.
# (예: ./logs/train_20260601_155000.log)
LOGFILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 학습을 시작합니다..."
echo "Logging to $LOGFILE"

# ---------------------------------------------------------
# [옵션 A] 포그라운드(Foreground) 실행 모드 (현재 활성화됨)
# 터미널 화면에 학습 진행 상황을 실시간으로 출력하면서, 동시에 로그 파일에도 저장합니다.
# ---------------------------------------------------------
# python3 -m torch.distributed.run 은 torchrun 명령어와 완전히 동일한 역할을 합니다.
python3 -m torch.distributed.run --nproc_per_node=8 train_new.py \
    --distributed \
    --nusc-root ./ \
    --label-path hybrid_teacher_labels_final.json \
    --batch-size 2 \
    --val-split 0.05 \
    --val-batch-size 2 \
    --epochs 30 \
    --output-dir ./checkpoints/light_drive 2>&1 | tee "$LOGFILE"
    # [설명] 2>&1 : 프로그램의 에러 메시지(표준 에러)를 일반 메시지(표준 출력)와 하나로 합칩니다.
    # [설명] | tee "$LOGFILE" : 합쳐진 메시지를 터미널 화면에 보여주는 동시에(T자 형태 분기), 파일로도 저장합니다.


# ---------------------------------------------------------
# [옵션 B] 백그라운드(Background) 실행 모드 
# 터미널 창을 끄거나 SSH 접속이 끊겨도 학습이 계속 돌아가게 하려면 
# 위의 [옵션 A] 코드를 주석 처리(#)하고, 아래 코드들의 주석을 해제하세요.
# ---------------------------------------------------------
# nohup torchrun --nproc_per_node=8 train_new.py \
#   --distributed \
#   --nusc-root ./ \
#   --label-path hybrid_teacher_labels_final.json \
#   --batch-size 2 \
#   --epochs 30 \
#   --output-dir ./checkpoints/light_drive > "$LOGFILE" 2>&1 &
    # [설명] nohup : 터미널이 끊겨도 프로세스를 종료하지 말라는 명령어입니다.
    # [설명] > "$LOGFILE" 2>&1 : 화면에 출력하지 말고 모든 메시지와 에러를 로그 파일로 바로 보냅니다.
    # [설명] & : 명령어를 백그라운드에서 실행하라는 뜻입니다.

# echo $! > "$LOG_DIR/train.pid"
    # [설명] $! : 방금 백그라운드(&)로 실행시킨 프로세스의 고유 ID(PID)를 의미합니다.
    # 이를 train.pid 파일에 저장해두면, 나중에 학습을 강제로 중단하고 싶을 때 
    # 터미널에서 `kill $(cat ./logs/train.pid)` 명령어를 입력하여 쉽게 종료할 수 있습니다.