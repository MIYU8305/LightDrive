#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 호환성 확보: 현재 셸이 bash가 아닐 경우 명시적으로 bash로 스크립트 재실행
# ---------------------------------------------------------------------------
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

# 파이프라인 에러 방지 (Robust Bash Strict Mode)
# -e: 에러 발생 시 즉시 종료 (Fail-fast)
# -u: 정의되지 않은 변수 참조 시 에러 발생
# -o pipefail: 파이프라인(|) 내 중간 명령어 실패를 최종 반환값으로 전달
set -euo pipefail

# 실험 결과 및 로그 저장을 위한 디렉토리 구성
LOG_DIR=./logs
mkdir -p "$LOG_DIR"

# 공통 학습 에포크 설정 (모든 실험에 동일한 10 에포크 적용)
EPOCHS=10

# ==============================================================================
# 하이퍼파라미터 및 아키텍처 탐색 공간(Search Space) 정의
# ==============================================================================
MODEL_TYPES=("baseline_a" "baseline_b" "light_drive")
BACKBONES=("deit_small_patch16_224" "vit_base_patch16_224")
CAM_EMB=("true" "false") # 카메라 ID 임베딩 Ablation
GEOM=("true" "false")    # 3D 기하학 조건화 Ablation
QFORMER=("true" "false") # Q-Former 기반 토큰 압축 Ablation

# 최상단 루프: 아키텍처 타입별 순회
for model in "${MODEL_TYPES[@]}"; do

  # ---------------------------------------------------------
  # 1. Baseline A (ResNet101 고정 모델)
  # 비전 백본(ResNet101)이 고정되어 있고 추가 조건화 모듈이 없으므로,
  # 하위 루프 탐색 없이 단독으로 1회만 실행 후 다음 모델로 스킵합니다.
  # ---------------------------------------------------------
  if [[ "$model" == "baseline_a" ]]; then
    exp_name="baseline_a_resnet101"
    OUT_DIR="./checkpoints/${exp_name}"
    # 타임스탬프를 부여하여 이전 실험 로그 덮어쓰기 방지
    LOGFILE="$LOG_DIR/${exp_name}_$(date +%Y%m%d_%H%M%S).log"

    echo "============================================================="
    echo "Running experiment: $exp_name (Epochs: $EPOCHS)"
    echo "============================================================="

    # torchrun(DDP) 기반 단일 노드 8-GPU 분산 학습 실행
    cmd=(python3 -m torch.distributed.run --nproc_per_node=8 train_new.py
      --distributed
      --model-type "$model"
      --nusc-root ./
      --label-path hybrid_teacher_labels_final.json
      --batch-size 2
      --val-split 0.05
      --val-batch-size 2
      --epochs "$EPOCHS"
      --output-dir "$OUT_DIR"
    )
    
    # 명령어 로깅 및 실행 (stdout과 stderr를 병합하여 파일 및 콘솔 동시 출력)
    echo "${cmd[*]}" | tee "$LOGFILE"
    "${cmd[@]}" 2>&1 | tee -a "$LOGFILE"
    echo "Finished experiment: $exp_name"
    echo "-------------------------------------------------------------"
    continue # Baseline A 실행 완료 후, 하위 루프(백본/모듈 탐색)를 건너뜀
  fi

  # Baseline B와 Light Drive를 위한 비전 백본 순회 루프
  for backbone in "${BACKBONES[@]}"; do
    
    # ---------------------------------------------------------
    # 2. Baseline B (비전 백본 변경 가능 모델)
    # 조건화 모듈(cam, geom, qformer) 적용이 불필요한 모델이므로,
    # 백본의 종류별로만 1회씩 실행하여 불필요한 중복 연산을 차단합니다.
    # ---------------------------------------------------------
    if [[ "$model" == "baseline_b" ]]; then
      exp_name="baseline_b_${backbone}"
      OUT_DIR="./checkpoints/${exp_name}"
      LOGFILE="$LOG_DIR/${exp_name}_$(date +%Y%m%d_%H%M%S).log"

      echo "============================================================="
      echo "Running experiment: $exp_name (Epochs: $EPOCHS)"
      echo "============================================================="

      cmd=(python3 -m torch.distributed.run --nproc_per_node=8 train_new.py
        --distributed
        --model-type "$model"
        --vision-backbone "$backbone"
        --nusc-root ./
        --label-path hybrid_teacher_labels_final.json
        --batch-size 2
        --val-split 0.05
        --val-batch-size 2
        --epochs "$EPOCHS"
        --output-dir "$OUT_DIR"
      )
      
      echo "${cmd[*]}" | tee "$LOGFILE"
      "${cmd[@]}" 2>&1 | tee -a "$LOGFILE"
      echo "Finished experiment: $exp_name"
      echo "-------------------------------------------------------------"
      continue # Baseline B 실행 완료 후, 하위 모듈 탐색 루프를 건너뜀
    fi

    # ---------------------------------------------------------
    # 3. Light Drive (Ablation Study 대상)
    # 메인 제안 아키텍처이므로 카메라 임베딩, 기하학 조건화, Q-Former의
    # 모든 on/off 경우의 수(2^3 = 8가지)에 대해 매트릭스 탐색을 수행합니다.
    # ---------------------------------------------------------
    for cam_emb in "${CAM_EMB[@]}"; do
      for geom in "${GEOM[@]}"; do
        for qformer in "${QFORMER[@]}"; do
          
          # 동적 실험 식별자 조합 (체크포인트 디렉토리 및 로깅용)
          exp_name="light_drive_${backbone}"
          
          if [[ "$cam_emb" == "true" ]]; then exp_name+="_cam_on"; else exp_name+="_cam_off"; fi
          if [[ "$geom" == "true" ]]; then exp_name+="_geom_on"; else exp_name+="_geom_off"; fi
          if [[ "$qformer" == "true" ]]; then exp_name+="_qf_on"; else exp_name+="_qf_off"; fi

          OUT_DIR="./checkpoints/${exp_name}"
          LOGFILE="$LOG_DIR/${exp_name}_$(date +%Y%m%d_%H%M%S).log"

          echo "============================================================="
          echo "Running experiment: $exp_name"
          echo "Model Type: $model"
          echo "Backbone: $backbone"
          echo "Camera Embedding: $cam_emb"
          echo "Geometry Conditioning: $geom"
          echo "Q-Former: $qformer"
          echo "Output dir: $OUT_DIR"
          echo "Log file: $LOGFILE"
          echo "============================================================="

          # Base 명령어 구성
          cmd=(python3 -m torch.distributed.run --nproc_per_node=8 train_new.py
            --distributed
            --model-type "$model"
            --vision-backbone "$backbone"
            --nusc-root ./
            --label-path hybrid_teacher_labels_final.json
            --batch-size 2
            --val-split 0.05
            --val-batch-size 2
            --epochs "$EPOCHS" # 공통 에포크 변수 동적 할당
          )

          # Boolean 파라미터에 따른 동적 Flag 주입 (argparse action 대응)
          if [[ "$cam_emb" == "true" ]]; then cmd+=(--use-camera-embedding); else cmd+=(--no-camera-embedding); fi
          if [[ "$geom" == "true" ]]; then cmd+=(--use-geometry-conditioning); else cmd+=(--no-geometry-conditioning); fi
          if [[ "$qformer" == "true" ]]; then cmd+=(--use-qformer); else cmd+=(--no-qformer); fi

          cmd+=(--output-dir "$OUT_DIR")

          # 실행 기록 보존 및 학습 프로세스 구동
          echo "${cmd[*]}" | tee "$LOGFILE"
          "${cmd[@]}" 2>&1 | tee -a "$LOGFILE"

          echo ""
          echo "Finished experiment: $exp_name"
          echo "-------------------------------------------------------------"
          
        done
      done
    done
  done
done
