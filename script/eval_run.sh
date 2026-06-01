#!/bin/bash

# ==============================================================================
# Multi-GPU Evaluation Script (모델별 결과 분리 저장)
# ==============================================================================

# ---------------------------------------------------------
# 1. 전역 변수 설정 (환경에 맞게 수정하는 부분)
# ---------------------------------------------------------
# 평가에 사용할 총 GPU 개수
NUM_GPUS=8

# 평가에 사용할 테스트 데이터셋의 정답(GT) 라벨 JSON 파일 경로
TEST_LABEL_PATH="./hybrid_teacher_labels_final.json"

# 모든 평가 결과가 저장될 최상위 디렉토리
BASE_OUTPUT_DIR="./evaluation_results"

# ---------------------------------------------------------
# 2. 체크포인트 경로 설정
# ---------------------------------------------------------
# 학습이 완료된 각 모델별 최적의 가중치(Best Model) 파일 경로
CHECKPOINT_LIGHT_DRIVE="./checkpoints/light_drive/best_model.pth"
CHECKPOINT_BASELINE_A="./checkpoints/baseline_a/best_model.pth"
CHECKPOINT_BASELINE_B="./checkpoints/baseline_b/best_model.pth"

echo "🚀 3개 모델에 대한 Multi-GPU 평가를 시작합니다..."

# ---------------------------------------------------------
# 3. 함수 정의: 모델별 평가 실행 및 결과 디렉토리 분리
# ---------------------------------------------------------
# 반복되는 코드를 줄이기 위해 함수(run_eval)로 묶어 처리합니다.
run_eval() {
    # 함수의 인자로 전달받은 값들을 지역 변수(local)로 저장
    local model_type=$1   # 첫 번째 인자: 모델 타입 이름 (예: light_drive)
    local checkpoint=$2   # 두 번째 인자: 체크포인트 경로
    
    # 해당 모델의 결과만 따로 모아둘 전용 폴더 경로 생성 (예: ./evaluation_results/light_drive)
    local save_dir="${BASE_OUTPUT_DIR}/${model_type}"
    
    echo "========================================================="
    echo " Evaluating [$model_type]"
    echo "========================================================="
    
    # 모델별로 결과 디렉토리 생성 (-p 옵션: 이미 있으면 무시, 상위 폴더 없으면 생성)
    mkdir -p "$save_dir"
    
    # torchrun을 사용하여 다중 GPU(DDP) 환경에서 파이썬 평가 스크립트 실행
    torchrun --nproc_per_node=$NUM_GPUS evaluate.py \
        --model-type "$model_type" \
        --checkpoint "$checkpoint" \
        --test-label-path "$TEST_LABEL_PATH" \
        --output-dir "$save_dir" # 위에서 생성한 각 모델 전용 폴더를 출력 경로로 지정
        
    echo "✅ [$model_type] 평가 완료! 결과 경로: $save_dir"
    echo ""
}

# ---------------------------------------------------------
# 4. 각 모델별 평가 순차적 실행
# ---------------------------------------------------------
# 위에서 정의한 run_eval 함수에 '모델명'과 '체크포인트경로'를 인자로 넘겨 실행합니다.
run_eval "light_drive" "$CHECKPOINT_LIGHT_DRIVE"
run_eval "baseline_a" "$CHECKPOINT_BASELINE_A"
run_eval "baseline_b" "$CHECKPOINT_BASELINE_B"

# ---------------------------------------------------------
# 5. 모든 작업 완료 알림
# ---------------------------------------------------------
echo "========================================================="
echo "🎉 모든 모델의 평가가 성공적으로 완료되었습니다!"
echo "각 모델의 요약 결과는 $BASE_OUTPUT_DIR/[모델명]/ 폴더에서 확인하세요."