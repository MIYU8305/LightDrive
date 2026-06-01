import argparse
import json
import os
import math
from tqdm import tqdm

import torch
import torch.distributed as dist
from torchvision import transforms
from transformers import AutoTokenizer

# 자연어 생성 품질 평가를 위한 메트릭 라이브러리
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 커스텀 모듈 임포트 (학습 데이터셋 및 인퍼런스용 모델)
from train_new import NuScenesTeacherDataset, CAMERA_NAMES
from visualization_new import MultiCameraPrefixVLMForInference, build_camera_tensors

# =========================================================
# NLTK 초기화
# =========================================================
# 텍스트 토큰화(단어 분리)에 필요한 NLTK punkt 데이터를 다운로드합니다. (최초 1회 실행)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def evaluate_model(args):
    # =========================================================
    # 1. Multi-GPU(Distributed) 환경 셋업
    # =========================================================
    # torchrun으로 실행 시 환경 변수에서 WORLD_SIZE(총 GPU 수)와 LOCAL_RANK(현재 GPU 번호)를 가져옵니다.
    is_distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    local_rank = int(os.environ.get('LOCAL_RANK', '0')) if is_distributed else 0
    world_size = int(os.environ.get('WORLD_SIZE', '1')) if is_distributed else 1

    # 분산 환경 초기화 (NCCL 백엔드 사용)
    if is_distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if local_rank == 0:
        print(f"[*] Starting Evaluation for {args.model_type} on {world_size} GPUs")

    # =========================================================
    # 2. 토크나이저 및 모델 로드
    # =========================================================
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 추론용 아키텍처 모델 생성
    model = MultiCameraPrefixVLMForInference(
        model_type=args.model_type,
        language_model_name=args.language_model,
        prefix_len=args.prefix_len,
    ).to(device)

    # 체크포인트 로드
    if local_rank == 0:
        print(f"[*] Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    
    # DDP 환경에서 저장된 가중치는 이름 앞에 'module.'이 붙어있으므로 이를 제거하여 로드합니다.
    clean_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval() # 평가 모드 설정 (Dropout 등 비활성화)

    # 입력 이미지 전처리 설정
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 평가용 정답(GT) 라벨 로드
    with open(args.test_label_path, 'r', encoding='utf-8') as f:
        test_labels = json.load(f)
    
    test_tokens = list(test_labels.keys())

    # =========================================================
    # 3. 각 GPU별로 평가할 데이터 분할 (Data Sharding)
    # =========================================================
    # 전체 테스트 데이터를 GPU 개수만큼 균등하게 잘라서 각 GPU가 자기 할당량만 평가하도록 합니다.
    if is_distributed:
        chunk_size = math.ceil(len(test_tokens) / world_size)
        test_tokens = test_tokens[local_rank * chunk_size : (local_rank + 1) * chunk_size]

    # NLP 평가지표 설정 (ROUGE, BLEU)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    total_rouge_l = 0.0
    total_bleu_4 = 0.0
    results_log = {}

    # =========================================================
    # 4. 추론 루프 시작
    # =========================================================
    with torch.no_grad(): # 그래디언트 계산 비활성화
        # 진행바는 마스터 노드(rank 0)에서만 출력하여 화면 겹침 방지
        progress_bar = tqdm(test_tokens, desc=f"Evaluating GPU {local_rank}") if local_rank == 0 else test_tokens
        for token in progress_bar:
            label_info = test_labels[token]
            ego_speed = label_info.get("ego_speed", 0.0)
            
            # 정답 텍스트(GT) 재구성 (카메라별 분석 + 전역 추론)
            gt_per_camera = "\n".join(label_info.get("per_camera_perception", []))
            gt_reasoning = label_info.get("global_reasoning", "")
            gt_text = f"### Per-Camera Analysis\n{gt_per_camera}\n\n{gt_reasoning}" if gt_per_camera.strip() else gt_reasoning

            # 데이터셋에서 이미지와 행렬(카메라 파라미터) 텐서를 가져옴
            sample = test_dataset.nusc.get('sample', token)
            input_imgs, input_mats, _ = build_camera_tensors(test_dataset.nusc, sample, transform)
            
            input_imgs = input_imgs.to(device)
            input_mats = input_mats.to(device)

            # 모델 텍스트 생성 (온도(Temperature)를 0으로 설정하여 결정론적 생성)
            pred_text = model.generate(
                input_imgs, input_mats, tokenizer, 
                ego_speed=ego_speed, 
                max_new_tokens=512, 
                temperature=0.0, 
                do_sample=False
            )[0].strip()

            # ROUGE-L 스코어 계산
            rouge_score = scorer.score(gt_text, pred_text)['rougeL'].fmeasure
            total_rouge_l += rouge_score

            # BLEU-4 스코어 계산
            gt_tokens = nltk.word_tokenize(gt_text.lower())
            pred_tokens = nltk.word_tokenize(pred_text.lower())
            bleu_score = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothie)
            total_bleu_4 += bleu_score

            # 개별 결과 로깅
            results_log[token] = {
                "prediction": pred_text,
                "ground_truth": gt_text,
                "rougeL": rouge_score,
                "bleu4": bleu_score
            }

    # =========================================================
    # 5. 각 GPU별 개별 디테일 결과 저장
    # =========================================================
    os.makedirs(args.output_dir, exist_ok=True)
    local_output_file = os.path.join(args.output_dir, f"metrics_{args.model_type}_rank{local_rank}_details.json")
    with open(local_output_file, 'w', encoding='utf-8') as f:
        json.dump({"details": results_log}, f, indent=4)

    # =========================================================
    # 🔥 6. 모든 GPU의 스코어 합산 및 최종 결과 출력
    # =========================================================
    # 현재 GPU의 총합 점수와 처리한 샘플 수를 텐서로 묶습니다.
    local_metrics = torch.tensor([total_rouge_l, total_bleu_4, len(test_tokens)], dtype=torch.float64, device=device)

    # dist.all_reduce를 사용하여 모든 GPU의 텐서 값을 하나로 더합니다(SUM).
    # 연산 이후 모든 GPU의 local_metrics 텐서에는 '전체 GPU 데이터의 합산값'이 저장됩니다.
    if is_distributed:
        dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)

    # 합산된 데이터 추출
    global_rouge_sum = local_metrics[0].item()
    global_bleu_sum = local_metrics[1].item()
    global_total_samples = int(local_metrics[2].item())

    # 0번 GPU(마스터 노드)에서만 최종 평균 결과를 계산하고 화면에 출력 및 저장
    if local_rank == 0:
        global_avg_rouge = global_rouge_sum / global_total_samples if global_total_samples > 0 else 0
        global_avg_bleu = global_bleu_sum / global_total_samples if global_total_samples > 0 else 0

        print("\n" + "="*50)
        print(f"🏆 Final Global Results for {args.model_type}")
        print(f"Total Evaluated Samples: {global_total_samples}")
        print(f"Average ROUGE-L : {global_avg_rouge:.4f}")
        print(f"Average BLEU-4  : {global_avg_bleu:.4f}")
        print("="*50 + "\n")

        # 전역 최종 스코어 요약 파일 저장
        global_output_file = os.path.join(args.output_dir, f"metrics_{args.model_type}_final_summary.json")
        with open(global_output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model_type": args.model_type,
                "total_samples": global_total_samples,
                "global_avg_rougeL": global_avg_rouge,
                "global_avg_bleu4": global_avg_bleu
            }, f, indent=4)
        print(f"[*] Saved final summary to: {global_output_file}")

    # =========================================================
    # 7. 종료 정리
    # =========================================================
    # 모든 프로세스가 끝날 때까지 대기(동기화)한 후 프로세스 그룹 소멸
    if is_distributed:
        dist.barrier() 
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["light_drive", "baseline_a", "baseline_b"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-label-path", default="./hybrid_teacher_labels_test.json") 
    parser.add_argument("--nusc-root", default="./")
    parser.add_argument("--language-model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--prefix-len", type=int, default=32)
    parser.add_argument("--output-dir", default="./evaluation_results")
    
    args = parser.parse_args()
    
    # Dataset 객체 초기화 (데이터 메타정보 접근 용도)
    tokenizer_dummy = AutoTokenizer.from_pretrained(args.language_model)
    global test_dataset 
    test_dataset = NuScenesTeacherDataset(args.nusc_root, args.test_label_path, tokenizer_dummy, None)
    
    evaluate_model(args)