import argparse
import json
import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import timm
import torch
import torch.nn as nn
import torch.distributed as dist

from nuscenes.nuscenes import NuScenes
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


# =========================================================
# Dataset
# =========================================================
class HybridFusionDataset(Dataset):
    def __init__(self, nusc, label_path, transform=None):
        self.nusc = nusc
        self.transform = transform
        self.camera_names = CAMERA_NAMES

        with open(label_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        self.samples = [
            sample
            for sample in nusc.sample
            if sample["token"] in self.labels
        ]

        print(f"[*] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def build_camera_vector(self, calib):
        intrinsic = torch.tensor(
            calib["camera_intrinsic"],
            dtype=torch.float32,
        ).flatten()

        rotation = torch.tensor(
            calib["rotation"],
            dtype=torch.float32,
        )

        translation = torch.tensor(
            calib["translation"],
            dtype=torch.float32,
        )

        vec = torch.cat([
            intrinsic,
            rotation,
            translation,
        ])

        return vec

    def __getitem__(self, idx):
        sample = self.samples[idx]

        imgs = []
        matrices = []

        for cam_name in self.camera_names:
            cam_data = self.nusc.get(
                "sample_data",
                sample["data"][cam_name]
            )

            img_path = os.path.join(
                self.nusc.dataroot,
                cam_data["filename"]
            )

            with Image.open(img_path) as img:
                img = img.convert("RGB")

                if self.transform:
                    img = self.transform(img)

                imgs.append(img)

            calib = self.nusc.get(
                "calibrated_sensor",
                cam_data["calibrated_sensor_token"]
            )

            matrices.append(
                self.build_camera_vector(calib)
            )

        return (
            torch.stack(imgs),
            torch.stack(matrices),
            self.labels[sample["token"]],
        )


# =========================================================
# Q-Former Style Cross Attention
# =========================================================
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        attn_out, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
        )

        x = self.norm1(query + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)

        return x


# =========================================================
# Multi-Camera Prefix VLM
# =========================================================
class MultiCameraPrefixVLM(nn.Module):
    def __init__(
        self,
        language_model_name="Qwen/Qwen2-0.5B-Instruct",
        prefix_len=32,
        num_qformer_layers=4,
        num_unfrozen_blocks=4,
    ):
        super().__init__()

        self.prefix_len = prefix_len

        # -------------------------------------------------
        # Vision Encoder
        # -------------------------------------------------
        self.vision_encoder = timm.create_model(
            "deit_small_patch16_224",
            pretrained=True,
        )
        self.vision_dim = 384

        # -------------------------------------------------
        # Language Model
        # -------------------------------------------------
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name,
            torch_dtype=torch.float16,
        )
        lm_dim = self.language_model.config.hidden_size

        # -------------------------------------------------
        # Projection
        # -------------------------------------------------
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

        # -------------------------------------------------
        # Geometry Encoder
        # -------------------------------------------------
        self.camera_encoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.GELU(),
            nn.Linear(128, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

        # -------------------------------------------------
        # Camera Embedding
        # -------------------------------------------------
        self.camera_embedding = nn.Parameter(
            torch.randn(len(CAMERA_NAMES), lm_dim)
        )

        # -------------------------------------------------
        # Query Tokens
        # -------------------------------------------------
        self.query_tokens = nn.Parameter(
            torch.randn(1, prefix_len, lm_dim) * 0.02
        )

        # -------------------------------------------------
        # Q-Former Blocks
        # -------------------------------------------------
        self.qformer_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=lm_dim,
                num_heads=8,
            )
            for _ in range(num_qformer_layers)
        ])

        self.prefix_norm = nn.LayerNorm(lm_dim)

        # -------------------------------------------------
        # Freeze LM
        # -------------------------------------------------
        self.freeze_language_model(num_unfrozen_blocks)

    def freeze_language_model(self, num_unfrozen_blocks):
        for p in self.language_model.parameters():
            p.requires_grad = False

        transformer_layers = self.language_model.model.layers

        for block in transformer_layers[-num_unfrozen_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

        for p in self.language_model.model.norm.parameters():
            p.requires_grad = True

        for p in self.language_model.lm_head.parameters():
            p.requires_grad = True

    def extract_patch_tokens(self, images):
        features = self.vision_encoder.forward_features(images)
        # remove cls token
        features = features[:, 1:, :]
        return features

    def encode_prefix(self, images, camera_matrices):
        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W)

        patch_tokens = self.extract_patch_tokens(images)
        patch_tokens = self.vision_proj(patch_tokens)

        num_patches = patch_tokens.shape[1]
        patch_tokens = patch_tokens.view(B, N, num_patches, -1)

        # -------------------------------------------------
        # Geometry Embedding
        # -------------------------------------------------
        geom_tokens = self.camera_encoder(camera_matrices)
        geom_tokens = geom_tokens.unsqueeze(2)

        # -------------------------------------------------
        # Camera Embedding
        # -------------------------------------------------
        cam_embed = self.camera_embedding.unsqueeze(0).unsqueeze(2)

        # -------------------------------------------------
        # Add geometry + camera embedding
        # -------------------------------------------------
        patch_tokens = patch_tokens + geom_tokens + cam_embed

        # flatten
        patch_tokens = patch_tokens.view(B, N * num_patches, -1)

        # -------------------------------------------------
        # Q-Former
        # -------------------------------------------------
        queries = self.query_tokens.expand(B, -1, -1)

        for blk in self.qformer_blocks:
            queries = blk(
                query=queries,
                key_value=patch_tokens,
            )

        # Fix 1: LLM의 원래 dtype(float16)과 맞춰주어 캐스팅 에러 방지
        return self.prefix_norm(queries).to(self.language_model.dtype)

    def build_inputs_embeds(
        self,
        images,
        camera_matrices,
        tokenizer,
        prompt_text,
    ):
        prefix_embeds = self.encode_prefix(images, camera_matrices)
        B = images.shape[0]
        prompt = [prompt_text] * B

        prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        ).input_ids.to(images.device)

        prompt_embeds = self.language_model.model.embed_tokens(prompt_ids)

        inputs_embeds = torch.cat([
            prefix_embeds,
            prompt_embeds,
        ], dim=1)

        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=images.device,
        )

        return (
            inputs_embeds,
            attention_mask,
            prompt_ids,
        )

    def forward(
        self,
        images,
        camera_matrices,
        tokenizer,
        labels,
    ):
        prompt_text = "Describe the driving scene in detail."

        (
            inputs_embeds,
            attention_mask,
            prompt_ids,
        ) = self.build_inputs_embeds(
            images,
            camera_matrices,
            tokenizer,
            prompt_text,
        )

        label_tokens = tokenizer(
            labels,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        input_ids = label_tokens.input_ids.to(images.device)
        target_embeds = self.language_model.model.embed_tokens(input_ids)

        full_inputs_embeds = torch.cat([
            inputs_embeds,
            target_embeds,
        ], dim=1)

        target_attention = label_tokens.attention_mask.to(images.device)
        full_attention_mask = torch.cat([
            attention_mask,
            target_attention,
        ], dim=1)

        prefix_ignore = torch.full(
            (input_ids.shape[0], inputs_embeds.shape[1]),
            -100,
            dtype=torch.long,
            device=images.device,
        )

        labels_tensor = input_ids.masked_fill(
            target_attention == 0,
            -100,
        )

        full_labels = torch.cat([
            prefix_ignore,
            labels_tensor,
        ], dim=1)

        outputs = self.language_model(
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )

        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        images,
        camera_matrices,
        tokenizer,
        max_new_tokens=80,
    ):
        self.eval()
        prompt_text = "Describe the driving scene in detail."

        (
            inputs_embeds,
            attention_mask,
            _,
        ) = self.build_inputs_embeds(
            images,
            camera_matrices,
            tokenizer,
            prompt_text,
        )

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        return tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )


# =========================================================
# Training
# =========================================================
def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    # 안전한 프로세스 그룹 해제
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def train(args):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token
    # Fix 2: Causal LM 배치의 올바른 텍스트 생성을 위해 Left Padding 필수로 적용
    tokenizer.padding_side = "left"

    nusc = NuScenes(
        version=args.nusc_version,
        dataroot=args.nusc_root,
        verbose=False,
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = HybridFusionDataset(
        nusc,
        args.label_path,
        transform,
    )

    train_size = int(0.9 * len(dataset))
    train_db, val_db = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
    )

    train_sampler = DistributedSampler(train_db)
    # Fix 4: DDP 환경에서 검증셋 중복 연산을 방지하기 위한 Sampler 정의
    val_sampler = DistributedSampler(val_db, shuffle=False)

    train_loader = DataLoader(
        train_db,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_db,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
    )

    model = MultiCameraPrefixVLM(
        language_model_name=args.language_model,
    )
    model = model.to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
    )

    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.max_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.03),
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.max_epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        pbar = tqdm(
            train_loader,
            disable=(local_rank != 0),
        )

        total_loss = 0.0

        for imgs, mats, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            mats = mats.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                loss = model(
                    imgs,
                    mats,
                    tokenizer,
                    labels,
                )

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                1.0,
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            if local_rank == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}"
                })

        if local_rank == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")

            os.makedirs("./checkpoints", exist_ok=True)
            torch.save(
                model.module.state_dict(),
                f"./checkpoints/epoch_{epoch}.pth",
            )


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nusc-root", default="./nuscenes")
    parser.add_argument("--nusc-version", default="v1.0-trainval")
    parser.add_argument("--label-path", default="./hybrid_labels.json")
    parser.add_argument("--language-model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()

    # Fix 3: 에러 발생 여부와 무관하게 항상 DDP가 안전하게 클린업되도록 제어
    try:
        train(args)
    finally:
        cleanup_ddp()
