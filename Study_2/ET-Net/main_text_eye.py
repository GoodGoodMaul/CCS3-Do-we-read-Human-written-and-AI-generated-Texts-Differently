import argparse
import ast
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from network import CA_SA, linear

torch.cuda.empty_cache()

def _parse_eye_matrix(raw_value, eye_feature_dim: int) -> torch.Tensor:
    """
    Normalize the raw eye feature field into a float tensor with shape [num_words, eye_feature_dim].
    The CSV can store this as a stringified list, a nested list, or a tensor when saved via torch.save.
    """
    if isinstance(raw_value, torch.Tensor):
        eye = raw_value
    else:
        if isinstance(raw_value, str):
            raw_value = ast.literal_eval(raw_value)
        eye = torch.tensor(raw_value)
    if eye.ndim == 1:
        eye = eye.unsqueeze(0)
    if eye.shape[1] != eye_feature_dim:
        raise ValueError(f"Expected eye feature dim {eye_feature_dim}, got {eye.shape[1]}")
    return eye.to(torch.float32)


def _detect_eye_dim(samples: List[dict]) -> int:
    for s in samples:
        m = s.get("Eye_Feature_Matrix")
        if m is None:
            continue
        try:
            if isinstance(m, str):
                m = ast.literal_eval(m)
            eye = torch.tensor(m)
            if eye.ndim == 1:
                continue
            if eye.shape[0] == 0:
                continue
            return eye.shape[1]
        except Exception:
            continue
    raise ValueError("Unable to auto-detect eye_feature_dim; please set --eye_feature_dim explicitly")


def load_samples(data_path: str) -> List[dict]:
    if data_path.endswith(".pt"):
        samples = torch.load(data_path)
    else:
        df = pd.read_csv(data_path)
        samples = df.to_dict("records")
    return samples


class TextEyeDataset(Dataset):
    def __init__(
        self,
        samples: List[dict],
        tokenizer: AutoTokenizer,
        split: str,
        max_length: int = 256,
        eye_feature_dim: int = 20,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eye_feature_dim = eye_feature_dim
        self.samples = [
            s
            for s in samples
            if str(s.get("Split", split)).lower() == split.lower()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        text = str(sample.get("Text", ""))
        eye_matrix = _parse_eye_matrix(
            sample.get("Eye_Feature_Matrix"), self.eye_feature_dim
        )
        label_field = sample.get("label", sample.get("Label"))
        if label_field is None:
            label_field = 1 if str(sample.get("Source", "AI")).lower() == "ai" else 0
        label = int(label_field)

        encoded = self.tokenizer(
            text,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        word_ids = encoded.word_ids()

        eye_features = torch.zeros(
            self.max_length, self.eye_feature_dim, dtype=torch.float32
        )
        eye_mask = torch.zeros(self.max_length, dtype=torch.float32)
        seen_words = set()

        for i, w_id in enumerate(word_ids):
            if attention_mask[i] == 0 or w_id is None:
                continue
            if w_id in seen_words:
                continue
            seen_words.add(w_id)
            if w_id < eye_matrix.shape[0]:
                eye_features[i] = eye_matrix[w_id]
                eye_mask[i] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "eye_features": eye_features,
            "eye_mask": eye_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


class EyeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        max_tokens: int = 512,
    ):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_size)
        self.max_tokens = max_tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, eye_features: torch.Tensor, eye_mask: torch.Tensor) -> torch.Tensor:
        x = self.project(eye_features)
        key_padding_mask = ~(eye_mask.bool())
        # simple downsample if sequence is too long to save memory
        if x.shape[1] > self.max_tokens:
            step = int(torch.ceil(torch.tensor(x.shape[1] / self.max_tokens)).item())
            # mean pool every step tokens
            new_len = (x.shape[1] + step - 1) // step
            x = x.view(x.shape[0], new_len, step, -1).mean(dim=2)
            eye_mask = eye_mask.view(eye_mask.shape[0], new_len, step).max(dim=2).values
            key_padding_mask = ~(eye_mask.bool())
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return x


class FunnyNetTextEye(nn.Module):
    def __init__(
        self,
        text_model_name: str,
        hidden_size: int = 256,
        eye_feature_dim: int = 20,
        eye_encoder_layers: int = 2,
        eye_encoder_heads: int = 8,
        eye_dropout: float = 0.1,
        fusion_dropout: float = 0.1,
        eye_max_tokens: int = 512,
        freeze_text: bool = True,
        use_text: bool = True,
        use_eye: bool = True,
        use_device_map: bool = False,
        compute_device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.use_text = use_text
        self.use_eye = use_eye
        self.hidden_size = hidden_size  # target projection dim
        self.compute_device = compute_device
        self.text_proj = None
        if self.use_text:
            text_kwargs = dict(
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            if use_device_map:
                text_kwargs["device_map"] = "auto"
            self.text_model = AutoModel.from_pretrained(
                text_model_name,
                **text_kwargs,
            )
            text_hidden = self.text_model.config.hidden_size
            self.text_proj = nn.Linear(text_hidden, self.hidden_size)
            if freeze_text:
                for p in self.text_model.parameters():
                    p.requires_grad = False
            self.text_model.eval()
        else:
            self.text_model = None
        final_hidden_size = self.hidden_size

        self.eye_encoder = EyeEncoder(
            input_dim=eye_feature_dim,
            hidden_size=final_hidden_size,
            num_layers=eye_encoder_layers,
            nhead=eye_encoder_heads,
            dropout=eye_dropout,
            max_tokens=eye_max_tokens,
        )

        self.cross_text = CA_SA(dim=final_hidden_size)
        self.cross_eye = CA_SA(dim=final_hidden_size)
        self.self_attn = CA_SA(dim=final_hidden_size)
        self.dropout = nn.Dropout(fusion_dropout)

        self.classifier = linear(final_hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        eye_features: torch.Tensor,
        eye_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_out = None
        if self.use_text:
            with torch.no_grad():
                text_out = self.text_model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state.to(self.compute_device, dtype=torch.float32)
            if self.text_proj is not None:
                text_out = self.text_proj(text_out)
            text_out = text_out * attention_mask.unsqueeze(-1)

        eye_hidden = self.eye_encoder(eye_features, eye_mask) if self.use_eye else None

        # Cross attention: text queries eye, eye queries text
        if text_out is not None and eye_hidden is not None:
            text_fused = self.cross_text(text_out, eye_hidden, key_padding_mask=eye_mask.bool())
            eye_fused = self.cross_eye(eye_hidden, text_out, key_padding_mask=attention_mask.bool())
            fused = torch.cat([text_fused, eye_fused], dim=1)
            pool_mask = torch.cat([attention_mask, eye_mask], dim=1)
        elif text_out is not None:
            fused = text_out
            pool_mask = attention_mask
        else:
            fused = eye_hidden
            pool_mask = eye_mask

        # Self-attn on fused tokens
        fused = self.self_attn(fused, fused, key_padding_mask=pool_mask.bool()) + fused
        fused = self.dropout(fused)
 
        mask = pool_mask.unsqueeze(-1)
        masked_sum = (fused * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled = masked_sum / denom
        logits = self.classifier(pooled)
        return logits


def train_one_epoch(
    model: FunnyNetTextEye,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    lr: float,
    mode: str,
    grad_accum_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    log_every = max(1, len(loader) // 10)
    optimizer.zero_grad()
    for step, batch in enumerate(loader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        eye_features = batch["eye_features"].to(device)
        eye_mask = batch["eye_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, eye_features, eye_mask)
        loss = criterion(logits, labels)

        loss = loss / grad_accum_steps
        loss.backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * labels.size(0) * grad_accum_steps
        if step % log_every == 0 or step == len(loader):
            print(
                f"[Train e{epoch}] step {step}/{len(loader)} "
                f"lr={lr} mode={mode} loss={loss.item():.4f}",
                flush=True,
            )
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: FunnyNetTextEye,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tag: str = "Eval",
    collect_preds: bool = False,
) -> tuple:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    log_every = max(1, len(loader) // 10)
    tp_pos = 0
    fp_pos = 0
    fn_pos = 0
    all_labels = [] if collect_preds else None
    all_preds = [] if collect_preds else None
    for step, batch in enumerate(loader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        eye_features = batch["eye_features"].to(device)
        eye_mask = batch["eye_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, eye_features, eye_mask)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        pc = preds.detach().cpu()
        lc = labels.detach().cpu()
        tp_pos += int(((pc == 1) & (lc == 1)).sum())
        fp_pos += int(((pc == 1) & (lc != 1)).sum())
        fn_pos += int(((pc != 1) & (lc == 1)).sum())
        if collect_preds:
            all_preds.append(pc)
            all_labels.append(lc)
        if step % log_every == 0 or step == len(loader):
            print(
                f"[{tag}] step {step}/{len(loader)} "
                f"loss={loss.item():.4f}",
                flush=True,
            )
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    eps = 1e-8
    precision = tp_pos / (tp_pos + fp_pos + eps)
    recall = tp_pos / (tp_pos + fn_pos + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    print(f"[{tag}] loss={avg_loss:.4f} acc={acc:.4f} f1={f1:.4f}")
    if collect_preds:
        labels_cat = torch.cat(all_labels).numpy()
        preds_cat = torch.cat(all_preds).numpy()
        return avg_loss, acc, f1, labels_cat, preds_cat
    return avg_loss, acc, f1


def build_loaders(
    samples: List[dict],
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
):
    # Auto-detect eye feature dim if requested
    eye_dim = args.eye_feature_dim
    if eye_dim == -1:
        eye_dim = _detect_eye_dim(samples)
        args.eye_feature_dim = eye_dim

    train_ds = TextEyeDataset(
        samples,
        tokenizer=tokenizer,
        split="train",
        max_length=args.max_length,
        eye_feature_dim=eye_dim,
    )
    val_ds = TextEyeDataset(
        samples,
        tokenizer=tokenizer,
        split="val",
        max_length=args.max_length,
        eye_feature_dim=eye_dim,
    )
    test_ds = TextEyeDataset(
        samples,
        tokenizer=tokenizer,
        split="test",
        max_length=args.max_length,
        eye_feature_dim=eye_dim,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FunnyNet-W Text+Eye refactor")
    parser.add_argument(
        "--data_path",
        type=str,
        default="processed_dataset_final.csv",
        help="CSV or PT file containing Text, Eye_Feature_Matrix, label, Split columns (for raw data.csv run data.py first)",
    )
    parser.add_argument(
        "--text_model",
        type=str,
        default="/research/projects/mllab/public_llms/Qwen2.5-7B-Instruct",
        help="Local path or HuggingFace model name for Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--max_length", type=int, default=140)
    parser.add_argument("--eye_feature_dim", type=int, default=-1, help="eye feature dimension; if -1, auto-detect from data")
    parser.add_argument("--hidden_size", type=int, default=3584, help="projection dim for both text and eye before cross-attention")
    parser.add_argument("--eye_encoder_layers", type=int, default=1)
    parser.add_argument("--eye_encoder_heads", type=int, default=2)
    parser.add_argument("--eye_dropout", type=float, default=0.1)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--eye_max_tokens", type=int, default=512, help="downsample eye seq len if > this")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2, help="early stopping patience on val_loss")
    parser.add_argument("--lr_reduce_patience", type=int, default=1, help="ReduceLROnPlateau patience on val_loss")
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5, help="LR reduce factor on plateau")
    parser.add_argument("--lr_min", type=float, default=1e-7, help="Minimum learning rate for scheduler")
    parser.add_argument("--unfreeze_text", action="store_true")
    parser.add_argument("--disable_text", action="store_true", help="run eye-only")
    parser.add_argument("--disable_eye", action="store_true", help="run text-only")
    parser.add_argument("--device_map_auto", action="store_true", help="load text model with device_map='auto' (HF dispatch across visible GPUs)")
    parser.add_argument(
        "--save_path",
        type=str,
        default="funnynet_text_eye.pt",
        help="Where to save the classifier weights",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "both"
    if args.disable_eye:
        mode = "text"
    elif args.disable_text:
        mode = "eye"

    tokenizer = AutoTokenizer.from_pretrained(
        args.text_model, use_fast=True, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    samples = load_samples(args.data_path)

    # Auto-enable HF device_map sharding if multiple GPUs are visible and flag not set; always log status.
    device_count = torch.cuda.device_count()
    auto_msg = None
    if device_count > 1 and not args.device_map_auto:
        args.device_map_auto = True
        auto_msg = "Auto-enabled"
    elif args.device_map_auto:
        auto_msg = "User-enabled"
    if auto_msg:
        print(f"{auto_msg} device_map='auto' for the HF text encoder; visible GPUs: {device_count}")

    train_loader, val_loader, test_loader = build_loaders(samples, tokenizer, args)

    model = FunnyNetTextEye(
        text_model_name=args.text_model,
        hidden_size=args.hidden_size,
        eye_feature_dim=args.eye_feature_dim,
        eye_encoder_layers=args.eye_encoder_layers,
        eye_encoder_heads=args.eye_encoder_heads,
        eye_dropout=args.eye_dropout,
        fusion_dropout=args.fusion_dropout,
        eye_max_tokens=args.eye_max_tokens,
        freeze_text=not args.unfreeze_text,
        use_text=not args.disable_text,
        use_eye=not args.disable_eye,
        use_device_map=args.device_map_auto,
        compute_device=device,
    )
    if args.device_map_auto:
        # Only move non-text modules; text model is dispatched by HF.
        model.eye_encoder.to(device)
        model.cross_text.to(device)
        model.cross_eye.to(device)
        model.self_attn.to(device)
        model.classifier.to(device)
        if model.text_proj is not None:
            model.text_proj.to(device)
        model.dropout.to(device)
    else:
        model = model.to(device)

    # class weights and label smoothing
    weight_tensor = None
    if args.use_class_weights:
        # compute counts
        from collections import Counter
        lbls = [int(s.get("label", s.get("Label", 0))) for s in samples]
        counter = Counter(lbls)
        total = sum(counter.values())
        weights = [total / counter.get(i, 1) for i in range(2)]
        weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_reduce_factor,
        patience=args.lr_reduce_patience,
        min_lr=args.lr_min,
        verbose=True,
    )

    best_val_f1 = float("-inf")
    patience_ctr = 0
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch}/{args.epochs}] mode={mode} lr={current_lr} "
            f"train_batches={len(train_loader)} val_batches={len(val_loader)}",
            flush=True,
        )
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.lr, mode, args.grad_accum_steps
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, tag="Val")
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"f1={val_f1:.4f}"
        )
        scheduler.step(val_loss)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved new best model to {args.save_path} (best val_f1={best_val_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience {args.patience})")
                break

    # Load best model before final test to avoid using a worse checkpoint
    save_path = Path(args.save_path)
    if save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded best model from {save_path} for final test (best val_f1={best_val_f1:.4f})")
    else:
        print(f"Warning: best model file {save_path} not found; using last epoch weights for test.")

    test_loss, test_acc, test_f1, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device, tag="Test", collect_preds=True
    )
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f}")
    print(classification_report(test_labels, test_preds, digits=4))
    print(confusion_matrix(test_labels, test_preds))


if __name__ == "__main__":
    main()
