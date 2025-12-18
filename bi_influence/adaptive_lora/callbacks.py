import os
import logging
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from peft.tuners.lora import LoraLayer
from .importance import compute_bi_scores
from .allocation import allocate_ranks_bi
from .utils import get_lora_layers, save_epoch_log

logger = logging.getLogger(__name__)


def resize_lora_layer_svd(
    layer: LoraLayer,
    new_rank: int,
    lora_alpha: int,
    adapter_name: str = "default",
    **kwargs
):
    # If adapter not present, fallback to normal init
    if adapter_name not in layer.lora_A:
        layer.update_layer(
            adapter_name=adapter_name,
            r=new_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=True,
            **kwargs,
        )
        return

    old_r = layer.r.get(adapter_name, 0)

    # Handle rank-zero cases safely
    if old_r == 0 or new_rank == 0:
        layer.update_layer(
            adapter_name=adapter_name,
            r=new_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=True,
            **kwargs,
        )
        return

    with torch.no_grad():
        A_old = layer.lora_A[adapter_name].weight
        B_old = layer.lora_B[adapter_name].weight
        old_alpha = layer.lora_alpha[adapter_name]
        old_scaling = old_alpha / old_r

        # Effective LoRA update
        W_delta = (B_old @ A_old) * old_scaling

        dtype = W_delta.dtype
        U, S, Vh = torch.linalg.svd(W_delta.float(), full_matrices=False)

        k = min(new_rank, S.numel())

        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]

        sqrt_S = torch.diag(torch.sqrt(S_k))
        B_new = (U_k @ sqrt_S).to(dtype)
        A_new = (sqrt_S @ Vh_k).to(dtype)

    # Resize LoRA layer WITHOUT reinitialization
    layer.update_layer(
        adapter_name=adapter_name,
        r=new_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=False,
        **kwargs,
    )

    # Correct write-back with padding logic
    with torch.no_grad():
        device = layer.lora_A[adapter_name].weight.device

        if k < new_rank:
            layer.lora_A[adapter_name].weight.data.zero_()
            layer.lora_B[adapter_name].weight.data.zero_()

            layer.lora_A[adapter_name].weight.data[:k, :] = A_new.to(device)
            layer.lora_B[adapter_name].weight.data[:, :k] = B_new.to(device)
        else:
            layer.lora_A[adapter_name].weight.data = A_new.to(device)
            layer.lora_B[adapter_name].weight.data = B_new.to(device)


class AdaptiveLoRACallback(TrainerCallback):
    def __init__(
        self,
        total_rank: int,
        val_dataloader,
        min_rank: int = 4,
        tau: float = 1.0,
        log_path: str = "./logs",
        verbose: bool = True,
        validate_batch_size: int = 4,
        lora_alpha_multiplier: int = 4,
    ):
        self.lora_alpha_multiplier = lora_alpha_multiplier
        self.total_rank = total_rank
        self.val_dataloader = val_dataloader
        self.tau = tau
        self.min_rank = min_rank
        self.verbose = verbose
        self.validate_batch_size = validate_batch_size
        self.log_file = os.path.join(log_path, "adaptive_lora_epoch_logs.csv")

        os.makedirs(log_path, exist_ok=True)

        self.latest_scores = {}
        self.latest_ranks = {}

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs,
    ):
        epoch = int(state.epoch) + 1 if state.epoch is not None else 0

        if self.verbose:
            print(f"\n--- AdaptiveLoRA: Preparing ranks for Epoch {epoch} ---")

        device = next(model.parameters()).device

        if self.verbose:
            print("Computing BI importance scores (pre-training)...")

        scores = compute_bi_scores(
            model,
            self.val_dataloader,
            device,
            batch_size=self.validate_batch_size,
        )

        if not scores:
            if self.verbose:
                print("⚠️ No LoRA layers or BI scores found. Skipping rank update.")
            return

        if self.verbose:
            print("Allocating new ranks based on BI scores...")

        new_ranks = allocate_ranks_bi(
            scores, self.total_rank, self.tau, self.min_rank
        )

        if self.verbose:
            print("Applying new ranks to LoRA modules for this epoch...")

        lora_layers = get_lora_layers(model)
        config = model.peft_config.get("default")
        if not config:
            logger.error("❌ PEFT config not found. Skipping update.")
            return

        for name, layer in lora_layers.items():
            new_rank = new_ranks.get(name)
            if new_rank is None:
                continue

            current_rank = layer.r.get("default", 0)
            score = scores.get(name, 0.0)

            if self.verbose:
                if current_rank != new_rank:
                    print(
                        f"  - {name}: r={current_rank} → {new_rank} (Score: {score:.4f})"
                    )
                else:
                    print(
                        f"  - {name}: r={new_rank} (Unchanged, Score: {score:.4f})"
                    )

            if current_rank != new_rank:
                lora_dropout_p = 0.0
                if hasattr(layer, "lora_dropout") and "default" in layer.lora_dropout:
                    lora_dropout_p = layer.lora_dropout["default"].p

                resize_lora_layer_svd(
                    layer=layer,
                    new_rank=new_rank,
                    lora_alpha=new_rank * self.lora_alpha_multiplier,
                    adapter_name="default",
                    lora_dropout=lora_dropout_p,
                )

        self.latest_scores = scores
        self.latest_ranks = new_ranks

        if self.verbose:
            print(f"✅ AdaptiveLoRA: Rank setup for Epoch {epoch} complete.\n")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs,
    ):
        epoch = int(state.epoch) if state.epoch is not None else -1

        if self.latest_ranks and self.latest_scores:
            save_epoch_log(self.log_file, epoch, self.latest_ranks, self.latest_scores)
            if self.verbose:
                print(
                    f"📄 Epoch {epoch}: Rank allocations logged to {self.log_file}\n"
                )
