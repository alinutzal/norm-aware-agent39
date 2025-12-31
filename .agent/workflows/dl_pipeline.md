---
description: Run Deep Learning Training (Handling Nondeterminism & AMP)
---

This workflow manages the complexities of GPU training, specifically addressing nondeterminism and mixed precision instability.

1. **Set Deterministic Environment Variables**
   Critical for reproducibility in CUDA environments.
   ```bash
   export CUBLAS_WORKSPACE_CONFIG=:4096:8
   export PYTHONHASHSEED=42
   export PL_GLOBAL_SEED=42
   ```

2. **Sanity Check Resources**
   Verify GPU visibility and AMP compatibility.
   // turbo
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')"
   ```

3. **Launch Training with Monitoring**
   Run the training script with flags to handle AMP (Automatic Mixed Precision) and gradient clipping. Anomaly detection is enabled to catch NaNs early.
   // turbo
   ```bash
   python train_dl.py \
     --config config/dl_model_v2.yaml \
     --use-amp true \
     --grad-clip 1.0 \
     --detect-anomaly true \
     --checkpoint-dir checkpoints/run_001
   ```

4. **Verify Checkpoint integrity**
   Ensure the saved checkpoint can be loaded and inference produces valid outputs (not NaNs).
   // turbo
   ```bash
   python scripts/verify_checkpoint.py --checkpoint checkpoints/run_001/best.pt
   ```
