from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/base.yaml")
print("seed from base:", cfg.seed)

viol_cfg = OmegaConf.load("configs/violations/excessive_batch_size.yaml")
print("seed from violation:", viol_cfg.seed)

merged = OmegaConf.merge(cfg, viol_cfg)
print("seed after merge:", merged.seed)
print("seed type:", type(merged.seed))
print("seed is None:", merged.seed is None)
print("seed == -1:", merged.seed == -1)

# Check getattr
seed_via_getattr = getattr(merged, "seed", None)
print("seed via getattr:", seed_via_getattr)
print("getattr is None:", seed_via_getattr is None)
