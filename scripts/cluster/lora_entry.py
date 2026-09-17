"""Entry point for distributed LoRA: `mlx.launch` runs a script file on every
host, so this wraps mlx_lm's LoRA CLI. mlx_lm detects the distributed group,
splits each global batch across workers and averages gradients."""
from mlx_lm.lora import main

if __name__ == "__main__":
    main()
