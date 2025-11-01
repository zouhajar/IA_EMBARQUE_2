import os
import subprocess

# Model name
MODEL_NAME = "mobilenet_quan"

# Define the 4 training configurations
training_configs = [
    {"name": "LowLR", "lr": 0.01, "clipping": 0.0, "randbet": 0},
    {"name": "Clipping", "lr": 0.1, "clipping": 0.1, "randbet": 0},
]

# Loop over each configuration
for i, cfg in enumerate(training_configs, 1):
    print("\n" + "=" * 70)
    print(f" Training {cfg['name']} model ({i}/4)")
    print("=" * 70 + "\n")

    # Build the command for this configuration
    cmd = [
        "python", "main.py",
        "--arch", MODEL_NAME,
        "--learning_rate", str(cfg["lr"]),
        "--clipping_coeff", str(cfg["clipping"])
    ]

    # Add randbet flag if needed
    if cfg["randbet"] == 1:
        cmd.append("--randbet")

    print(f"Executing command: {' '.join(cmd)}\n")

    # Execute the command
    subprocess.run(cmd)

print("\nAll 4 training runs are complete!")
print("Check your ./save/ directory for the trained models.")