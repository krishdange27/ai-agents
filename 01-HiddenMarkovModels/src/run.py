from src.dataset import generate_raw_logs
from src.pipeline import encode_observations, build_model

username = "your_github_username"

logs = generate_raw_logs(username)
encoded = [encode_observations(seq) for seq in logs]

model = build_model(encoded)

print("Model built successfully.")