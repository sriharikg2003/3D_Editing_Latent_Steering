import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils
import imageio

LATENT_DIR = "interfacegan_trellis/latents"
SCORE_PATH = "interfacegan_trellis/scores_continuous.npy"
MODEL_PATH = "/mnt/data/srihari/MODELS/TRELLIS-text-xlarge"

X_raw = np.array([np.load(f"{LATENT_DIR}/lat_{i}.npy").flatten() for i in range(500)])
y_raw = np.load(SCORE_PATH)

X_tensor = torch.from_numpy(X_raw).float().cuda()
y_tensor = torch.from_numpy(y_raw).float().cuda().view(-1, 1)

class LatentClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_tensor.shape[1]
model = LatentClassifier(input_dim).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

indices = np.argsort(y_raw)
thin_idx = indices[:100]
thick_idx = indices[-100:]
train_idx = np.concatenate([thin_idx, thick_idx])

X_train = X_tensor[train_idx]
y_train = torch.cat([torch.zeros(100), torch.ones(100)]).cuda().view(-1, 1)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

model.eval()
avg_latent = X_train.mean(dim=0, keepdim=True).requires_grad_(True)
prediction = model(avg_latent)
prediction.backward()

direction = avg_latent.grad.cpu().numpy()
direction = direction / np.linalg.norm(direction)
np.save("thick_direction_mlp.npy", direction)

pipeline = TrellisTextTo3DPipeline.from_pretrained(MODEL_PATH)
pipeline.cuda()

base_shape = pipeline.encode_text([""]).shape
thick_vec = torch.from_numpy(direction).cuda().float().view(base_shape)

@torch.no_grad()
def generate(prompt, strength):
    base_cond = pipeline.encode_text([prompt])
    edited_cond = base_cond + (strength * thick_vec)
    cond_dict = {'cond': edited_cond, 'neg_cond': pipeline.text_cond_model['null_cond']}
    torch.manual_seed(42)
    coords = pipeline.sample_sparse_structure(cond_dict)
    slat = pipeline.sample_slat(cond_dict, coords)
    outputs = pipeline.decode_slat(slat, formats=['gaussian'])
    return outputs['gaussian'][0]

test_prompt = "A simple minimalist chair"
for s in [15.0, 0.0, -15.0]:
    gs = generate(test_prompt, s)
    video = render_utils.render_video(gs)['color']
    imageio.mimsave(f"edit_strength_{s}.mp4", video, fps=30)