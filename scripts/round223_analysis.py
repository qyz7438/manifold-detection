"""Plan 2.23: Gate frequency response visualization (analysis-only, no training).

Load mid06_5ep checkpoint, analyze gate output distribution on val set.
Output: gate_suppression vs input_magnitude scatter, per-frequency histograms.
"""
import sys, json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import load_checkpoint
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round216pp_mid06_s42/checkpoint_last.pth"
RUN_DIR = Path("runs/round223_gate_viz")
RUN_DIR.mkdir(parents=True, exist_ok=True)

set_seed(42)
cfg = {
    "model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
              "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
              "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320,
              "afm_channels": 256, "afm_type": "mplseg_mid"},
}
model = build_detector(cfg).to(DEV)
load_checkpoint(model, CKPT, DEV)
model.eval()

afm = model.roi_heads.box_head.afm
_, val_loader = build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320,
    "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 1}})

# Hook to collect gate inputs and outputs
gate_inputs = []
gate_outputs = []

def gate_hook(module, inp, out):
    # Capture the sigmoid(log(mag)) input to mp conv
    # We need to hook inside the AFM forward
    pass

# Instead, run full model and extract AFM internals via forward hook
mag_inputs = []
mag_outputs = []
phase_inputs = []
phase_outputs = []

original_forward = afm.forward

@torch.no_grad()
def hooked_forward(x):
    fr = torch.fft.rfft2(x, norm="ortho")
    mag = torch.abs(fr)
    pha = torch.angle(fr + 1e-3)

    log_mag = torch.sigmoid(torch.log(mag + 1e-3))
    mp_out = afm.mp(log_mag)
    mag_inputs.append(log_mag.detach().cpu().flatten().numpy())
    mag_outputs.append(mp_out.detach().cpu().flatten().numpy())

    pa_out = afm.pa(pha)
    phase_inputs.append(pha.detach().cpu().flatten().numpy())
    phase_outputs.append(pa_out.detach().cpu().flatten().numpy())

    mag = mag * (1.0 - afm.gate_strength * mp_out)
    pha = pha + pa_out
    fr = mag * torch.exp(1j * pha)
    freq_out = torch.fft.irfft2(fr, s=x.shape[-2:], norm="ortho")
    freq_out = F.relu(freq_out, inplace=False)
    return x + afm.residual_scale * freq_out

afm.forward = hooked_forward

# Run through val set (first 20 images for speed)
max_images = 20
for i, (images, _) in enumerate(tqdm(val_loader, desc="analyzing gate", total=min(max_images, len(val_loader)))):
    if i >= max_images:
        break
    _ = model.backbone(images[0].unsqueeze(0).to(DEV))
    # Run a dummy detection pass to trigger AFM
    try:
        model(images, None)
    except Exception:
        pass

afm.forward = original_forward

# Aggregate and save
all_mag_in = np.concatenate(mag_inputs) if mag_inputs else np.array([])
all_mag_out = np.concatenate(mag_outputs) if mag_outputs else np.array([])
all_phase_in = np.concatenate(phase_inputs) if phase_inputs else np.array([])
all_phase_out = np.concatenate(phase_outputs) if phase_outputs else np.array([])

results = {
    "plan": "2.23",
    "mag_input_mean": float(np.mean(all_mag_in)) if len(all_mag_in) > 0 else 0,
    "mag_input_std": float(np.std(all_mag_in)) if len(all_mag_in) > 0 else 0,
    "mag_output_mean": float(np.mean(all_mag_out)) if len(all_mag_out) > 0 else 0,
    "mag_output_std": float(np.std(all_mag_out)) if len(all_mag_out) > 0 else 0,
    "mag_suppression": float(np.mean(all_mag_out)) if len(all_mag_out) > 0 else 0,
    "phase_input_mean": float(np.mean(all_phase_in)) if len(all_phase_in) > 0 else 0,
    "phase_output_mean": float(np.mean(all_phase_out)) if len(all_phase_out) > 0 else 0,
    "phase_modulation": float(np.std(all_phase_out)) if len(all_phase_out) > 0 else 0,
}

# Save raw distributions for plotting
np.savez(RUN_DIR / "gate_distributions.npz",
         mag_input=all_mag_in, mag_output=all_mag_out,
         phase_input=all_phase_in, phase_output=all_phase_out)
json.dump(results, open(RUN_DIR / "gate_analysis.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(results, indent=2, ensure_ascii=False))
