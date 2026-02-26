from models.single_scale_deit import SingleScaleDeiT

import wandb
import torch
run = wandb.init()
artifact = run.use_artifact('ehenicke-friedrich-schiller-universit-t-jena/fmow/run-single-deit-adamw-5e-05-02-26:v20', type='model')
artifact_dir = artifact.download()
checkpoint_path = f'{artifact_dir}/checkpoint.pt'
print(f'Checkpoint downloaded to: {checkpoint_path}')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model = SingleScaleDeiT(num_labels=62)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
print(model)