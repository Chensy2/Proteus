# Proteus Baseline Core

This repository is a lightweight baseline package for using Proteus-style
unsupervised target adaptation with your own traffic-classification models.

It intentionally keeps only:

- JSON/JSONL/directory traffic data conversion to `.npz`.
- Core Proteus adaptation logic.
- A generic runner that imports an external PyTorch model.

It no longer vendors the original WFlib attack models or the authors' dataset
scripts. Your main repository should provide the attack model, source-trained
checkpoint, and final evaluation code.

## Data Format

The converter writes the minimal split layout:

```text
datasets/<DatasetName>/
  train.npz
  valid.npz
  test.npz
  label_mapping.json
  manifest.json
```

Each `.npz` contains:

```python
X  # [num_samples, seq_len]
y  # [num_samples]
```

For a faithful DF + Proteus baseline, store signed packet-size sequences in
`X` and run Proteus with `--feature DIR`. The loader applies `np.sign(X)` and
turns signed packet sizes into direction sequences.

## Convert JSON to NPZ

```powershell
python custom_dataset\json_to_npz.py `
  --source E:\data\source_json_or_dir `
  --target E:\data\target_json_or_dir `
  --out_dataset MyDrift `
  --view dir `
  --seq_len 5000 `
  --valid_ratio 0.1 `
  --seed 2024
```

Supported JSON layouts:

- `.json` list of samples.
- `.jsonl` one sample per line.
- Directory layout such as `class_a/*.json`, `class_b/*.json`.
- Dictionary layout such as `{"class_a": [...], "class_b": [...]}`.

Common keys are auto-detected:

- Packet-size sequence: `ps`, `packet_size`, `packet_length`, `flow`,
  `sequence`, `X`.
- IAT sequence: `iat`, `time`, `arrive_time_delta`, `inter_arrival_time`.
- Label: `label`, `y`, `class`, `class_name`, `site`, `website`.

Use explicit keys if needed:

```powershell
python custom_dataset\json_to_npz.py `
  --source E:\data\source.json `
  --target E:\data\target.json `
  --out_dataset MyDrift `
  --view dir `
  --sequence_key signed_ps `
  --label_key website_id `
  --seq_len 5000
```

## Run Proteus With an External Model

Your model must be a PyTorch module whose forward method returns:

```python
logits, features = model(inputs)
```

Example:

```powershell
python run_proteus.py `
  --train datasets\MyDrift\train.npz `
  --target datasets\MyDrift\test.npz `
  --feature DIR `
  --seq_len 5000 `
  --model_module my_repo.models.df `
  --model_class DF `
  --checkpoint E:\my_repo\checkpoints\df_source.pth `
  --output_checkpoint E:\my_repo\checkpoints\df_proteus.pth `
  --history_json E:\my_repo\results\df_proteus_history.json `
  --device cuda:0 `
  --batch_size 128 `
  --epochs 100 `
  --lr 1e-3 `
  --gmm_threshold 0.6
```

If your checkpoint is a dictionary containing the state dict under a key:

```powershell
--checkpoint_key state_dict
```

If the model constructor needs extra arguments:

```powershell
--model_kwargs "{`"dropout`": 0.1}"
```

## Proteus Objective

For each adaptation step, Proteus uses:

```text
source CE
+ pseudo-label target CE
+ target entropy/balance loss
+ source-target MMD
```

Target labels are not used in the adaptation loss. If target labels exist in
the `.npz`, they are used only for reporting target accuracy in the adaptation
history.

## Recommended Baseline Comparison

For your multi-modal method, keep the baseline single-view and faithful:

| Method | Backbone | Input View | Adaptation |
|---|---|---|---|
| Source-only | DF | DIR | None |
| Proteus baseline | DF | DIR | Proteus |
| Your single-view | DF-style | DIR | Yours |
| Your multi-view | DF-style encoders | DIR + PS + IAT | Yours |

This keeps the baseline clean while making the multi-view contribution explicit.
