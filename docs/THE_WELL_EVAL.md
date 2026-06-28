# PDEformer Evaluation and Fine-Tuning on The Well

This repo includes lightweight adapters for evaluating and fine-tuning
PDEformer on datasets loaded with `the_well.data.WellDataset`.

The dataset-specific equation DAGs live in
`src/data/well_equations.py`. The registry is populated from the Well dataset
descriptions in `PolymathicAI the_well master datasets`, and is used by both
the evaluator and the `single_pde` fine-tuning dataset.

## Fine-Tuning

```bash
sbatch --export=ALL,WELL_BASE_PATH=/scratch/the_well,WELL_DATASET=gray_scott_reaction_diffusion \
  scripts/submit_the_well_finetune_slurm.sh
```

Useful overrides:

```bash
sbatch --export=ALL,\
WELL_BASE_PATH=/scratch/the_well,\
WELL_DATASET=viscoelastic_instability,\
FIELD_INDICES=0,1,2,3,4,5,6,7,\
MAX_FIELDS=8,\
TRAIN_SAMPLES=8,\
TEST_SAMPLES=8,\
EPOCHS=20,\
CHECKPOINT=model-L.pt \
scripts/submit_the_well_finetune_slurm.sh
```

The Slurm wrapper writes a concrete YAML file into `slurm_logs/` using
`configs/finetune/the_well_model-L.yaml` as the base. That generated config
uses:

- `data.single_pde.param_name: the_well`
- `data.single_pde.train: ["DATASET:train"]`
- `data.single_pde.test: ["DATASET:valid"]`
- `data.single_pde.well` options for normalization, time stride, field
  selection, and coordinate normalization

Only 2D Well datasets are currently supported for PDEformer-2 fine-tuning.
The 3D dataset equations are registered for documentation/notebook completeness,
but the adapter intentionally raises on non-2D grids.

Per-dataset notebooks are available in `notebooks/the_well_*_finetune.ipynb`.

## Evaluation

```bash
python scripts/evaluate_the_well.py \
  --config configs/inference/model-L.yaml \
  --checkpoint model-L.pt \
  --well-base-path /path/to/the_well \
  --well-dataset active_matter \
  --split test \
  --num-samples 16 \
  --pde-preset well_equation \
  --well-normalization zscore \
  --output exp/the_well/active_matter_test_pdeformer_eval.json
```

For Slurm:

```bash
sbatch --export=ALL,WELL_BASE_PATH=/path/to/the_well,WELL_DATASET=active_matter \
  scripts/submit_the_well_eval_slurm.sh
```

For Hugging Face streaming, use:

```bash
sbatch --export=ALL,WELL_BASE_PATH=hf://datasets/polymathic-ai/,WELL_DATASET=active_matter \
  scripts/submit_the_well_eval_slurm.sh
```

For a private or synthetic Well-format dataset root, bypass the public dataset
name registry and use a generic preset unless you add the dataset to
`src/data/well_equations.py`:

```bash
sbatch --export=ALL,WELL_PATH=/path/to/my_dataset,WELL_DATASET=my_dataset,PDE_PRESET=generic_unknown \
  scripts/submit_the_well_eval_slurm.sh
```

## What The Evaluator Does

The Well stores trajectories and metadata, while PDEformer expects a symbolic
PDE graph. The adapter therefore builds a PDEformer graph for the selected
output channels:

1. Load a sample from `WellDataset`.
2. Use the final input frame as the initial condition for the selected unknowns.
3. Interpolate that initial condition to PDEformer's function-encoder grid
   resolution, usually 128 by 128.
4. Build a dataset-specific PDE graph using `--pde-preset well_equation`.
5. Query PDEformer on the requested Well output time grid.
6. Compare predictions against the Well target fields.

The default `well_equation` preset uses `src/data/well_equations.py`.
The legacy `generic_unknown` preset represents
`u_t + F(u, channel_id) = 0`, where `F` is a PDEformer arbitrary-transform node.
The `constant` preset represents `u_t = 0`. These presets are adapters, not
dataset-specific physical equations.

## Benchmark-Matching Pieces

The script follows The Well benchmark conventions where practical:

- Data is loaded through `the_well.data.WellDataset`.
- `--well-normalization zscore` and `--well-normalization rms` use The Well's
  own `ZScoreNormalization` and `RMSNormalization` classes.
- With normalization enabled, PDEformer receives normalized initial conditions,
  and predictions/targets are denormalized before metrics.
- Metrics are computed on channel-last tensors shaped like
  `[batch, time, spatial..., channels]`.
- The script runs The Well validation metric suite:
  `RMSE`, `NRMSE`, `LInfinity`, `VRMSE`, `binned_spectral_mse`, and `PearsonR`.

`VRMSE` in The Well package is the root variance-scaled MSE: it is `NRMSE` with
the target standard deviation used as the normalization factor.

## Important Differences

This is not identical to evaluating a native Well benchmark model:

- PDEformer is equation-conditioned, but Well samples do not include PDEformer
  DAGs. The adapter supplies DAGs from the local Well equation registry.
- Some Well READMEs reference paper equations or include unresolved forcing,
  source, equation-of-state, geometry, or closure terms. Those DAGs encode the
  visible terms and use learned closure nodes for the unspecified pieces.
- The Well benchmark models roll out autoregressively by feeding predictions
  back into the model. This adapter predicts the requested output grid from the
  first input-frame DAG and does not currently rebuild the DAG per rollout step.
- The script currently supports 2D Well grids. Several Well datasets are 3D or
  have domain-specific structure that needs a custom adapter.
- Channel selection is explicit. By default, the script evaluates up to
  `--max-fields` leading flattened variable channels. Use `--field-indices`
  for precise channel choices.
- Spectral metrics are computed on the selected channel subset, so averages are
  not directly comparable to full-field Well benchmark logs unless all channels
  are selected.

## Useful Slurm Overrides

The Slurm wrapper is configured by environment variables:

```bash
sbatch --export=ALL,\
WELL_BASE_PATH=/scratch/the_well,\
WELL_DATASET=active_matter,\
CONFIG_PATH=configs/inference/model-L.yaml,\
CHECKPOINT=model-L.pt,\
NUM_SAMPLES=64,\
FIELD_INDICES=0,1,\
WELL_NORMALIZATION=zscore,\
OUTPUT_PATH=exp/the_well/active_matter_test_model-L.json \
scripts/submit_the_well_eval_slurm.sh
```

Set `WELL_NORMALIZATION=none` if you want to feed PDEformer raw Well fields.
That can be useful when comparing against PDEformer’s original raw-scale
behavior, but it is less close to The Well benchmark model pipeline.

Set `--no-well-metrics` on the Python script if you only want the simpler
relative-L2 summary and want to skip spectral metrics.
