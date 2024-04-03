Code for the paper:

*"Confounder-Aware Image Synthesis for Pathology Segmentation in New Magnetic Resonance Imaging Sequences"*

## Usage

Create a conda environment with:

```bash
conda env create -f env.yml --name synthpath
```

Activate the environment:

```bash
conda activate synthpath
```

Install the source code in editable mode:

```bash
pip install -e .
```

Train a model (see configs folder for inspiration):

```bash
synthpath_cli fit --config path_to_your_config.yaml
```

## SLAug baseline

We provide a link to our forked repository of the original code to aid transparency:

https://github.com/Jesse-Phitidis/SynthPath_SLAug_Baselines