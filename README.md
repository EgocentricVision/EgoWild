<!-- # EgoWild
Implementation of Bringing Online Egocentric Action Recognition into the wild
 -->
# Bringing Online Egocentric Action Recognition into the wild
##### Gabriele Goletto*, Mirco Planamente*, Barbara Caputo, Giuseppe Averta.

This repository contains the code for the [paper](https://arxiv.org/abs/2211.03004): "Bringing Online Egocentric Action Recognition into the wild".

*Abstract:* To enable a safe and effective human-robot cooperation, it is crucial to develop models for the identification of human activities. Egocentric vision seems to be a viable solution to solve this problem, and therefore many works provide deep learning solutions to infer human actions from first person videos. However, although very promising, most of these do not consider the major challenges that comes with a realistic deployment, such as the portability of the model, the need for real-time inference, and the robustness with respect to the novel domains (i.e., new spaces, users, tasks). With this paper, we set the boundaries that egocentric vision models should consider for realistic applications, defining a novel setting of egocentric action recognition in the wild, which encourages researchers to develop novel, applications-aware solutions. We also present a new model-agnostic technique that enables the rapid repurposing of existing architectures in this new context, demonstrating the feasibility to deploy a model on a tiny device (Jetson Nano) and to perform the task directly on the edge with very low energy consumption (2.4W on average at 50 fps).

![dropo_general_framework](figures/teaser.png)

# Bringing Online Egocentric Action Recognition into the wild
# Getting started
Install the required libraries using the provided `requirements.yml` file
```
cd EgoWild
conda env create -f requirements.yml
conda activate egowild
```

# Data structure
The code expects the EPIC-Kitchens data to be organized according to the following structure. 
You can download the EPIC-Kitchens frames using [this script](https://github.com/jonmun/MM-SADA_Domain_Adaptation_Splits/blob/master/download_script.sh).

<details> <summary><b> </b></summary><br/>

```
├── dataset_root
|   ├── P01_01
|   |   ├── img_0000000000.jpg
|   |   ├── x_0000000000.jpg
|   |   ├── y_0000000000.jpg
|   |   ├── .
|   |   ├── .
|   |   ├── .
|   |   ├── img_0000000100.jpg
|   |   ├── x_0000000100.jpg
|   |   ├── y_0000000100.jpg
|   ├── .
|   ├── .
|   ├── .
|   ├── P22_17
|   |   ├── img_0000000000.jpg
|   |   ├── x_0000000000.jpg
|   |   ├── y_0000000000.jpg
|   |   ├── .
|   |   ├── .
|   |   ├── .
|   |   ├── img_0000000250.jpg
|   |   ├── x_0000000250.jpg
|   |   ├── y_0000000250.jpg
```

</details>


# Configuration
Each experiment is defined by a different configuration file (in the `configs/` directory).

All the experiments inherit the configuration inserted in the field `extends`.
Experiments may indicate additional parameters or override the default values.

Before running any experiments, the following values in the default configuration file must be updated 
to point to the local replica of the EPIC-Kitchens dataset.
```yaml
dataset:
  RGB:
    data_path: /path/to/EPIC-KITCHENS/dataset_root
  Flow:
    data_path: /path/to/EPIC-KITCHENS/dataset_root
```

# Python commands to launch the training for MoViNet
Standard clip-wise validation
```bash
python -m train.train \
          action=train \
          name=<model_name> \
          dataset.shift=<shift_source>-<shift_target> \
          config=movinet.yaml
```

# Python commands to launch all the different validation proposed for MoViNet

In the following all the commands which we used for the experiments in our work will be listed.
<details> <summary><b> </b></summary><br/>

## Standard clip-wise validation
```bash
python -m train.train \
          action=validate \
          name=<model_name> \
          dataset.shift=<shift_source>-<shift_target>
```

## Trimmed standard validation
```bash
python -m train.train \
          action=validate \
          name=<model_name> \
          dataset.shift=<shift_source>-<shift_target> \
          config=trimmed.yaml
```
For all boundaries detection methods in trimmed scenario just use the same arguments as in the untrimmed one 
(notice that ABD is not implemented for the trimmed case)

## Untrimmed validation - single buffer - boundaries supervision
```bash
python -m train.train \
          action=validate \
          name=<model_name> \
          dataset.shift=<shift_source>-<shift_target> \
          config=untrimmed.yaml
```

## Untrimmed validation - single buffer - SBL
```bash
python -m train.train \
          action=validate \
          name=<model_name> \
          dataset.shift=<shift_source>-<shift_target> \
          config=untrimmed.yaml \
          boundaries_supervision=False \
          SBL_k=<k>
```

## Untrimmed validation - single buffer - DBL
```bash
python -m train.train \
          action=validate \
          name=<model_name> \
          dataset.shift=<shift_source>-<shift_target> \
          config=untrimmed.yaml \
          boundaries_supervision=False \
          DBL_threshold=<DBL_threshold>
```

## Untrimmed validation - single buffer - ABD
```bash
python -m train.train \
          action=validate \
          name=<model_name> \
          dataset.shift=<shift_source>-<shift_target> \
          config=untrimmed.yaml \
          boundaries_supervision=False \
          ABD=<ABD_size>
```

## Untrimmed validation - double buffer - boundaries supervision
```bash
python -m train.train \
          action=validate \
          name=<model_name> \
          dataset.shift=<shift_source>-<shift_target> \
          config=untrimmed_db.yaml 
```

## Untrimmed validation - double buffer - DBL
```bash
python -m train.train \
          action=validate \
          name=<model_name> \
          dataset.shift=<shift_source>-<shift_target> \
          config=untrimmed_db.yaml \
          boundaries_supervision=False \
          DBL_threshold=<DBL_threshold>
```
</details>

# I3D training and validation
We report all the equivalent of MoViNet config files also with I3D model. 
To run the equivalent train/test experiment with I3D you should just:
1) Pick the command among the MoViNet one
2) Choose the proper file in the `configs` folder
3) Insert the I3D config version in the command

# Cite us
If you use this work, please consider citing

```
@article{goletto2023bringing,
  title={Bringing Online Egocentric Action Recognition into the wild},
  author={Goletto, Gabriele and Planamente, Mirco and Caputo, Barbara and Averta, Giuseppe},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
```

# Contact :pushpin:
If you have any question, do not hesitate to contact me:

- Gabriele Goletto: <code> gabriele.goletto@polito.it</code>