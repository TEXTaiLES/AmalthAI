# AmalthAI Machine Learning Platform

<p align="center">
    <a href = "https://github.com/TEXTaiLES/AmalthAI" target="_blank">
        <img src="images/AmalthAI-Logo.png" alt="AmalthAI" width="250"/>
    </a>
</p>

AmalthAI is a machine learning platform developed as part of the [TEXTaiLES](https://textailes-eccch.eu/) toolbox which offers end-to-end interaction with the whole machine learning project pipeline using an intuitive interface.

AmalthAI is designed for uses in cultural heritage while follows an easy-to-interact architecture. 

<p align="center">
    <a href = "https://textailes-eccch.eu/" target="_blank">
        <img src="images/Logo-Textailes-Colour-RGB-Hor.png" alt="TEXTaiLES" width="600"/>
    </a>
</p>

## Basic Installation

The whole platform architecture is depicted in the following diagram:
<p align="center">
  <img src="images/amalthai_architecture.png" alt="Platform Architecture" width="800">
</p>

Minimum system requirements for a local installation are:

 - CPU: at least 8 cores
 - RAM: at least 16 GB
 - GPU: NVIDIA GPU with at least 12 GB VRAM
 - OS: Ubuntu 22.04 or newer
 - Browser: Google Chrome or Microsoft Edge

### Before You Start
Make sure that you have cloned the following repository on your local machine:

```bash
git clone https://github.com/TEXTaiLES/AmalthAI
```
- The `AmalthAI_WebApp` folder contains the web app code, which is necessary for the platform's UI. 

- The `Backend` folder contains all the necessary code for the platform's functionality which includes machine learning models, data processing scripts, and deployment configurations. Ensure that the content of the backend folder is also shared across the Kubernetes cluster and the corresponding containers.

### Step 1 - Docker Installation
Make sure that you have a local installation of [Docker](https://www.docker.com/). You can follow the installation process described [here](https://docs.docker.com/engine/install/ubuntu/):

<p align="center">
    <a href = "https://docs.docker.com/engine/install/ubuntu/" target="_blank">
        <img src="images/docker_logo.png" alt="Docker" width="200"/>
    </a>
</p>

### Step 2 - Kubernetes Installation
[Kubernetes](https://kubernetes.io/) installation is required to orchestrate the different components of the platform. You can follow the official installation tutorial [here](https://kubernetes.io/docs/setup/):

<p align="center">
    <a href = "https://kubernetes.io/docs/setup/" target="_blank">
        <img src="images/kubernetes_logo.png" alt="Kubernetes" width="200"/>
    </a>
</p>

### Step 3 - Kubeflow Installation
[Kubeflow](https://www.kubeflow.org/) installation is also required because experiment initialization is based on it. For installation, follow the instructions [here](https://www.kubeflow.org/docs/started/installing-kubeflow/):

<p align="center">
    <a href = "https://www.kubeflow.org/docs/started/installing-kubeflow/" target="_blank">
        <img src="images/kubeflow_logo.png" alt="Kubeflow" width="200"/>
    </a>
</p>

For local development and quick testing, the [kind](https://kind.sigs.k8s.io/) tool can be used.

Ensure that the Kubeflow installation:

 - is accessible from the Python environment, i.e. the Kubeflow Python SDK can connect to the API
 
 - has at least one directory mounted as a `PersistentVolume` or a `hostPath`

 - has GPU access

When the above directory is mounted, make sure that you move every folder from the `Backend` folder inside that directory so that the three tasks are accessible from the Kubeflow pipelines.

Important Note: Make sure that inside this folder you have created a folder named `Datasets` where all the datasets will be stored. Inside the `Datasets` folder, create three subfolders named `Segmentation`, `Classification` and `Object-Detection` for each task respectively.

### Step 4 - Docker Images Setup
Machine learning models require appropriate environments to run on. Because the platform is Kubernetes-based, there is need for ready-to-use docker containers. 

1) For the semantic segmentation and the classification mode, to run the models, a Docker container based on PyTorch is needed.

First of all, download the basic image using the following command:

```bash
docker pull nvcr.io/nvidia/pytorch:22.12-py3
```

After pulling the base image, you have to create an updated image with all the necessary libraries installed. To do that, change your directory to `Backend/Segmentation/` and utilize the Dockerfile inside that folder by running the following command:

```bash
docker build -t segm_cls_image .
```

The Platform's backend dynamically creates containers on demand for each segmentation, classification and object detection task. These containers are single-use and they are instantiated only for the duration of the task execution and automatically destroyed upon completion.

2) For object detection task, you can use the official Ultralytics Docker image that has all the necessary libraries installed for running YOLO models.

To build this image locally, you can perform the following steps:

```bash
docker pull ultralytics/ultralytics:latest-python-export
```

<p align="center">
    <a href = "https://pytorch.org/" target="_blank">
        <img src="images/pytorch_logo.png" alt="Kubeflow" width="200"/>
    </a>
</p>

Important Note: Make sure that you keep the config.yml file updated inside `/AmalthAI_WebApp` folder with the correct image names and the shared directory path where the `Segmentation`, `Classification` and `ObjectDetection` folders are located.

### Step 5 - Upload docker images into kind cluster

To upload a locally built Docker image to your Kubernetes cluster, you have to run the following command:

```
kind load docker-image myimage:latest --name name-of-your-cluster
``` 

For the AmalthAI platform, you have to upload both the `segm_cls_image` and the `ultralytics/ultralytics:latest-python-export` images into the kind cluster.

### Step 6 - CVAT Installation
For the annotation purposes of this platform, we utilize [CVAT](https://www.cvat.ai/) annotation tool. To install it on your system, follow the instructions that are provided [here](https://docs.cvat.ai/docs/administration/basics/installation/).

<p align="center">
    <a href = "https://www.cvat.ai/" target="_blank">
        <img src="images/cvat_logo.png" alt="Kubeflow" width="200"/>
    </a>
</p>

### Step 7 - Platform UI Setup
For the platform's UI, you have to create a Docker container with the appropriate libraries to run the web app.

1) First open your terminal inside the `/AmalthAI_WebApp` folder and run:

```shell
docker build -t amalthai .
```

2) After the build is completed, you can run the container with the following command:
```shell
docker run --rm --network=host -v $(pwd):/app -v /var/run/docker.sock:/var/run/docker.sock -v /path/to/backend/shared/folder:/data -v ~/.kube/config:/root/.kube/config amalthai
```

Then you can easily navigate to the web app, by opening your browser and going to:
```
http://0.0.0.0:5000
```

## Documentation

Additional documentation can be found on the documentation portal [here](https://textailes.github.io/AmalthAI-documentation/).

## Citation

If you use this software, please cite it using the following BibTeX entry:

```bibtex
@software{Athena_Research_Center_AmalthAI_Machine_Learning_2025,
author = {{Athena Research Center}},
license = {MIT},
month = nov,
title = {{AmalthAI Machine Learning Platform}},
url = {https://github.com/TEXTaiLES/AmalthAI},
version = {0.1.0},
year = {2025}
}
```

## License

The code is released under the [MIT License](https://opensource.org/licenses/MIT).
