# AmalthAI Machine Learning Platform

<p align="center">
    <a href="https://amalthai-ml.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue" /></a>
    <a href="https://arxiv.org/abs/2608.13343"><img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv" /></a>
    <a href="https://textailes.github.io/AmalthAI-documentation/"><img src="https://img.shields.io/badge/Docs-Portal-orange?logo=readthedocs" /></a>
    <a href="https://opensource.org/license/agpl-v3"><img src="https://img.shields.io/badge/License-AGPL--3.0-green?logo=gnu" /></a>
</p>
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
 - RAM: at least 32 GB
 - GPU: NVIDIA GPU with at least 16 GB VRAM
 - OS: Ubuntu 22.04 or newer
 - Browser: Google Chrome or Microsoft Edge

### Before You Start
Make sure that you have cloned the following repository on your local machine:

```bash
git clone https://github.com/TEXTaiLES/AmalthAI
```
- The `AmalthAI_WebApp` folder contains the web app code, which is necessary for the platform's UI. 

- The `Backend` folder contains all the necessary code for the platform's functionality which includes machine learning models, data processing scripts, and deployment configurations.

### Step 1 - Docker Installation
Make sure that you have a local installation of [Docker](https://www.docker.com/). You can follow the installation process described [here](https://docs.docker.com/engine/install/ubuntu/):

<p align="center">
    <a href = "https://docs.docker.com/engine/install/ubuntu/" target="_blank">
        <img src="images/docker_logo.png" alt="Docker" width="200"/>
    </a>
</p>

### Step 2 - kubectl and kind Installation
Instead of a full Kubernetes installation, this platform uses kind (Kubernetes in Docker) for local cluster management. For kubectl installation, follow the instructions [here](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/). For kind installation, follow the instructions [here](https://kind.sigs.k8s.io/docs/user/quick-start/#installation).

<p align="center">
    <a href = "https://kubernetes.io/docs/setup/" target="_blank">
        <img src="images/kubernetes_logo.png" alt="Kubernetes" width="200"/>
    </a>
</p>

### Step 3 - Cluster Setup and Katib Installation

**Create the kind cluster:**
```bash
kind create cluster --name=kubeflow --config=cluster-config.yml
docker exec -ti kubeflow-control-plane ln -s /sbin/ldconfig /sbin/ldconfig.real
```
The `cluster-config.yml` file can be found under the `Backend` folder.

**Install Helm:**
```bash
sudo snap install helm
```

Before you continue, make sure to install the NVIDIA toolkit on the local machine to establish connection with GPU resources.

**Install NVIDIA GPU Operator:**
```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia || true
helm repo update
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator --set driver.enabled=false
```

**Install Katib (standalone):**
```bash
kubectl apply -k "github.com/kubeflow/katib.git/manifests/v1beta1/installs/katib-standalone?ref=v0.17.0"
```

Also, ensure that the cluster:

 - has at least one directory shared between the cluster and the local machine (located inside the `config.yml` file under `extraMounts` as `containerPath` and `hostPath` respectively).

When the above directory is mounted, make sure that you move every folder from the `Backend` folder inside that directory so that the three tasks are accessible from the Katib pipelines.

<p align="center">
    <a href = "https://www.kubeflow.org/" target="_blank">
        <img src="images/kubeflow_logo.png" alt="Kubeflow" width="200"/>
    </a>
</p>

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
docker pull ultralytics/ultralytics:8.4.112
```

<p align="center">
    <a href = "https://pytorch.org/" target="_blank">
        <img src="images/pytorch_logo.png" alt="PyTorch" width="200"/>
    </a>
</p>

### Step 5 - Upload docker images into kind cluster

To upload a locally built Docker image to your Kubernetes cluster, you have to run the following command:

```
kind load docker-image myimage:latest --name name-of-your-cluster
``` 

For the AmalthAI platform, you have to upload both the `segm_cls_image` and the `ultralytics/ultralytics:latest` images into the kind cluster.

### Step 6 - CVAT Installation
For the annotation purposes of this platform, we utilize [CVAT](https://www.cvat.ai/) annotation tool. To install it on your system, follow the instructions that are provided [here](https://docs.cvat.ai/docs/administration/basics/installation/).

<p align="center">
    <a href = "https://www.cvat.ai/" target="_blank">
        <img src="images/cvat_logo.png" alt="CVAT" width="200"/>
    </a>
</p>

### Step 7 - VLM Installation

For the platform's VLM-based inference (visual description and misclassification explanation), the platform uses [vLLM](https://github.com/vllm-project/vllm) to serve Qwen2-VL-2B-Instruct.

First, pull the vLLM OpenAI-compatible server image:

```bash
docker pull vllm/vllm-openai:latest
```

Then, start the container on the `textailes` Docker network:

```bash
docker run --name qwen-vlm \
  --network textailes \
  --gpus all \
  --restart always \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --shm-size=4g \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384
```

This exposes an OpenAI-compatible API on port `8000` that the AmalthAI backend uses as the VLM inference endpoint.

### Step 8 - Platform UI Setup

Before launching the application, both the `config.yml` and `docker-compose.yml` files in the `/AmalthAI_WebApp` folder must be configured according to your local environment.

#### `config.yml` Configuration

Open `/AmalthAI_WebApp/config.yml` and verify the following settings:

* The Docker image names under `images` are correct.
* `base_host_path_out` points to the host directory containing the `Segmentation`, `Classification`, and `ObjectDetection` folders.
* `base_host_path` points to the corresponding data directory **inside the container**.
* The `chown_uid` and `chown_gid` values match the user and group IDs of the host machine.
* The required service URLs, tokens, and other configuration parameters are correctly set.

For example:

```yaml
paths:
  base_host_path: /data
  base_host_path_out: /home/user/kubeflow/kind-data
```

In this example, `/home/user/kubeflow/kind-data` is the directory on the **host machine** that contains the three task folders:

```text
/home/user/kubeflow/kind-data/
├── Segmentation/
├── Classification/
└── ObjectDetection/
```

#### `docker-compose.yml` Configuration

The `docker-compose.yml` file must also be updated to match the paths and services configured above.

In particular, the host path mounted to `/data` **must correspond to `base_host_path_out` in `config.yml`**.

For example, if `config.yml` contains:

```yaml
base_host_path_out: /home/user/kubeflow/kind-data
```

then `docker-compose.yml` should contain:

```yaml
volumes:
  - /home/user/kubeflow/kind-data:/data
```

This mapping makes the same host directory available inside the AmalthAI container at `/data`.

Also verify that:

* The remaining host paths under `volumes` are correct for your machine.
* The Kubernetes configuration is correctly mounted through `/root/.kube/`.
* `KUBECONFIG` points to the correct Kubernetes configuration file.
* `HESTIA_BASE_URL` and `HESTIA_API_KEY` contain the correct Hestia configuration.
* `AMALTHAI_URL` is replaced with the URL/domain used to access the application.
* The external `textailes` Docker network exists on the host machine.

After configuring both files, build the AmalthAI Docker image.

1) Open a terminal inside the `/AmalthAI_WebApp` folder and run:

```bash
docker build -t amalthai .
```

2) After the build is completed, start the application using Docker Compose:

```bash
docker compose up -d
```

3) Verify that the container is running:

```bash
docker ps
```

4) The application can then be accessed through the configured `AMALTHAI_URL`. For a local installation using the default port, open:

```text
http://0.0.0.0:8056
```

## Documentation

Additional documentation can be found on the documentation portal [here](https://textailes.github.io/AmalthAI-documentation/).

## Citation

If you use this software, please cite it using the following BibTeX entry:

```bibtex
@article{chatzisavvas2026amalthai,
  title={AmalthAI: An Open-Source Computer Vision Platform for Cultural Heritage},
  author={Chatzisavvas, Christos and Alvanos, Stelios and Politis, Efstratios and Rigas, Panagiotis and Pappas, Thomas and Giannoukos, Ioannis and Mitianoudis, Nikolaos and Ulanowska, Agata and Margariti, Christina and Pavlidis, George and others},
  journal={arXiv preprint arXiv:2608.13343},
  year={2026}
}
```

## License

This project is licensed under the AGPL-3.0 License. For more details, please refer to the [LICENSE](https://opensource.org/license/agpl-v3).
