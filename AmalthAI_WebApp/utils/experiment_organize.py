import yaml
import copy
import kubeflow.katib as katib
import random

'''
This file contains functions to conduct Katib experiments for different types of models: segmentation, object detection, and classification.
Each function creates a Katib experiment configuration, writes it to a YAML file, and then uses the Katib client to create and monitor the experiment.
'''

def conduct_experiment_seg(model_selection, timestamp_path, dataset, lr_left, lr_right, bs_left, bs_right, epochs_left, epochs_right, blur, scale, rotate, flip):
    allowed_bs = [2, 4, 8, 16, 32, 64]
    filtered_bs = [str(x) for x in allowed_bs if bs_left <= x <= bs_right]

    basic_segmentation_katib_experiment = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {
            "name": "random-experiment",
            "namespace": "kubeflow-user-example-com",
            "annotations": {
                "katib.kubeflow.org/metrics-collector-injection": "enabled"
            }
        },
        "spec": {
            "objective": {
                "type": "maximize",
                "goal": 0.4,
                "objectiveMetricName": "meaniou"
            },
            "algorithm": {
                "algorithmName": "random"
            },
            "parallelTrialCount": 1,
            "maxTrialCount": 1,
            "maxFailedTrialCount": 1,
            "parameters": [
                {
                    "name": "learning-rate",
                    "parameterType": "double",
                    "feasibleSpace": {"min": str(lr_left), "max": str(lr_right)}
                },
                {
                    "name": "batch-size",
                    "parameterType": "categorical",
                    "feasibleSpace": {"list": filtered_bs}
                },
                {
                    "name": "epochs",
                    "parameterType": "int",
                    "feasibleSpace": {"min": str(epochs_left), "max": str(epochs_right)}
                }
            ],
            "trialTemplate": {
                "primaryContainerName": "training-container",
                "trialParameters": [
                    {"name": "learningRate", "description": "Learning rate", "reference": "learning-rate"},
                    {"name": "batchSize", "description": "Batch size", "reference": "batch-size"},
                    {"name": "numEpochs", "description": "Epochs", "reference": "epochs"}
                ],
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "template": {
                            "metadata": {
                                "annotations": {"sidecar.istio.io/inject": "false"}
                            },
                            "spec": {
                                "containers": [
                                    {
                                        "name": "training-container",
                                        "image": "segmentation-utils:v1",
                                        "imagePullPolicy": "IfNotPresent",
                                        "command": [
                                            "python",
                                            "-u",
                                            "/segmentation/train.py",
                                            "--config", "/segmentation/config.json",
                                            "--lr", "${trialParameters.learningRate}",
                                            "--bs", "${trialParameters.batchSize}",
                                            "--epochs", "${trialParameters.numEpochs}",
                                            "--localpath", timestamp_path,
                                            "--dataset", dataset,
                                            "--blur", str(blur).lower(),
                                            "--rotate", str(rotate).lower(),
                                            "--flip", str(flip).lower(),
                                            "--scale", str(scale).lower()
                                        ],
                                        "volumeMounts": [
                                            {"mountPath": "/dev/shm", "name": "shm"},
                                            {"mountPath": "/segmentation", "name": "segmentation"},
                                            {"mountPath": "/segm_datasets", "name": "segmentationdat"},
                                        ]
                                    }
                                ],
                                "volumes": [
                                    {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}},
                                    {"name": "segmentation", "hostPath": {"path": "/host/Segmentation", "type": "Directory"}},
                                    {"name": "segmentationdat", "hostPath": {"path": "/host/Datasets/Segmentation", "type": "Directory"}}
                                ],
                                "restartPolicy": "OnFailure"
                            }
                        }
                    }
                }
            }
        }
    }

    config_file = f"/segmentation/configfiles/{model_selection.lower()}.json"

    unet_experiment = copy.deepcopy(basic_segmentation_katib_experiment)
    unet_experiment["metadata"]["name"] = f"{model_selection.lower()}-{random.randint(100, 9999)}"
    unet_experiment["spec"]["trialTemplate"]["trialSpec"]["spec"]["template"]["spec"]["containers"][0]["command"] = [
        "python",
        "-u",
        "/segmentation/train.py",
        "--config", config_file,
        "--lr", "${trialParameters.learningRate}",
        "--bs", "${trialParameters.batchSize}",
        "--epochs", "${trialParameters.numEpochs}",
        "--localpath", timestamp_path,
        "--dataset", dataset,
        "--blur", str(blur).lower(),
        "--rotate", str(rotate).lower(),
        "--flip", str(flip).lower(),
        "--scale", str(scale).lower()
    ]

    # Create YAML
    with open("experiment.yaml", "w") as f:
        yaml.dump(unet_experiment, f, sort_keys=False)

    # Perform Katib Experiment based on yaml
    with open("experiment.yaml", "r") as file:
        experiment_config = yaml.safe_load(file)

    katib_client = katib.KatibClient(namespace="kubeflow-user-example-com")
    katib_client.create_experiment(experiment=experiment_config)
    result = katib_client.wait_for_experiment_condition(
        name=experiment_config['metadata']['name']
    )

    conditions = result.status.conditions
    for condition in conditions:
        print(condition.type)

    # Get the last condition
    last_condition = conditions[-1] if conditions else None

    # Return the last status type
    last_status_type = last_condition.type if last_condition else None
    print(type(last_status_type))

    # Return the last status type
    return last_status_type

def conduct_experiment_od(model_selection, timestamp_path, dataset, lr_left, lr_right, bs_left, bs_right, epochs_left, epochs_right, hue, saturation, value, rotate, flip):
    basic_yolo_katib_experiment = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {
            "name": "random-experiment",
            "namespace": "kubeflow-user-example-com",
            "annotations": {
                "katib.kubeflow.org/metrics-collector-injection": "enabled"
            }
        },
        "spec": {
            "objective": {
                "type": "maximize",
                "goal": 0.4,
                "objectiveMetricName": "map5095"
            },
            "algorithm": {
                "algorithmName": "random"
            },
            "parallelTrialCount": 1,
            "maxTrialCount": 1,
            "maxFailedTrialCount": 1,
            "parameters": [
                {
                    "name": "learning-rate",
                    "parameterType": "double",
                    "feasibleSpace": {"min": str(lr_left), "max": str(lr_right)}
                },
                {
                    "name": "batch-size",
                    "parameterType": "int",
                    "feasibleSpace": {"min": str(bs_left), "max": str(bs_right)}
                },
                {
                    "name": "epochs",
                    "parameterType": "int",
                    "feasibleSpace": {"min": str(epochs_left), "max": str(epochs_right)}
                }
            ],
            "trialTemplate": {
                "primaryContainerName": "training-container",
                "trialParameters": [
                    {"name": "learningRate", "description": "Learning rate", "reference": "learning-rate"},
                    {"name": "batchSize", "description": "Batch size", "reference": "batch-size"},
                    {"name": "numEpochs", "description": "Epochs", "reference": "epochs"}
                ],
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "template": {
                            "metadata": {
                                "annotations": {"sidecar.istio.io/inject": "false"}
                            },
                            "spec": {
                                "containers": [
                                    {
                                        "name": "training-container",
                                        "image": "ultralytics/ultralytics:latest",
                                        "imagePullPolicy": "IfNotPresent",
                                        "command": [
                                            "python",
                                            "-u",
                                            "/yolo/ultralytics/core_train.py",
                                            "--epochs", "${trialParameters.numEpochs}",
                                            "--bs", "${trialParameters.batchSize}",
                                            "--lr", "${trialParameters.learningRate}",
                                            "--dataset", dataset,
                                            "--timestamp", timestamp_path,
                                            "--hue", str(hue),
                                            "--saturation", str(saturation),
                                            "--value", str(value),
                                            "--rotate", str(rotate),
                                            "--flip", str(flip)
                                        ],
                                        "volumeMounts": [
                                            {"mountPath": "/dev/shm", "name": "shm"},
                                            {"mountPath": "/yolo", "name": "yolo"},
                                            {"mountPath": "/objdet_datasets", "name": "objdetdat"}
                                        ]
                                    }
                                ],
                                "volumes": [
                                    {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}},
                                    {"name": "yolo", "hostPath": {"path": "/host/ObjectDetection", "type": "Directory"}},
                                    {"name": "objdetdat", "hostPath": {"path": "/host/Datasets/Object-Detection", "type": "Directory"}}
                                ],
                                "restartPolicy": "OnFailure"
                            }
                        }
                    }
                }
            }
        }
    }
    config_file = f"/yolo/ultralytics/{model_selection.lower()}.py"
    
    yolo_experiment = copy.deepcopy(basic_yolo_katib_experiment)
    yolo_experiment["metadata"]["name"] = f"{model_selection.lower()}-{random.randint(100, 9999)}" 
    yolo_experiment["spec"]["trialTemplate"]["trialSpec"]["spec"]["template"]["spec"]["containers"][0]["command"] = [
        "python",
        "-u",
        config_file,
        "--epochs", "${trialParameters.numEpochs}",
        "--bs", "${trialParameters.batchSize}",
        "--lr", "${trialParameters.learningRate}",
        "--dataset", dataset,
        "--timestamp", timestamp_path,
        "--hue", str(hue),
        "--saturation", str(saturation),
        "--value", str(value),
        "--rotate", str(rotate),
        "--flip", str(flip)
    ]

    # Create YAML
    with open("experiment.yaml", "w") as f:
        yaml.dump(yolo_experiment, f, sort_keys=False)

    # Perform Katib Experiment based on yaml
    with open("experiment.yaml", "r") as file:
        experiment_config = yaml.safe_load(file)

    katib_client = katib.KatibClient(namespace="kubeflow-user-example-com")
    katib_client.create_experiment(experiment=experiment_config)
    result = katib_client.wait_for_experiment_condition(
        name=experiment_config['metadata']['name']
    )

    conditions = result.status.conditions
    for condition in conditions:
        print(condition.type)

    # Get the last condition
    last_condition = conditions[-1] if conditions else None

    # Return the last status type
    last_status_type = last_condition.type if last_condition else None
    print(type(last_status_type))

    # Return the last status type
    return last_status_type

def conduct_experiment_cls(model_selection, timestamp_path, dataset, lr_left, lr_right, bs_left, bs_right, epochs_left, epochs_right, blur, rotate, flip):
    basic_classification_katib_experiment = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {
            "name": "random-experiment",
            "namespace": "kubeflow-user-example-com",
            "annotations": {
                "katib.kubeflow.org/metrics-collector-injection": "enabled"
            }
        },
        "spec": {
            "objective": {
                "type": "maximize",
                "goal": 90,
                "objectiveMetricName": "accuracy"
            },
            "algorithm": {
                "algorithmName": "random"
            },
            "parallelTrialCount": 1,
            "maxTrialCount": 1,
            "maxFailedTrialCount": 1,
            "parameters": [
                {
                    "name": "learning-rate",
                    "parameterType": "double",
                    "feasibleSpace": {"min": str(lr_left), "max": str(lr_right)}
                },
                {
                    "name": "batch-size",
                    "parameterType": "int",
                    "feasibleSpace": {"min": str(bs_left), "max": str(bs_right)}
                },
                {
                    "name": "epochs",
                    "parameterType": "int",
                    "feasibleSpace": {"min": str(epochs_left), "max": str(epochs_right)}
                }
            ],
            "trialTemplate": {
                "primaryContainerName": "training-container",
                "trialParameters": [
                    {"name": "learningRate", "description": "Learning rate", "reference": "learning-rate"},
                    {"name": "batchSize", "description": "Batch size", "reference": "batch-size"},
                    {"name": "numEpochs", "description": "Epochs", "reference": "epochs"}
                ],
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "template": {
                            "metadata": {
                                "annotations": {"sidecar.istio.io/inject": "false"}
                            },
                            "spec": {
                                "containers": [
                                    {
                                        "name": "training-container",
                                        "image": "segmentation-utils:v1",
                                        "imagePullPolicy": "IfNotPresent",
                                        "command": [
                                            "python",
                                            "-u",
                                            "/classification/train.py",
                                            "--dataset", dataset,
                                            "--model", model_selection,
                                            "--epochs", "${trialParameters.numEpochs}",
                                            "--batch_size", "${trialParameters.batchSize}",
                                            "--lr", "${trialParameters.learningRate}",
                                            "--save_path", timestamp_path,
                                            "--blur", blur,
                                            "--rotate", rotate,
                                            "--flip", flip
                                        ],
                                        "volumeMounts": [
                                            {"mountPath": "/dev/shm", "name": "shm"},
                                            {"mountPath": "/classification", "name": "classification"},
                                            {"mountPath": "/class_datasets", "name": "classdat"}
                                        ]
                                    }
                                ],
                                "volumes": [
                                    {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}},
                                    {"name": "classification", "hostPath": {"path": "/host/Classification", "type": "Directory"}},
                                    {"name": "classdat", "hostPath": {"path": "/host/Datasets/Classification", "type": "Directory"}}
                                ],
                                "restartPolicy": "OnFailure"
                            }
                        }
                    }
                }
            }
        }
    }


    classification_experiment = copy.deepcopy(basic_classification_katib_experiment)
    classification_experiment["metadata"]["name"] = f"{model_selection.lower()}-{random.randint(100, 9999)}"
    classification_experiment["spec"]["trialTemplate"]["trialSpec"]["spec"]["template"]["spec"]["containers"][0]["command"] = [
        "python",
        "-u",
        "/classification/train.py",
        "--dataset", dataset,
        "--model", model_selection,
        "--epochs", "${trialParameters.numEpochs}",
        "--batch_size", "${trialParameters.batchSize}",
        "--lr", "${trialParameters.learningRate}",
        "--save_path", timestamp_path,
        "--blur", str(blur).lower(),
        "--rotate", str(rotate).lower(),
        "--flip", str(flip).lower()
    ]

    # Create YAML
    with open("experiment.yaml", "w") as f:
        yaml.dump(classification_experiment, f, sort_keys=False)

    # Perform Katib Experiment based on yaml
    with open("experiment.yaml", "r") as file:
        experiment_config = yaml.safe_load(file)

    katib_client = katib.KatibClient(namespace="kubeflow-user-example-com")
    katib_client.create_experiment(experiment=experiment_config)
    result = katib_client.wait_for_experiment_condition(
        name=experiment_config['metadata']['name']
    )

    conditions = result.status.conditions
    for condition in conditions:
        print(condition.type)

    # Get the last condition
    last_condition = conditions[-1] if conditions else None

    # Return the last status type
    last_status_type = last_condition.type if last_condition else None
    print(type(last_status_type))

    # Return the last status type
    return last_status_type