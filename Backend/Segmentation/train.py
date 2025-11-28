import os
import json
import argparse
import torch
import dataloaders
import models
import inspect
import math
from utils import losses
from utils import Logger
from utils.torchsummary import summary
from trainer import Trainer

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def update_config(config_path, args):
    with open(config_path, 'r') as f:
        config = json.load(f)

    if args.lr is not None:
        config["optimizer"]["args"]["lr"] = args.lr
    if args.bs is not None:
        config["train_loader"]["args"]["batch_size"] = args.bs
        config["val_loader"]["args"]["batch_size"] = args.bs
    if args.epochs is not None:
        config["trainer"]["epochs"] = args.epochs
    if args.momentum is not None:
        config["optimizer"]["args"]["momentum"] = args.momentum
    if args.localpath is not None:
        config["trainer"]["localpath"]= args.localpath
    if args.dataset is not None:
        config["train_loader"]["args"]["data_dir"]= args.dataset
        config["val_loader"]["args"]["data_dir"]= args.dataset
        # palette_file update along with dataset
        config["train_loader"]["args"]["palette_file"] = os.path.join(args.dataset, "labelmap.txt")
        config["val_loader"]["args"]["palette_file"] = os.path.join(args.dataset, "labelmap.txt")
    if args.blur is not None:
        config["train_loader"]["args"]["blur"]= args.blur
    if args.rotate is not None:
        config["train_loader"]["args"]["rotate"]= args.rotate
    if args.flip is not None:
        config["train_loader"]["args"]["flip"]= args.flip
    if args.scale is not None:
        config["train_loader"]["args"]["scale"]= args.scale

    return config

def main(config, resume,mode):
    train_logger = Logger()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)

    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    print(f'\n{model}\n')

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    if mode == 'train':
        trainer.train()
    elif mode == 'val':
        val_log = trainer._valid_epoch(epoch=0)
        print("Validation metrics:", val_log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str, help='Path to model checkpoint')
    parser.add_argument('-d', '--device', default=None, type=str, help='GPU indices')
    parser.add_argument('--mode', default='train', choices=['train', 'val'], help='Mode: train or val')

    #CLI arguments to override config values / working with katib
    parser.add_argument('--lr', type=float, help='Learning rate override')
    parser.add_argument('--bs', type=int, help='Batch size override')
    parser.add_argument('--epochs', type=int, help='Number of epochs override')
    parser.add_argument('--momentum', type=float, help='Momentum override')
    parser.add_argument('--localpath', type = str, help = "Current Local Path") # comes from time 
    parser.add_argument('--dataset', type = str, help = "Dataset Path to start training")
    parser.add_argument('--blur', type=lambda x: x.lower() == 'true', help='Apply Gaussian Blur augmentation')
    parser.add_argument('--rotate', type=lambda x: x.lower() == 'true', help='Apply Random Rotation augmentation')
    parser.add_argument('--flip', type=lambda x: x.lower() == 'true', help='Apply Random Flip augmentation')
    parser.add_argument('--scale', type=lambda x: x.lower() == 'true', help='Apply Random Scale augmentation')

    args = parser.parse_args()

    config = update_config(args.config, args)  # Update config before loading
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume, args.mode)