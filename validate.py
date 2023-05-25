import argparse
import torch
import random
import os
from torch import nn
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms.functional import center_crop
from tqdm import tqdm

from dataset.videomatte import (
    VideoMatteDataset,
    VideoMatteTrainAugmentation,
    VideoMatteValidAugmentation
)
from dataset.imagematte import (
    ImageMatteDataset,
    ImageMatteAugmentation
)
from dataset.coco import (
    CocoPanopticDataset,
    CocoPanopticTrainAugmentation,
)
from dataset.spd import (
    SuperviselyPersonDataset
)
from dataset.youtubevis import (
    YouTubeVISDataset,
    YouTubeVISAugmentation
)
from dataset.augmentation import (
    TrainFrameSampler,
    ValidFrameSampler
)

from model import MattingNetwork
from train_config import DATA_PATHS
from train_loss import matting_loss, segmentation_loss

class Trainer:
    def __init__(self, rank, world_size):
        self.parse_args()
        self.init_distributed(rank, world_size)
        self.init_datasets()
        self.init_model()
        self.init_writer()
        self.train()
        self.cleanup()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Model
        #parser.add_argument('--model-variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
        # Matting dataset
        parser.add_argument('--dataset', type=str, required=True, choices=['videomatte', 'imagematte'])
        # Learning rate
        #parser.add_argument('--learning-rate-backbone', type=float, required=True)
        #parser.add_argument('--learning-rate-aspp', type=float, required=True)
        #parser.add_argument('--learning-rate-decoder', type=float, required=True)
        #parser.add_argument('--learning-rate-refiner', type=float, required=True)
        # Training setting
        parser.add_argument('--train-hr', action='store_true')
        parser.add_argument('--resolution-lr', type=int, default=512)
        parser.add_argument('--resolution-hr', type=int, default=2048)
        parser.add_argument('--seq-length-lr', type=int, required=True)
        parser.add_argument('--seq-length-hr', type=int, default=6)
        parser.add_argument('--downsample-ratio', type=float, default=0.25)
        parser.add_argument('--batch-size-per-gpu', type=int, default=1)
        parser.add_argument('--num-workers', type=int, default=8)
        #parser.add_argument('--epoch-start', type=int, default=0)
        #parser.add_argument('--epoch-end', type=int, default=16)
        # Tensorboard logging
        parser.add_argument('--log-dir', type=str, required=True)
        parser.add_argument('--log-train-loss-interval', type=int, default=20)
        parser.add_argument('--log-train-images-interval', type=int, default=500)
        # Checkpoint loading and saving
        parser.add_argument('--checkpoint', type=str)
        #parser.add_argument('--checkpoint-dir', type=str, required=True)
        #parser.add_argument('--checkpoint-save-interval', type=int, default=500)
        # Distributed
        parser.add_argument('--distributed-addr', type=str, default='localhost')
        parser.add_argument('--distributed-port', type=str, default='12355')
        # Debugging
        parser.add_argument('--disable-progress-bar', action='store_true')
        parser.add_argument('--disable-validation', action='store_true')
        parser.add_argument('--disable-mixed-precision', action='store_true')
        self.args = parser.parse_args()


    def init_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.log('Initializing distributed')
        os.environ['MASTER_ADDR'] = self.args.distributed_addr
        os.environ['MASTER_PORT'] = self.args.distributed_port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def init_datasets(self):
        self.log('Initializing matting datasets')
        size_hr = (self.args.resolution_hr, self.args.resolution_hr)
        size_lr = (self.args.resolution_lr, self.args.resolution_lr)

        # Matting datasets:
        if self.args.dataset == 'videomatte':
            self.dataset_lr_train = VideoMatteDataset(
                videomatte_dir=DATA_PATHS['videomatte']['train'],
                background_image_dir=DATA_PATHS['background_images']['train'],
                background_video_dir=DATA_PATHS['background_videos']['train'],
                size=self.args.resolution_lr,
                seq_length=self.args.seq_length_lr,
                seq_sampler=TrainFrameSampler(),
                transform=VideoMatteTrainAugmentation(size_lr))
            if self.args.train_hr:
                self.dataset_hr_train = VideoMatteDataset(
                    videomatte_dir=DATA_PATHS['videomatte']['train'],
                    background_image_dir=DATA_PATHS['background_images']['train'],
                    background_video_dir=DATA_PATHS['background_videos']['train'],
                    size=self.args.resolution_hr,
                    seq_length=self.args.seq_length_hr,
                    seq_sampler=TrainFrameSampler(),
                    transform=VideoMatteTrainAugmentation(size_hr))
            self.dataset_valid = VideoMatteDataset(
                videomatte_dir=DATA_PATHS['videomatte']['valid'],
                background_image_dir=DATA_PATHS['background_images']['valid'],
                background_video_dir=DATA_PATHS['background_videos']['valid'],
                size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
                seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
                seq_sampler=ValidFrameSampler(),
                transform=VideoMatteValidAugmentation(size_hr if self.args.train_hr else size_lr))
        else:
            self.dataset_lr_train = ImageMatteDataset(
                imagematte_dir=DATA_PATHS['imagematte']['train'],
                background_image_dir=DATA_PATHS['background_images']['train'],
                background_video_dir=DATA_PATHS['background_videos']['train'],
                size=self.args.resolution_lr,
                seq_length=self.args.seq_length_lr,
                seq_sampler=TrainFrameSampler(),
                transform=ImageMatteAugmentation(size_lr))
            if self.args.train_hr:
                self.dataset_hr_train = ImageMatteDataset(
                    imagematte_dir=DATA_PATHS['imagematte']['train'],
                    background_image_dir=DATA_PATHS['background_images']['train'],
                    background_video_dir=DATA_PATHS['background_videos']['train'],
                    size=self.args.resolution_hr,
                    seq_length=self.args.seq_length_hr,
                    seq_sampler=TrainFrameSampler(),
                    transform=ImageMatteAugmentation(size_hr))
            self.dataset_valid = ImageMatteDataset(
                imagematte_dir=DATA_PATHS['imagematte']['valid'],
                background_image_dir=DATA_PATHS['background_images']['valid'],
                background_video_dir=DATA_PATHS['background_videos']['valid'],
                size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
                seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
                seq_sampler=ValidFrameSampler(),
                transform=ImageMatteAugmentation(size_hr if self.args.train_hr else size_lr))

        # Matting dataloaders:
        self.datasampler_lr_train = DistributedSampler(
            dataset=self.dataset_lr_train,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)
        self.dataloader_lr_train = DataLoader(
            dataset=self.dataset_lr_train,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            sampler=self.datasampler_lr_train,
            pin_memory=True)
        if self.args.train_hr:
            self.datasampler_hr_train = DistributedSampler(
                dataset=self.dataset_hr_train,
                rank=self.rank,
                num_replicas=self.world_size,
                shuffle=True)
            self.dataloader_hr_train = DataLoader(
                dataset=self.dataset_hr_train,
                batch_size=self.args.batch_size_per_gpu,
                num_workers=self.args.num_workers,
                sampler=self.datasampler_hr_train,
                pin_memory=True)
        self.dataloader_valid = DataLoader(
            dataset=self.dataset_valid,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            pin_memory=True)

        # Segementation datasets
        self.log('Initializing image segmentation datasets')
        """
        self.dataset_seg_image = ConcatDataset([
            CocoPanopticDataset(
                imgdir=DATA_PATHS['coco_panoptic']['imgdir'],
                anndir=DATA_PATHS['coco_panoptic']['anndir'],
                annfile=DATA_PATHS['coco_panoptic']['annfile'],
                transform=CocoPanopticTrainAugmentation(size_lr)),
            SuperviselyPersonDataset(
                imgdir=DATA_PATHS['spd']['imgdir'],
                segdir=DATA_PATHS['spd']['segdir'],
                transform=CocoPanopticTrainAugmentation(size_lr))
        ])
        """
        self.dataset_seg_image = SuperviselyPersonDataset(
                imgdir=DATA_PATHS['spd']['imgdir'],
                segdir=DATA_PATHS['spd']['segdir'],
                transform=CocoPanopticTrainAugmentation(size_lr))
        self.datasampler_seg_image = DistributedSampler(
            dataset=self.dataset_seg_image,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)
        self.dataloader_seg_image = DataLoader(
            dataset=self.dataset_seg_image,
            batch_size=self.args.batch_size_per_gpu * self.args.seq_length_lr,
            num_workers=self.args.num_workers,
            sampler=self.datasampler_seg_image,
            pin_memory=True)


        self.log('Initializing video segmentation datasets')
        self.dataset_seg_video = YouTubeVISDataset(
            videodir=DATA_PATHS['youtubevis']['videodir'],
            annfile=DATA_PATHS['youtubevis']['annfile'],
            size=self.args.resolution_lr,
            seq_length=self.args.seq_length_lr,
            seq_sampler=TrainFrameSampler(speed=[1]),
            transform=YouTubeVISAugmentation(size_lr))
        self.datasampler_seg_video = DistributedSampler(
            dataset=self.dataset_seg_video,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)
        self.dataloader_seg_video = DataLoader(
            dataset=self.dataset_seg_video,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            sampler=self.datasampler_seg_video,
            pin_memory=True)


    def init_model(self):
        self.log('Initializing model')
        #self.model = MattingNetwork(self.args.model_variant, pretrained_backbone=True).to(self.rank)
        self.model = MattingNetwork(pretrained_backbone=False).to(self.rank)

        if self.args.checkpoint:
            self.log(f'Restoring from checkpoint: {self.args.checkpoint}')
            self.log(self.model.load_state_dict(
                torch.load(self.args.checkpoint, map_location=f'cuda:{self.rank}')))
        """
        sr_weights = self.model.backbone.block1[0].attn.sr.weight
        sr_bias = self.model.backbone.block1[0].attn.sr.bias
        norm_weights = self.model.backbone.block1[0].attn.norm.weight
        norm_bias = self.model.backbone.block1[0].attn.norm.bias

        print("sr_weight:")
        print(sr_weights)
        print("sr_bias:")
        print(sr_bias)
        print("norm_weight:")
        print(norm_weights)
        print("norm_bias:")
        print(norm_bias)
        """
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.model_ddp = DDP(self.model, device_ids=[self.rank], broadcast_buffers=False, find_unused_parameters=True)
        """
        self.optimizer = Adam([
            {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone},
            #{'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp},
            {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
            {'params': self.model.project_mat.parameters(), 'lr': self.args.learning_rate_decoder},
            {'params': self.model.project_seg.parameters(), 'lr': self.args.learning_rate_decoder},
            {'params': self.model.refiner.parameters(), 'lr': self.args.learning_rate_refiner},
        ])
        self.scaler = GradScaler()
        """

    def init_writer(self):
        if self.rank == 0:
            self.log('Initializing writer')
            self.writer = SummaryWriter(self.args.log_dir)

    def train(self):

            self.epoch = 1
            self.step = 1 * len(self.dataloader_lr_train)

            if not self.args.disable_validation:
                self.validate()
            torch.cuda.empty_cache()


    def validate(self):

        if self.rank == 0:
            self.log(f'Validating at the start of epoch: {self.epoch}')
            self.model_ddp.eval()
            total_loss, total_count = 0, 0
            with torch.no_grad():
                with autocast(enabled=not self.args.disable_mixed_precision):
                    for true_fgr, true_pha, true_bgr in tqdm(self.dataloader_valid,
                                                             disable=self.args.disable_progress_bar,
                                                             dynamic_ncols=True):
                        true_fgr = true_fgr.to(self.rank, non_blocking=True)
                        true_pha = true_pha.to(self.rank, non_blocking=True)
                        true_bgr = true_bgr.to(self.rank, non_blocking=True)
                        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
                        batch_size = true_src.size(0)
                        pred_fgr, pred_pha = self.model(true_src)[:2]
                        total_loss += matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)['total'].item() * batch_size
                        total_count += batch_size
            avg_loss = total_loss / total_count
            self.log(f'Validation set average loss: {avg_loss}')
            self.writer.add_scalar('valid_loss', avg_loss, self.step)
            self.model_ddp.train()
        dist.barrier()

    def cleanup(self):
        dist.destroy_process_group()

    def log(self, msg):
        print(f'[GPU{self.rank}] {msg}')

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(
        Trainer,
        nprocs=world_size,
        args=(world_size,),
        join=True)