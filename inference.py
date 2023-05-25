"""
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "CHECKPOINT" \
    --device cuda \
    --input-source "input.mp4" \
    --output-type video \
    --output-composition "composition.mp4" \
    --output-alpha "alpha.mp4" \
    --output-foreground "foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1
"""

import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm

from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter, ImageReader, ConstantImage

def convert_video(model,
                  input_source: str,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  bgr_source: Optional[str] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_rec: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):
    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.

        bgr_source: A video file, image sequence directory, or an individual image.
            This is only applicable if you choose output_type == video.

        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """
    
    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground, output_rec]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'
    
    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),  # reverse (w, h)
            #transforms.Resize(input_resize[::-1], antialias=True),  # reverse (w, h)
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)
    
    # Initialize writers
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))

        if output_rec is not None:
            writer_rec = VideoWriter(
                path=output_rec,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')

        if output_rec is not None:
            writer_rec = ImageSequenceWriter(output_rec, 'png')

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
    
    if (output_composition is not None) and (output_type == 'video'):
        if bgr_source is not None:
            if os.path.isfile(bgr_source):
                if os.path.splitext(bgr_source)[-1].lower() in [".png", ".jpg"]:
                    bgr_raw = ImageReader(bgr_source, transform)
                else:
                    bgr_raw = VideoReader(bgr_source, transform)
            else:
                bgr_raw = ImageSequenceReader(bgr_source, transform)
        else:
            #bgr_raw = ConstantImage(120, 255, 155, device=device, dtype=dtype)
            bgr_raw = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
            #print(bgr_raw.shape())
        #bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)    # green screen background
    
    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)     # 进度条
            rec = [None] * 4
            #idx = 1
            #for src in reader:
            for index, src in enumerate(reader):

                if downsample_ratio is None:
                    downsample_ratio = auto_downsample_ratio(*src.shape[2:])

                #_, _, w, h = src.size()
                src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, W, H] 升维
                #print(type(src))
                #fgr, pha, res,  *rec = model(src, *rec, downsample_ratio)
                fgr, pha, *rec = model(src, *rec, downsample_ratio)

                """
                if(re.size(2) % 2 != 0):
                    re = re[:, :, :-1, :]
                if(re.size(3) % 2 != 0):
                    re = re[:, :, :, :-1]
                """
                #print(re.size())
                #if(idx == 1):
                    #print(fgr)
                    #idx += 1

                if output_foreground is not None:
                    writer_fgr.write(fgr[0])    # (T,C,H,W)
                    #print(fgr.size())
                if output_alpha is not None:
                    writer_pha.write(pha[0])    # (T,C,H,W)
                if output_composition is not None:
                    if output_type == 'video':
                        #bgr_raw.resize()
                        if bgr_source is not None:
                            if os.path.isfile(bgr_source):
                                bgr = bgr_raw[index].to(device, dtype, non_blocking=True).unsqueeze(0)
                            else:
                                bgr = bgr_raw[index].to(device, dtype, non_blocking=True).unsqueeze(0)  # [B, T, C, H, W] 升维
                        else:
                            #bgr = bgr_raw.to(device, dtype, non_blocking=True).unsqueeze(0)
                            bgr = bgr_raw
                        com = fgr * pha + bgr * (1 - pha)
                    else:
                        fgr = fgr * pha.gt(0)
                        #bgr = bgr_raw  # del
                        com = torch.cat([fgr, pha], dim=-3)     # RGBA png images
                        #com = fgr * pha + bgr * (1 - pha)
                    writer_com.write(com[0])

                #if output_rec is not None:
                    #writer_rec.write(re)
                    #print(re.size())
                    #print(rec[0].shape, rec[1].shape, rec[2].shape, rec[3].shape)
                #if output_rec is not None:
                #    writer_rec.write(res[0])
                
                bar.update(src.size(1))

    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


class Converter:
    #def __init__(self, variant: str, checkpoint: str, device: str):
    def __init__(self,  checkpoint: str, device: str):
        #self.model = MattingNetwork().eval().to(device)
        self.model = MattingNetwork().eval().to(device)
        #self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model.load_state_dict(torch.load(checkpoint, map_location=device),strict=True)
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        self.device = device
    
    def convert(self, *args, **kwargs):
        convert_video(self.model, device=self.device, dtype=torch.float32, *args, **kwargs)
    
if __name__ == '__main__':
    import argparse
    from model import MattingNetwork
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('--variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--input-source', type=str, required=True)
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--downsample-ratio', type=float)
    parser.add_argument('--bgr-source', type=str)
    parser.add_argument('--output-composition', type=str)
    parser.add_argument('--output-alpha', type=str)
    parser.add_argument('--output-foreground', type=str)
    parser.add_argument('--output-rec', type=str)
    parser.add_argument('--output-type', type=str, required=True, choices=['video', 'png_sequence'])
    parser.add_argument('--output-video-mbps', type=int, default=1)
    parser.add_argument('--seq-chunk', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--disable-progress', action='store_true')
    args = parser.parse_args()
    
    #converter = Converter(args.variant, args.checkpoint, args.device)
    converter = Converter(args.checkpoint, args.device)
    converter.convert(
        input_source=args.input_source,
        input_resize=args.input_resize,
        downsample_ratio=args.downsample_ratio,
        bgr_source=args.bgr_source,
        output_type=args.output_type,
        output_composition=args.output_composition,
        output_alpha=args.output_alpha,
        output_foreground=args.output_foreground,
        output_rec=args.output_rec,
        output_video_mbps=args.output_video_mbps,
        seq_chunk=args.seq_chunk,
        num_workers=args.num_workers,
        progress=not args.disable_progress
    )
    
    
