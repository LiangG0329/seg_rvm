import torch
import cv2
import numpy
from torchvision import transforms
from PIL import Image

from inference_utils import ImageReader

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 最小分辨率 320x240 ?
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

cam_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
cam_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

# 加载模型和权重
#model = torch.hub.load('senguptaumd/Background-Matting', 'vitb_rn50_384')
from model import MattingNetwork
from model import MattingNetwork_

#model = MattingNetwork(variant="mobilenetv3").eval().cuda()

#model = MattingNetwork(pretrained_backbone=False).eval().cuda()
model = MattingNetwork_(pretrained_backbone=False).eval()

model.load_state_dict(
    #torch.load('work_dirs/rvm_mobilenetv3.pth', map_location='cuda')
    #torch.load('checkpoint/stage2_2/epoch-27.pth',map_location='cpu')
    torch.load('model_2/2/checkpoint/stage3/epoch-23.pth',map_location='cpu')
    #torch.load('checkpoint/stage2_2/epoch-27.pth',map_location='cuda')
)
#model.eval()
#.cuda()


rec = [None] * 4  # 初始记忆
#bgr = torch.tensor([0.47, 1, 0.6]).view(3, 1, 1).cuda()  # 绿色背景
bgr = torch.tensor([0.47, 1, 0.6]).view(3, 1, 1)   # 绿色背景

bgr_source = r"bgr_source/img/landscape_232.flat-landscape.jpg"
input_size = (320, 240)
transform = transforms.Compose([
            transforms.Resize(input_size[::-1]),  # reverse (w, h)
            #transforms.Resize(input_resize[::-1], antialias=True),  # reverse (w, h)
            transforms.ToTensor()
        ])
bgr_img = Image.open(bgr_source)   #自定义背景
#bgr_img =  ImageReader(bgr_source, transform)
#to_tensor = transforms.ToTensor()
bgr_tensor = transform(bgr_img)

loader = transforms.ToTensor()
#loader = transform

while True:
    with torch.no_grad():
        ret, frame = capture.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #src = loader(numpy.asarray(frame)).to("cuda", torch.float32).unsqueeze(0)
        src = loader(numpy.asarray(frame)).to(torch.float32).unsqueeze(0)

        fgr, pha, *rec = model(src, *rec, downsample_ratio=0.85)

        #com = fgr * pha + bgr * (1 - pha)  # 绿色背景
        #for i in range(0, 1):
        #    print(com.shape)
        com = fgr * pha + bgr_tensor * (1 - pha)  #自定义背景
        com = com.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
        com = cv2.cvtColor(com, cv2.COLOR_RGB2BGR)

        #fgr = fgr.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
        #fgr = cv2.cvtColor(fgr, cv2.COLOR_RGB2BGR)

        #cv2.namedWindow("com", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("fgr", cv2.WINDOW_NORMAL)

        cv2.imshow("com", com)
        #cv2.imshow("fgr", fgr)

    # not working
        #torch.cuda.empty_cache()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()