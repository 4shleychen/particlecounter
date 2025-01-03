from fastsam import FastSAM, FastSAMPrompt
import torch 

model = FastSAM('FastSAM.pt')
IMAGE_PATH = './images/dogs.jpg'
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
everything_results = model(
    IMAGE_PATH,
    device=DEVICE,
    retina_masks=True,
    imgsz=1024,
    conf=0.4,
    iou=0.9,
)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# # everything prompt
ann = prompt_process.everything_prompt()

ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])

prompt_process.plot(
    annotations=ann,
    output='./output/',
    mask_random_color=True,
    better_quality=True,
    retina=False,
    withContours=True,
)
