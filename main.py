import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from cross_processor import CrossAttnProcessor2_0, init_local_mask_flex
model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
height=768
width=1280
frame = 128
device = torch.device('cuda')
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16)
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()
#pipe.to(device)
prompt = "A sleek white yacht gliding across a crystal-blue sea at sunset, camera circles the vessel as golden light sparkles on gentle waves, slight lens distortion."
attenable = len(pipe.tokenizer(prompt)['input_ids'])
group_t, group_h, group_w = 4,8,8
mask = init_local_mask_flex(frame//4, height // 16, width // 16, text_length=256, attenable_text=attenable, group_t=group_t, group_h=group_h,group_w=group_w, device=device)
attn_processors = {}
for k,v in transformer.attn_processors.items():
    if "token_refiner" in k:
        attn_processors[k] = v
    else:
        attn_processors[k] = CrossAttnProcessor2_0(mask, frame//4, height // 16, width // 16, group_t, group_h, group_w,text_length=256)
transformer.set_attn_processor(attn_processors)
output = pipe(
    prompt=prompt,
    height=height,
    width=width,
    num_frames=frame,
    num_inference_steps=30
).frames[0]