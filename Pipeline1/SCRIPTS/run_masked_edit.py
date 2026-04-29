import os
os.environ['SPCONV_ALGO'] = 'native'

import imageio
import torch
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils
from masked_slat_edit import GroundedSAMSegmenter, run_and_save, run_edit_region

OUTPUT_DIR = "/mnt/data/srihari/my_TRELLIS/ALL_OUTPUTS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline  = TrellisTextTo3DPipeline.from_pretrained("/mnt/data/srihari/MODELS/TRELLIS-text-xlarge")
pipeline.cuda()
segmenter = GroundedSAMSegmenter(device="cuda")

result_orig = run_and_save(
    pipeline,
    prompt  = "a girl with black dress",
    formats = ['gaussian'],
    seed    = 42,
)
video = render_utils.render_video(result_orig['gaussian'][0])['color']
imageio.mimwrite(os.path.join(OUTPUT_DIR, "girl_orig.mp4"), video, fps=30)
print("saved original")

for s in [0.3, 0.5, 0.7, 1.0]:
    result_edit = run_edit_region(
        pipeline,
        segmenter   = segmenter,
        edit_region = "hair",
        prompt_edit = "a girl with golden hair and red dress",
        strength    = s,
        seed        = 42,
        num_views   = 8,
        min_votes   = 1,
        formats     = ['gaussian'],
    )
    video = render_utils.render_video(result_edit['gaussian'][0])['color']
    imageio.mimwrite(os.path.join(OUTPUT_DIR, f"girl_golden_hair_s{int(s*10):02d}.mp4"), video, fps=30)
    print(f"saved strength={s}")