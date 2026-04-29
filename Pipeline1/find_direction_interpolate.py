import numpy as np
import os
import imageio
from PIL import Image
import cv2

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

os.environ['SPCONV_ALGO'] = 'native'


IMAGE_PAIRS = [
    (
        "slim_to_thick",
        "INPUT_IMAGE_PAIRS/slim.png",
        "INPUT_IMAGE_PAIRS/thick.png",
    ),
    (
        "squeeze_down",
        "INPUT_IMAGE_PAIRS/scale1.jpeg",
        "INPUT_IMAGE_PAIRS/scale2.jpeg",
    ),
    (
        "cyl_to_cube",
        "INPUT_IMAGE_PAIRS/cylinder.jpeg",
        "INPUT_IMAGE_PAIRS/cuboidal_cylinder.jpeg",
    ),
    (
        "cyl_to_timer",
        "INPUT_IMAGE_PAIRS/cyl_normal_for_timer.png",
        "INPUT_IMAGE_PAIRS/timer_cyl.png",
    ),
    (
        "thin_to_thick_leg_chair",
        "INPUT_IMAGE_PAIRS/thin_leg_chair.png",
        "INPUT_IMAGE_PAIRS/thick_leg_chair.png",
    ),
    (
        "carve_out_center",
        "INPUT_IMAGE_PAIRS/thick_for_carve.png",
        "INPUT_IMAGE_PAIRS/carved.png",
    ),
]

PAIR_IDX = 3

ASSETS = [
    "f8d3ae1aaa1b579bc9206689365aaef6dbd7b1153a1c7665e845998fbe9def05.png",
    "e6321a36e7203ebe2659de41229aef4db3f17645e44f2da48b9d52c20fa94546.png",
    "cb7a9b86c2f528efabb56b3b45cf214dbf5e5c9dcc7307d8ed1e94ac233fe94a.png",
    "ddc13b45aee4a015116fa320a5647d982741ea8963b43e72f7326509fdc1fbc4.png",
    "adbaf3855efce3770e16bdca23aadbe6e9e1eaeb5015684a7759329c4aed0878.png",
    "bd5634345a4cd53b5d5c2eb9179b753a27ef2f9ef82cc6ba7171a1cae2614c04.png",
    "49356c8b0f671a434b5df740c873bb6532e78d24f23d9b166a9d750c9ff2ad99.png",
    "865f088d055464fa051e24feee47c077c09f1d30075555a5344828062b2c3540.png",
    "f219310062b9cf5de027d401cda8a7251d4463eb77def828743ffdf0e14da361.png",
    "fc899dbcf637f8cf960ee62b30e96a8cade32234226da9346c1b3715abcd085a.png",
    "ff1855e8a29e59504992ed3847f30c11c1af036ab94ec470b82c9c00ef9055ca.png",
    "f359c97b078a84dd340b209a56962db59c0f7d9a08ec7e0e9dbf0742cd3f8656.png",
    "ea2b85a9a07e797379d82ba9c95e57f972ee4fab7335cfe1db5dfec8742d3175.png",
    "e0f833dc199ebda9ad34b8a951376785b1fa710881d95a9581f246aa3b0005d3.png",
    "410524b80467ceca37a5e0e315302dcf5c5c7ac0f58d1a2896d0e9e88e3af281.png",
]

ASSETS_BASE = "assets/250_SAMPLED_FROM_HSSD"

ALPHAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BASE_OUT = "SUBMISSION_OUTPUTS"

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read {video_path}")
    return frame

def build_grid(folder):
    try:
        gs_files = sorted(
            [(float(f.replace("_gs.mp4", "")), os.path.join(folder, f))
             for f in os.listdir(folder) if f.endswith("_gs.mp4")],
            key=lambda x: x[0]
        )
        if not gs_files:
            return
        frames = []
        for _, p in gs_files:
            try:
                frames.append(get_first_frame(p))
            except Exception as e:
                print(f"    [WARN] grid: skipping frame {p}: {e}")
                continue
        if not frames:
            return
        h, w, _ = frames[0].shape
        grid = np.zeros((h, w * len(frames), 3), dtype=np.uint8)
        for i, frame in enumerate(frames):
            grid[:, i*w:(i+1)*w] = frame
        cv2.imwrite(os.path.join(folder, "grid.png"), grid)
    except Exception as e:
        print(f"    [WARN] build_grid failed for {folder}: {e}")

def run_direction(pipeline, img_a, img_b, actual_image, out_dir):
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        print(f"    [WARN] makedirs failed for {out_dir}: {e}")
        return

    for alpha in ALPHAS:
        try:
            alpha = float(alpha)
            outputs = pipeline.run_sparse_interp_with_direction(
                image1=img_a,
                image2=img_b,
                actual_image=actual_image,
                seed=1,
                alpha=alpha,
            )
            video = render_utils.render_video(outputs['gaussian'][0])['color']
            imageio.mimsave(os.path.join(out_dir, f"{alpha:.4f}_gs.mp4"), video, fps=30)
            print(f"    alpha={alpha:.4f} done")
        except Exception as e:
            print(f"    [WARN] alpha={alpha:.4f} failed: {e}")
            continue

    build_grid(out_dir)


try:
    pipeline = TrellisImageTo3DPipeline.from_pretrained(
        "TRELLIS_MODELS/TRELLIS-image-large"
    )
    pipeline.cuda()
except Exception as e:
    print(f"[ERROR] Pipeline load failed: {e}")
    raise

try:
    pair_name, img1_path, img2_path = IMAGE_PAIRS[PAIR_IDX]
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
except Exception as e:
    print(f"[ERROR] Failed to load image pair: {e}")
    raise

pair_root    = os.path.join(BASE_OUT, pair_name)
forward_root = os.path.join(pair_root, "forward")
reverse_root = os.path.join(pair_root, "reverse")

for asset_fname in ASSETS:

    asset_stem   = os.path.splitext(asset_fname)[0]
    actual_image = Image.open(os.path.join(ASSETS_BASE, asset_fname))

    run_direction(pipeline, img1, img2, actual_image,
                    os.path.join(forward_root, asset_stem))

    run_direction(pipeline, img2, img1, actual_image,
                      os.path.join(reverse_root, asset_stem))


print("\nAll done.")