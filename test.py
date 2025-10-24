from argparse import ArgumentParser
import os

import torch
import torchvision
import wandb
from PIL import Image as PILImage
from tqdm import tqdm
import numpy as np

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from model import BPrimitiveSubdivision
from render import Renderer
from scene import Scene
from utils.general_utils import safe_state

from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
import pdb

def render_set(model_path, name, iteration, views, bprimitives, pipeline, background, enable_wandb: bool):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    renderer = Renderer()


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = renderer(
            bprimitives,
            view,
            background,
            None,
            pipeline.num_segments_per_bprimitive_edge,
            pipeline.log_blur_radius,
            pipeline.scale_gaussian,
            "debug"
        )[0].flip([1, 2])
        gt = view.original_image[0:3, :, :]


        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
     
@torch.no_grad()
def render_sets(dataset : ModelParams, optimzer_args: OptimizationParams, checkpoint : str, pipeline : PipelineParams, skip_train : bool, skip_test : bool, mode:int, enable_wandb: bool):
    with torch.no_grad():
        bprimitives = BPrimitiveSubdivision(dataset.order, dataset.sh_degree)
        scene = Scene(dataset, bprimitives, shuffle=False)
        (model_params, _) = torch.load(checkpoint)
        bprimitives.restore(model_params, optimzer_args)

        bprimitives.mode(mode)


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), bprimitives, pipeline, background, enable_wandb)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), bprimitives, pipeline, background, enable_wandb)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--mode", required=True, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)





    render_sets(model.extract(args), optimization.extract(args), args.checkpoint, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, False)
