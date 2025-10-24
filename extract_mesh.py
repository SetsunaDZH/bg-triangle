from argparse import ArgumentParser, Namespace
import datetime
import os
import random
import sys
import wandb
import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from tqdm import tqdm
import viser

from arguments import (
    GroupParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
import pdb

from model import BPrimitiveSubdivision, GaussianModel
from render import Renderer, network_gui, render_3dgs
from scene import Scene
from scene.cameras import Camera
from utils.general_utils import safe_state
from utils.image_utils import psnr, edge_detection_rgb, grayscale_dilation, calculate_normals
from utils.loss_utils import ssim, l1_loss
import trimesh



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])


    dataset_args = lp.extract(args)
    optimzer_args = op.extract(args)
    pipeline_args = pp.extract(args)


    bprimitive_object = BPrimitiveSubdivision(
                order=dataset_args.order,
                sh_degree=3,
            )
    
    (model_params, first_iter) = torch.load(args.checkpoint)
    bprimitive_object.restore(model_params, optimzer_args)
    bprimitive_object.to("cuda")

    vertices, faces = bprimitive_object.generate_regular_mesh(10)
    vertices = vertices.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    vertices = vertices.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    with open("out.obj", 'w') as f:
        f.write(trimesh.exchange.obj.export_obj(mesh))


