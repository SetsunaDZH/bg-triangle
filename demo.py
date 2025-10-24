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

from model import BPrimitiveSubdivision, GaussianModel
from render import Renderer, network_gui, render_3dgs
from scene import Scene
from scene.cameras import Camera
from utils.general_utils import safe_state
from utils.image_utils import psnr, edge_detection_rgb, grayscale_dilation, calculate_normals
from utils.loss_utils import ssim, l1_loss

@torch.no_grad()
def training_report(tb_writer, iteration, l2_loss, loss, testing_iterations, scene: Scene, renderer, background, renderArgs, logwandb):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l2_loss', l2_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()}, 
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )



        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l2_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = renderer(
                        scene.bprimitive_object,
                        viewpoint,
                        background,
                        server["server"],
                        server["num_segments_per_bprimitive_edge"].value,
                        server["slider_boundary_scale"].value,
                        server["gaussian_scale"].value,
                        "debug"
                    )[0]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0).flip([1, 2])
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l2_test += torch.nn.functional.mse_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l2_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L2 {} PSNR {}".format(iteration, config['name'], l2_test, psnr_test))
                if logwandb:
                    wandb.log({config['name']: psnr_test}, commit=False)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l2_loss', l2_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)



        if tb_writer:
            tb_writer.add_scalar('total_points', scene.bprimitive_object.control_point.size(0), iteration)
        torch.cuda.empty_cache()


def prepare_output_and_logger(args: GroupParams):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        args.model_path = os.path.join("./outputs/", unique_str)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


class Trainer(object):
    def __init__(self,
        dataset_args: GroupParams,
        optimzer_args: GroupParams,
        pipeline_args: GroupParams,
        checkpoint: str
    ) -> None:
        self.first_iter = 0
        self.tb_writer = prepare_output_and_logger(dataset_args)

        self.bprimitive_object = BPrimitiveSubdivision(
            order=dataset_args.order,
            sh_degree=dataset_args.sh_degree,
            optimizer_type=optimzer_args.optimizer_type
        )
        self.gaussian_ori = GaussianModel()
        self.scene = Scene(
            args=dataset_args,
            bprimitive_object=self.bprimitive_object, 
            gaussian_ori = self.gaussian_ori
        )
        # self.bprimitive_object.training_setup(optimzer_args)
        if checkpoint:
            (model_params, self.first_iter) = torch.load(checkpoint)
            self.bprimitive_object.restore(model_params, optimzer_args)
            if self.first_iter>15200:
                self.bprimitive_object.mode(2)


        bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.dataset_args = dataset_args
        self.optimzer_args = optimzer_args
        self.pipeline_args = pipeline_args
        self.ema_loss_for_log = 0.0

        self.renderer = Renderer()


    def train(self, server, testing_iterations, checkpoint_iterations, debug_from, disable_viewer) -> None:
        if not disable_viewer:
            network_gui.try_connect()
        else:
            server["server"] = None

        viewpoint_stack = None
        self.progress_bar = tqdm(range(self.first_iter, self.optimzer_args.iterations), desc="Training progress")
        self.first_iter += 1

        if self.bprimitive_object.boundary_mode == 2:
            server["mode"].value=2


        for iteration in range(self.first_iter, self.optimzer_args.iterations + 1):

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
            #viewpoint_cam = viewpoint_stack[int(iteration//20)]

            # Render
            if (iteration - 1) == debug_from:
                self.pipeline_args.debug = True

            bg = torch.rand((3), device="cuda") if self.optimzer_args.random_background else self.background


            if server["checkbox_pts_mode"].value:
                self.train_step_3dgs(server, iteration, viewpoint_cam, testing_iterations, checkpoint_iterations, bg)
            else:
                self.train_step(server, iteration, viewpoint_cam, testing_iterations, checkpoint_iterations, bg)

    def train_step(self, server, iteration, gt_camera, testing_iterations, checkpoint_iterations, bg) -> None:



        ## Mode switch -----------------------------------------------------------------
        if iteration == self.pipeline_args.mode2_iters:
            server["mode"].value=2
            server["slider_boundary_scale"].value = self.pipeline_args.log_blur_radius_tex
            self.bprimitive_object.create_feature_texture()


        if iteration > self.pipeline_args.fix_iters :
            server["checkbox_split"].value = False

        


       
        if server["mode"].value==0:
            server["slider_boundary_scale"].value = -9.0
            if self.bprimitive_object.mode(0):
                self.optimzer_args.position_lr_init *= 5
                self.bprimitive_object.training_setup(self.optimzer_args)

        if server["mode"].value==1:
            if self.bprimitive_object.mode(1):
                self.bprimitive_object.training_setup(self.optimzer_args)
                if server["slider_boundary_scale"].value<-7:
                    server["slider_boundary_scale"].value = -7.0

        if server["mode"].value==2:
            if self.bprimitive_object.mode(2):
                server["slider_boundary_scale"].value = self.pipeline_args.log_blur_radius_tex
                if self.pipeline_args.fix_iters < 30000:
                    server["checkbox_split"].value = False
                self.bprimitive_object.active_sh_degree = 0
                self.bprimitive_object.training_setup(self.optimzer_args)



        
        render_type = network_gui.on_gui_change()

        if server["mode"].value==1:
            lrs = self.bprimitive_object.update_learning_rate(iteration)
            #if server["wandb"]:
            #    wandb.log(lrs, commit=False)
        else:
            if iteration>15200:
                lrs = self.bprimitive_object.update_learning_rate(iteration-15000)
               

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.bprimitive_object.one_up_sh_degree()

        gt_image = gt_camera.original_image.cuda().flip([1, 2])



        if server["button_train_state"]:



            rendered_image, debug_info = self.renderer(
                self.bprimitive_object,
                gt_camera,
                bg,
                server["server"],
                server["num_segments_per_bprimitive_edge"].value,
                server["slider_boundary_scale"].value,
                server["gaussian_scale"].value,
                render_type
            )

            if server["mode"].value<2:
                photometric_loss = torch.nn.functional.mse_loss(rendered_image, gt_image)
            else:
                photometric_loss = l1_loss(rendered_image, gt_image)

            ssim_value = ssim(rendered_image, gt_image)

            loss = torch.lerp(photometric_loss, 1.0 - ssim_value, self.optimzer_args.lambda_dssim)


            loss.backward()



            with torch.no_grad():
                # Progress bar
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
                if iteration % 10 == 0:
                    self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                    self.progress_bar.update(10)
                if iteration == self.optimzer_args.iterations:
                    self.progress_bar.close()

                if server["wandb"]:
                    wandb.log({"loss": self.ema_loss_for_log, "#primitives": self.bprimitive_object.num_primitives})

                # Log and save
                training_report(self.tb_writer, iteration, photometric_loss, loss, testing_iterations, self.scene, self.renderer, self.background, self.pipeline_args, server["wandb"])
                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((self.bprimitive_object.capture(), iteration), self.scene.model_path + "/chkpnt" + str(iteration) + ".pth")


                # log  grad info
                self.bprimitive_object.add_densification_stats_v3()
                self.bprimitive_object.add_densification_stats_v4(debug_info["bprimitive_image"])
            
                if  self.bprimitive_object.boundary_mode>0:
                    edges = edge_detection_rgb(gt_image.unsqueeze(0))



                    boundary_image = grayscale_dilation(debug_info['colored_boundary'][...,0].float()).squeeze()
                    mask_edge = (boundary_image>250)
                    edges[:,mask_edge] = 0
                    
                    edges[edges<0.2] = 0
                
                    
                    mask = debug_info["valid_mask"]
                    edges[:,~mask] = 0

                    edges[edges<0] = 0

                    self.bprimitive_object.add_densification_stats_v2(
                        edges,
                        debug_info["bprimitive_image"],
                    )



                # Optimizer step
                if server["button_optimize_state"]:
                    if iteration < self.optimzer_args.iterations:
                        self.bprimitive_object.optimizer.step()
                        self.bprimitive_object.optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((self.bprimitive_object.capture(), iteration), self.scene.model_path + "/chkpnt" + str(iteration) + ".pth")


                # Densification # TODO
                if iteration < self.optimzer_args.densify_until_iter:
                    if iteration %self.optimzer_args.densification_interval==0 and server["checkbox_split"].value:
                        self.bprimitive_object.optimizer.zero_grad()
                        self.bprimitive_object.densify_and_prune(
                            edge_threshold = self.pipeline_args.edge_threshold,
                            grad_threshold = self.pipeline_args.grad_threshold, 
                            vis_threshold = self.pipeline_args.vis_threshold,
                            area_threshold = self.pipeline_args.area_threshold
                        )
                        self.bprimitive_object.training_setup(self.optimzer_args, server["learning_rate"].value)

        if server["server"] is None:
            return

        with torch.no_grad():
            client = server["client"]
            RT_w2v = viser.transforms.SE3(wxyz_xyz=np.concatenate([client.camera.wxyz, client.camera.position], axis=-1)).inverse()
            R = torch.tensor(RT_w2v.rotation().as_matrix().astype(np.float32))
            T = torch.tensor(RT_w2v.translation().astype(np.float32))

            #R = gt_camera.R
            #T = gt_camera.T
            R = R.numpy()
            T = T.numpy()
            FoVx = gt_camera.FoVx # TODO: client fov
            FoVy = gt_camera.FoVy

            camera = Camera(
                resolution=gt_image.shape[-2:],
                colmap_id=None,
                R=R,
                T=T,
                FoVx=FoVx,
                FoVy=FoVy,
                depth_params=None,
                image=gt_image,
                invdepthmap=None,
                image_name="",
                uid=None,
            )

            rendered_image, debug_info = self.renderer(
                self.bprimitive_object,
                camera,
                self.background,
                server["server"],
                server["num_segments_per_bprimitive_edge"].value,
                server["slider_boundary_scale"].value,
                server["gaussian_scale"].value,
                render_type
            )

            mask = debug_info["valid_mask"]
            if render_type == "Accumulated Gradient Image":
                accum_grad = self.bprimitive_object.gradient_accum / (self.bprimitive_object.denom + 1)
                accum_grad_map = torch.zeros_like(gt_image)
                accum_grad_map[:, mask] = accum_grad[debug_info["bprimitive_image"][mask]].squeeze()
                accum_grad_map = accum_grad_map / self.pipeline_args.grad_threshold

            if render_type == "Accum_Edge":
                accum_edge = self.bprimitive_object.edge_accum / (self.bprimitive_object.denom_edge + 1)
                accum_edge_map = torch.zeros_like(gt_image)
                accum_edge_map[:, mask] = accum_edge[debug_info["bprimitive_image"][mask]].squeeze()
                accum_edge_map = accum_edge_map / self.pipeline_args.edge_threshold

            if render_type == "Accum_Vis":
                accum_vis = self.bprimitive_object.vis_accum / (self.bprimitive_object.denom_vis + 1)
                accum_vis_map = torch.zeros_like(gt_image)
                accum_vis_map[:, mask] = accum_vis[debug_info["bprimitive_image"][mask]].squeeze()
                accum_vis_map = accum_vis_map / self.pipeline_args.vis_threshold



        output = None
        if render_type == "Depth Map":
            output = debug_info['depth']
        elif render_type == "Segmentation":
            output = debug_info['colored_seg']
        elif render_type == "Colored Boundary Points":
            output = debug_info['colored_boundary'].detach().cpu().numpy()
        elif render_type == "Colored UVW":
            output = debug_info['Colored UVW']
        elif render_type == "debug":
            rendered_image = rendered_image.detach().cpu().permute(1, 2, 0)
            rendered_image = rendered_image * 255
            rendered_image = rendered_image.byte().numpy()
            output = rendered_image
        elif render_type == "GT Image":
            output = gt_image.detach().cpu().permute(1, 2, 0)
            output = output * 255
            output = output.byte().numpy()
        elif render_type == "Surface Normal":
            output = debug_info['normal_image'].detach().cpu()
            output = output * 255
            output = output.byte().numpy()
        elif render_type == "Depth Normal":
            output = debug_info['depth_normal'].detach().cpu()
            output = output * 255
            output = output.byte().numpy()
        elif render_type == "Accumulated Gradient Image":
            accum_grad_map = accum_grad_map.clamp(0, 1)
            output = accum_grad_map.detach().cpu().permute(1, 2, 0)
            output = output * 255
            output = output.byte().numpy()
        elif render_type == "Accum_Edge":
            accum_edge_map = accum_edge_map.clamp(0, 1)
            output = accum_edge_map.detach().cpu().permute(1, 2, 0)
            output = output * 255
            output = output.byte().numpy()
        elif render_type == "Accum_Vis":
            accum_vis_map = accum_vis_map.clamp(0, 1)
            output = accum_vis_map.detach().cpu().permute(1, 2, 0)
            output = output * 255
            output = output.byte().numpy()
        elif render_type == "edges":
            output = edges.detach().cpu().permute(1, 2, 0)
            output = output * 255
            output = output.byte().numpy()
        else:
            print(f"Unsupported render type: {render_type}")

        client.scene.set_background_image(
            output,
            format="jpeg"
        )
        if not hasattr(server['server'], "num_primtives"):
            server['server'].num_primtives = server['server'].add_text("#Primitives", "0")
        server['server'].num_primtives.value = f"{self.bprimitive_object.control_point.size(0)}"
        torch.cuda.empty_cache()

    def train_step_3dgs(self, server, iteration, gt_camera, testing_iterations, checkpoint_iterations, bg) -> None:

        gt_image = gt_camera.original_image.cuda()

        if server["button_train_state"]:
            self.gaussian_ori.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                self.gaussian_ori.oneupSHdegree()

            render_pkg = render_3dgs(gt_camera, self.gaussian_ori, bg)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]



            Ll1 = l1_loss(image, gt_image)
            
            ssim_value = ssim(image, gt_image)

            loss = (1.0 - self.optimzer_args.lambda_dssim) * Ll1 + self.optimzer_args.lambda_dssim * (1.0 - ssim_value)


            # Optimizer step
            loss.backward()

            with torch.no_grad():
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
                if iteration % 10 == 0:
                    self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                    self.progress_bar.update(10)

                # Densification

                # Keep track of max radii in image-space for pruning
                self.gaussian_ori.max_radii2D[visibility_filter] = torch.max(self.gaussian_ori.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussian_ori.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > self.optimzer_args.densify_from_iter and iteration % 100 == 0:
                    size_threshold = 20 if iteration > self.optimzer_args.opacity_reset_interval else None
                    self.gaussian_ori.densify_and_prune(self.optimzer_args.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
            
                if iteration % self.optimzer_args.opacity_reset_interval == 0 :
                    self.gaussian_ori.reset_opacity()

                # Optimizer step
                if iteration < self.optimzer_args.iterations:
                    self.gaussian_ori.optimizer.step()
                    self.gaussian_ori.optimizer.zero_grad(set_to_none = True)

                if iteration == 7000:
                    self.gaussian_ori.save_pointcloud(self.dataset_args.colmap_scale)

        if server["server"] is None:
            return

        with torch.no_grad():
            client = server["client"]
            RT_w2v = viser.transforms.SE3(wxyz_xyz=np.concatenate([client.camera.wxyz, client.camera.position], axis=-1)).inverse()
            R = torch.tensor(RT_w2v.rotation().as_matrix().astype(np.float32))
            T = torch.tensor(RT_w2v.translation().astype(np.float32))

            #R = gt_camera.R
            #T = gt_camera.T
            R = R.numpy()
            T = T.numpy()
            FoVx = gt_camera.FoVx
            FoVy = gt_camera.FoVy

            camera = Camera(
                resolution=gt_image.shape[-2:],
                colmap_id=None,
                R=R,
                T=T,
                FoVx=FoVx,
                FoVy=FoVy,
                depth_params=None,
                image=gt_image,
                invdepthmap=None,
                image_name="",
                uid=None,
            )
            render_pkg = render_3dgs(camera, self.gaussian_ori, bg)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        
        if server["render_type"].value == "debug":
            rendered_image = image.flip([1, 2]).detach().cpu().permute(1, 2, 0)
            rendered_image = rendered_image * 255
            rendered_image = rendered_image.byte().numpy()
            output = rendered_image
        elif server["render_type"].value == "GT Image":
            output = gt_image.detach().cpu().permute(1, 2, 0)
            output = output * 255
            output = output.byte().numpy()

        client.scene.set_background_image(
            output,
            format="jpeg"
        )
        if not hasattr(server['server'], "num_primtives"):
            server['server'].num_primtives = server['server'].add_text("#Primitives", "0")
        server['server'].num_primtives.value = f"{self.gaussian_ori._xyz.size(0)}"




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,10000, 15000,20000, 25000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3000, 7_000, 15000,20000,30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.checkpoint_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

   

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    server = network_gui.init(args.ip, args.port, pp.extract(args))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)


    if args.wandb:
        wandb.init(
        # set the wandb project where this run will be logged
            project="vetorized_primitives",
        # track hyperparameters and run metadata
            config=vars(args),
            name = args.model_path.split('/')[-1]
        )
        server['wandb'] = True
    else:
        server['wandb'] = False
        

    trainer = Trainer(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint)
    trainer.train(server, args.test_iterations, args.checkpoint_iterations, args.debug_from, args.disable_viewer)

    # All done
    print("\nTraining complete.")
