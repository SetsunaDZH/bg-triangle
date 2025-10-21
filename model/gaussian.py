import math
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.bprimitive_subdivision import BPrimitiveSubdivision
from diff_brasterization import GaussianRasterizer, GaussianRasterizationSettings


def pixel_to_non_square_ndc(coords_2d, H, W):
    range = 2.0
    if H > W:
        range = (H * 2.0) / W
    offset = range / 2.0
    coords_2d[:, 1] =  (range * coords_2d[:, 1] + offset) / H - offset

    range = 2.0
    if W > H:
        range = (W * 2.0) / H
    offset = range / 2.0
    coords_2d[:, 0] = (range * coords_2d[:, 0] + offset) / W - offset

    return coords_2d


def non_square_ndc_to_pixel(ndc, H, W):
    range = 2.0
    if H > W:
        range = (H * 2.0) / W
    offset = range / 2.0
    ndc[:, 1] = ((ndc[:, 1] + offset) * H - offset) / range

    range = 2.0
    if W > H:
        range = (W * 2.0) / H
    offset = range / 2.0
    ndc[:, 0] = ((ndc[:, 0] + offset) * W - offset) / range

    return ndc


class GaussianObject(nn.Module):
    def __init__(self, xyz, scaling, rotation, opacity, features_dc, features_rest, sh_degree, normal=None):
        super().__init__()

        self.sh_degree = sh_degree

        self.xyz = xyz
        self.scaling = scaling
        self.rotation = rotation
        self.opacity =opacity
        self.features_dc = features_dc
        self.features_rest = features_rest
        self.normal = normal

    @classmethod
    def from_bprimitive_version_1(cls, valid_bids, valid_uvws, bprimitive: BPrimitiveSubdivision):
        """
        For each pixel in the bprimitive_image is a GS.
        """
        xyz = bprimitive.evaluate(valid_bids, valid_uvws)
        scaling = bprimitive.get_scaling()[valid_bids]
        rotation = torch.zeros(xyz.size(0), 4, dtype=xyz.dtype).to(xyz.device)
        rotation[:,0] = 1
        opacity = torch.full((xyz.size(0), 1), 0.92, dtype=xyz.dtype).to(xyz.device)

       
       

        xx =  (valid_uvws[:, 0]*5).int()
        yy = (valid_uvws[:, 1]*5).int()

        bprimitive.vis_map[valid_bids,xx,yy] += 1
     
        
        features_dc = bprimitive.evaluate_dc(valid_bids,valid_uvws).unsqueeze(1)



        features_rest = bprimitive.single_features_rest[valid_bids]
        
        

        return cls(xyz, scaling, rotation, opacity, features_dc, features_rest, bprimitive.active_sh_degree)

    @classmethod
    def from_bprimitive_version_2(cls,  bprimitive: BPrimitiveSubdivision):
        """
        Each  bprimitive has own Gaussians
        """
        order = 6

        raise Exception("Shouldn't be here !!!")


        split_points = bprimitive.generate_uvw(order).repeat(bprimitive.num_primitives, 1).cuda().contiguous()
        bids = torch.arange(0, bprimitive.num_primitives, device="cuda")
        bids = torch.repeat_interleave(bids, (order+1)*(order+2)//2)


        xyz = bprimitive.evaluate(bids, split_points)
        scaling = bprimitive.get_scaling()[bids]
        rotation = torch.zeros(xyz.size(0), 4, dtype=xyz.dtype).to(xyz.device)
        opacity =  bprimitive.get_opacity()[bids]

        '''
        features_dc = bprimitive.features_dc[bids]
        features_rest = bprimitive.features_rest[bids]


        '''

        '''
        features_dc = []
        features_rest = []
        size = 65536
        for i in range(0, len(bids), size):
            coords = split_points[i:i + size, :2].view(-1, 1, 1, 2) * 2 - 1
            features_dc.append(F.grid_sample(bprimitive.features_dc[bids[i:i + size]], coords, align_corners=True, mode='bilinear').reshape(-1, 1, 3))
            features_rest.append(F.grid_sample(bprimitive.features_rest[bids[i:i + size]], coords, align_corners=True, mode='bilinear').reshape(-1, bprimitive.features_rest.size(1) // 3, 3))
        features_dc = torch.cat(features_dc, dim=0)
        features_rest = torch.cat(features_rest, dim=0)
        '''


        features_dc = bprimitive.evaluate_dc(bids,split_points).unsqueeze(1)
        features_rest = bprimitive.evaluate_rest(bids,split_points).reshape(-1, bprimitive.features_rest.size(1) // 3, 3)


        #scaling =  F.grid_sample(bprimitive.scaling[bids], split_points[..., :2].view(-1, 1, 1, 2) * 2 - 1, align_corners=True).reshape(-1,  3)

        return cls(xyz, scaling, rotation, opacity, features_dc, features_rest, bprimitive.active_sh_degree), bids
    
    @classmethod
    def from_bprimitive_version_3(cls, valid_bids, valid_uvws, bprimitive: BPrimitiveSubdivision):
        """
        Each  bprimitive has own Gaussians
        """

        xyz = bprimitive.evaluate(valid_bids, valid_uvws)

        opacity = torch.full((xyz.size(0), 1), 0.92, dtype=xyz.dtype).to(xyz.device)


 
        features_dc = []
        features_rest = []
        features_scaling = []
        features_rotation = []
        size = 65536


        for i in range(0, len(valid_bids), size):
            coords = valid_uvws[i:i + size, :2].view(-1, 1, 1, 2) * 2 - 1
            features_dc.append(F.grid_sample(bprimitive.features_dc[valid_bids[i:i + size]], coords, align_corners=True, mode='bilinear').reshape(-1, 1, 3))
            if bprimitive.max_sh_degree >0:
                features_rest.append(F.grid_sample(bprimitive.features_rest[valid_bids[i:i + size]], coords, align_corners=True, mode='bilinear').reshape(-1, bprimitive.features_rest.size(1) // 3, 3))
            features_scaling.append(F.grid_sample(bprimitive.features_scaling[valid_bids[i:i + size]], coords, align_corners=True, mode='bilinear').reshape(-1,  3))
            features_rotation.append(F.grid_sample(bprimitive.features_rotation[valid_bids[i:i + size]], coords, align_corners=True, mode='bilinear').reshape(-1,  4))
        features_dc = torch.cat(features_dc, dim=0)

        if bprimitive.max_sh_degree ==0:
            features_rest = torch.zeros(features_dc.size(0), (bprimitive.max_sh_degree + 1) ** 2 - 1, 3, device="cuda")
        else:
            features_rest = torch.cat(features_rest, dim=0)



        features_scaling = torch.cat(features_scaling, dim=0)
        features_scaling[features_scaling>-4] = -6
        features_scaling = bprimitive.scaling_activation(features_scaling)
        features_rotation = torch.cat(features_rotation, dim=0)
        features_rotation = bprimitive.rotation_activation(features_rotation)


        xx =  (valid_uvws[:, 0]*5).int()
        yy = (valid_uvws[:, 1]*5).int()

        bprimitive.vis_map[valid_bids,xx,yy] += 1



        #features_dc = outs[:,:3].reshape(-1,1,3)
        #features_rest = outs[:,3:].reshape(features_dc.size(0),-1,3)

        return cls(xyz, features_scaling, features_rotation, opacity, features_dc, features_rest, bprimitive.active_sh_degree)

class GaussianRenderer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
        gaussian_object: GaussianObject,
        camera,
        bg,
        boundary_params: dict,
        **kwargs
    ):
        tanfovx = math.tan(camera.FoVx * 0.5)
        tanfovy = math.tan(camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=gaussian_object.sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gaussian_object.xyz
        means2D = torch.zeros_like(means3D, requires_grad=True, device="cuda")
        try:
            means2D.retain_grad()
        except:
            pass

        shs = torch.cat([gaussian_object.features_dc, gaussian_object.features_rest], dim=1)
        colors_precomp = None
        cov3D_precomp = None



        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = gaussian_object.opacity,
            scales = gaussian_object.scaling,
            rotations = gaussian_object.rotation,
            cov3D_precomp = cov3D_precomp,
            boundary_points_3d = boundary_params["boundary_points_3d"],
            boundary_params = boundary_params
        )

        return rendered_image, radii, means2D
