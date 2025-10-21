from pytorch3d.renderer.cameras import FoVPerspectiveCameras, try_get_projection_transform
import torch
import torch.nn as nn
import pdb
from model.bprimitive_subdivision import BPrimitiveSubdivision
from model.gaussian import pixel_to_non_square_ndc
from brasterizer import BMeshRasterizer, BMeshRasterizationSettings
from utils.image_utils import gaussian_blur, median_filter_2d
from utils.point_utils import depth_to_normal


class BMeshObject(object):
    """
    Data class for B-primitive generated mesh.
    """
    def __init__(self,
        face_vertices: torch.Tensor,
        face_vertices_uvw: torch.Tensor,
        face_bp_idx: torch.Tensor,
        bp_first_idx: torch.Tensor,
    ) -> None:
        self.face_vertices = face_vertices
        self.face_vertices_uvw = face_vertices_uvw
        self.face_bp_idx = face_bp_idx
        self.bp_first_idx = bp_first_idx


    def to(self, device: torch.device) -> "BMeshObject":
        self.face_vertices = self.face_vertices.to(device)
        self.face_vertices_uvw = self.face_vertices_uvw.to(device)
        self.face_bp_idx = self.face_bp_idx.to(device)
        self.bp_first_idx = self.bp_first_idx.to(device)
        return self


    @classmethod
    def from_bprimitive_version_1(cls,
        bprimitive: BPrimitiveSubdivision,
        num_segments_per_bprimitive_edge: int
    ) -> "BMeshObject":
        """
        Args:
            bprimitive (BPrimitiveSubdivision): B-primitive.
            num_segments_per_bprimitive_edge (int): Number of segments per edge.

        Returns:
            face_verts (torch.Tensor): 3D coordinates for each vertex (F, 3, 3).
            face_verts_uvw (torch.Tensor): UVW coordinates for each vertex (F, 3, 3).
            face_bp_idx (torch.Tensor): Primitive ID for each face (F).
            bp_first_idx (torch.Tensor): The index of the first face of each primitive (S).
        """
        vertices, faces = bprimitive.generate_regular_mesh(num_segments_per_bprimitive_edge)
        vertices = vertices.reshape(-1, 3)
        faces = faces.reshape(-1, 3)

        uvw = bprimitive.generate_uvw(num_segments_per_bprimitive_edge).to(vertices.device)

        face_vertices = vertices[faces]
        face_vertices_uvw = uvw.unsqueeze(0).expand(bprimitive.num_primitives, -1, -1).reshape(-1, 3)[faces]
        face_bp_idx = torch.arange(bprimitive.num_primitives, device=vertices.device).reshape(-1, 1, 1).expand(-1, num_segments_per_bprimitive_edge * num_segments_per_bprimitive_edge, -1).reshape(-1)
        bp_first_idx = torch.arange(bprimitive.num_primitives, device=vertices.device) * (num_segments_per_bprimitive_edge * num_segments_per_bprimitive_edge)

        return BMeshObject(
            face_vertices=face_vertices,
            face_vertices_uvw=face_vertices_uvw,
            face_bp_idx=face_bp_idx,
            bp_first_idx=bp_first_idx,
        )


def mesh_transform(bmesh_object: BMeshObject, camera: FoVPerspectiveCameras, **kwargs) -> BMeshObject:
    """
    Args:
        meshes_world: a Meshes object representing a batch of meshes with
            vertex coordinates in world space.

    Returns:
        bmesh_proj: a Meshes object with the vertex positions projected
        in NDC space

    NOTE: keeping this as a separate function for readability but it could
    be moved into forward.
    """
    verts_world = bmesh_object.face_vertices

    # NOTE: Retaining view space z coordinate for now.
    eps = kwargs.get("eps", None)
    verts_view = camera.get_world_to_view_transform(**kwargs).transform_points(
        verts_world, eps=eps
    )
    to_ndc_transform = camera.get_ndc_camera_transform(**kwargs)
    projection_transform = try_get_projection_transform(camera, kwargs)
    if projection_transform is not None:
        projection_transform = projection_transform.compose(to_ndc_transform)
        verts_ndc = projection_transform.transform_points(verts_view, eps=eps)
    else:
        # Call transform_points instead of explicitly composing transforms to handle
        # the case, where camera class does not have a projection matrix form.
        verts_proj = camera.transform_points(verts_world, eps=eps)
        verts_ndc = to_ndc_transform.transform_points(verts_proj, eps=eps)

    verts_ndc[..., 2] = verts_view[..., 2]
    bmesh_object.face_verts = verts_ndc

    return bmesh_object


class BMeshRenderer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rasterizer = BMeshRasterizer(backend="cuda")

    def forward(self,
        camera,
        bprimitive_object: BPrimitiveSubdivision,
        bmesh_object: BMeshObject,
        rasterizer_settings: BMeshRasterizationSettings,
        render_type: str
    ) -> torch.Tensor:
        
        camera_old = camera

        # Convert the GS Camera to a PyTorch3D Camera
        camera = FoVPerspectiveCameras(
            znear=camera.znear,
            zfar=camera.zfar,
            aspect_ratio= 1.0,
            fov=camera.FoVy,
            degrees=False,
            R=torch.from_numpy(camera.R).unsqueeze(0),
            T=torch.from_numpy(camera.T).unsqueeze(0),
            device=bmesh_object.face_vertices.device
        )


        bmesh_proj = mesh_transform(bmesh_object, camera)

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.
        clip_barycentric_coords = rasterizer_settings.clip_barycentric_coords
        if clip_barycentric_coords is None:
            clip_barycentric_coords = rasterizer_settings.blur_radius > 0.0
        
        # If not specified, infer perspective_correct and z_clip_value from the camera
        if rasterizer_settings.perspective_correct is not None:
            perspective_correct = rasterizer_settings.perspective_correct
        else:
            perspective_correct = camera.is_perspective()

        if rasterizer_settings.z_clip_value is not None:
            z_clip = rasterizer_settings.z_clip_value
        else:
            znear = camera.get_znear()
            if isinstance(znear, torch.Tensor):
                znear = znear.min().item()
            z_clip = None if not perspective_correct or znear is None else znear / 2

        pix_to_face, zbuf, bary_coords, dists, debug_info = self.rasterizer(
            bmesh_proj,
            image_size=rasterizer_settings.image_size,
            blur_radius=rasterizer_settings.blur_radius,
            faces_per_pixel=rasterizer_settings.faces_per_pixel,
            bin_size=rasterizer_settings.bin_size,
            max_faces_per_bin=rasterizer_settings.max_faces_per_bin,
            clip_barycentric_coords=clip_barycentric_coords,
            perspective_correct=perspective_correct,
            cull_backfaces=rasterizer_settings.cull_backfaces,
            z_clip_value=z_clip,
            cull_to_frustum=rasterizer_settings.cull_to_frustum,
        )


        face_verts_uvw = bmesh_proj.face_vertices_uvw
        bary_coords = bary_coords.squeeze()
        valid_mask = debug_info['valid_mask']

        bprimitive_image = debug_info['bprimitive_image']
        boundary_points_2d = debug_info['outline_coords']

        uvw_image = torch.zeros_like(bary_coords)
        uvw_image[valid_mask,:] = (bary_coords[valid_mask,:].unsqueeze(-1).repeat(1,1,3)*face_verts_uvw[pix_to_face[valid_mask],:,:]).sum(dim=1)
        debug_info['uvw_image'] = uvw_image

        debug_info['valid_bids'] = bprimitive_image[valid_mask]
        debug_info['valid_uvws'] = uvw_image[valid_mask]

        boundary_bp_id = bprimitive_image[boundary_points_2d[:, 1], boundary_points_2d[:, 0]]
        boundary_uvw = uvw_image[boundary_points_2d[:, 1], boundary_points_2d[:, 0]]
        boundary_points_3d = bprimitive_object.evaluate(boundary_bp_id, boundary_uvw)

        # Debuging *******************************************************

        if render_type == "Depth Map":
            debug_info['raw_depth'] = gaussian_blur(zbuf[0].float().squeeze())
            depth = (zbuf/zbuf.max())[0].float().repeat(1,1,3)
            depth[depth<0]=0
            depth = (depth*255).detach().cpu().byte().numpy()
            debug_info['depth'] = depth

        elif render_type == "Segmentation":
            color_lut = torch.tensor([
                [200, 100, 0],    # Red
                [50, 200, 50],    # Green
                [50, 50, 200],    # Blue
                [100, 150, 50],  # Yellow
                [120, 5, 200],  # Magenta
            ], dtype=torch.uint8, device=bprimitive_image.device)
            color_indices = bprimitive_image % 5

            colored_seg = torch.zeros((*bprimitive_image.shape, 3), dtype=torch.uint8, device=bprimitive_image.device)
            colored_seg[valid_mask] = color_lut[color_indices[valid_mask]]
            colored_seg = colored_seg.detach().cpu().numpy()
            debug_info['colored_seg'] = colored_seg

        elif render_type == "Colored UVW":
            color_lut = torch.tensor([
                [200, 100, 0],    # Red
                [50, 200, 50],    # Green
                [50, 50, 200],    # Blue
                [100, 150, 50],  # Yellow
                [120, 5, 200],  # Magenta
            ], dtype=torch.uint8, device=bprimitive_image.device)
            color_indices = bprimitive_image % 5

            colored_image = torch.zeros((*bprimitive_image.shape, 3), dtype=torch.uint8, device=bprimitive_image.device)
            bg = torch.ones((*bprimitive_image.shape, 3), dtype=torch.uint8, device=bprimitive_image.device)*255
            colored_image[valid_mask] = (color_lut[color_indices[valid_mask]].float() * uvw_image[valid_mask] + bg[valid_mask].float() * (1.0-uvw_image[valid_mask])).byte()
            colored_image = colored_image.detach().cpu().numpy()
            debug_info['Colored UVW'] = colored_image

        
       

        elif render_type == "Surface Normal":
            debug_info['normal_image'] = self.render_normal(bprimitive_object, debug_info['valid_bids'], debug_info['valid_uvws'], valid_mask)

        elif render_type == "Depth Normal":
            debug_info['raw_depth'] = gaussian_blur(zbuf[0].float().squeeze())
            debug_info['depth_normal'] = (depth_to_normal(camera_old, debug_info['raw_depth'].flip([0,1])).flip([0,1]) + 1)/2

        debug_info['colored_boundary'] = torch.zeros((*bprimitive_image.shape, 3), dtype=torch.uint8)
        debug_info['colored_boundary'][boundary_points_2d[:, 1], boundary_points_2d[:, 0],:] = 255

        return boundary_points_3d, boundary_bp_id, debug_info

    def render_normal(self, bprimitive_object, valid_bids, valid_uvws, valid_mask):
        normal = bprimitive_object.evaluate_normal(valid_bids, valid_uvws)
        normal = torch.nn.functional.normalize(normal, dim=-1)

        normal_image = torch.zeros((*valid_mask.shape, 3), dtype=torch.float32, device=valid_mask.device)
        normal_image[valid_mask] = (normal + 1) / 2
        return normal_image
