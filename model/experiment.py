from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
    TexturesVertex
)
from pytorch3d.structures import Meshes
import torch

from model.bprimitive_subdivision import BPrimitiveSubdivision


def map_3d_to_2d(cube_coords):
    x, y, z = cube_coords.unbind(-1)

    # 创建一个UV坐标张量
    uvs = torch.zeros_like(cube_coords[:, :2])

    # x-plane: 如果 x 是 -1 或 1，则 z 和 y 决定 u 和 v
    mask_x = torch.abs(x) == 1
    uvs[mask_x, 0] = (z[mask_x] + 1) / 2  # u 由 z 决定
    uvs[mask_x, 1] = (y[mask_x] + 1) / 2  # v 由 y 决定

    # y-plane: 如果 y 是 -1 或 1，则 x 和 z 决定 u 和 v
    mask_y = torch.abs(y) == 1
    uvs[mask_y, 0] = (x[mask_y] + 1) / 2  # u 由 x 决定
    uvs[mask_y, 1] = (z[mask_y] + 1) / 2  # v 由 z 决定

    # z-plane: 如果 z 是 -1 或 1，则 x 和 y 决定 u 和 v
    mask_z = torch.abs(z) == 1
    uvs[mask_z, 0] = (x[mask_z] + 1) / 2  # u 由 x 决定
    uvs[mask_z, 1] = (y[mask_z] + 1) / 2  # v 由 y 决定

    return uvs

def experimental_cube_get_bprimitive_gaussian(
    num_samples_per_cube_edge: int,
    order: int
):
    """
    Attach BP on a surface of cube.
    Attach GS on the BP.

    Args:
        num_samples_per_cube_edge (int): Number of samples per cube edge.
        order (int): Order of the B-primitive.

    Returns:
        bp_gs (GaussianModel): Gaussian model (6 * num_samples_per_cube_edge ** 2 * (num_samples + 1) * (num_samples + 2) // 2).
    """

    transforms = torch.Tensor([
        [[2, 0, 0, -1], [0, 0, 2, -1], [0, -2, 0, 1]],
        [[2, 0, 0, -1], [0, 0, 2, 1], [0, -2, 0, 1]],
        [[0, 0, -2, -1], [2, 0, 0, -1], [0, -2, 0, 1]],
        [[0, 0, -2, 1], [2, 0, 0, -1], [0, -2, 0, 1]],
        [[2, 0, 0, -1], [0, -2, 0, 1], [0, 0, -2, 1]],
        [[2, 0, 0, -1], [0, -2, 0, 1], [0, 0, -2, -1]]
    ])

    # Triangles on one face of the cube

    yv, xv = torch.meshgrid(
        torch.linspace(0, 1, steps=num_samples_per_cube_edge + 1),
        torch.linspace(0, 1, steps=num_samples_per_cube_edge + 1),
        indexing='ij'
    )
    points = torch.stack([xv.flatten(), yv.flatten(), torch.zeros_like(xv.flatten()), torch.ones_like(xv.flatten())], dim=1)

    indices = torch.arange((num_samples_per_cube_edge + 1) ** 2).view(num_samples_per_cube_edge + 1, num_samples_per_cube_edge + 1)

    top_left = indices[:-1, :-1].flatten()
    top_right = indices[:-1, 1:].flatten()
    bottom_left = indices[1:, :-1].flatten()
    bottom_right = indices[1:, 1:].flatten()

    tri1 = torch.stack([top_left, top_right, bottom_left], dim=-1)
    tri2 = torch.stack([top_right, bottom_right, bottom_left], dim=-1)
    indices = torch.cat([tri1, tri2], dim=0) # [2 * num_samples_per_cube_edge ** 2, 3]

    # Triangles on six faces of the cube

    all_points = torch.einsum("sij, nj -> sni", transforms, points).reshape(-1, 3) # [6 * (num_samples_per_cube_edge + 1) ** 2, 3]
    all_indices = (indices + torch.arange(6).reshape(-1, 1, 1) * (num_samples_per_cube_edge + 1) ** 2).reshape(-1, 3) # [6 * 2 * num_samples_per_cube_edge ** 2, 3]

    uvw = BPrimitiveSubdivision.generate_uvw(order)
    control_points = torch.einsum("sxd, nx-> snd", all_points[all_indices], uvw) # [s, (n + 2) * (n + 1) // 2, 3]

    return control_points

def experimental_tetrahedron_get_bprimitive_gaussian(
    num_samples_per_cube_edge: int,
    order: int
):
    """
    Attach BP on a surface of tetrahedron.
    Attach GS on the BP.

    Args:
        num_samples_per_cube_edge (int): Number of samples per cube edge.
        order (int): Order of the B-primitive.

    Returns:
        bp_gs (GaussianModel): Gaussian model (6 * num_samples_per_cube_edge ** 2 * (num_samples + 1) * (num_samples + 2) // 2).
    """

    tetrahedron_bprimitive = BPrimitiveSubdivision(4, sh_degree=0)
    base_control_points = torch.tensor([
        [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, -1.0]],
        [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [-1.0, -1.0, -1.0]],
        [[1.0, 1.0, 1.0], [0.0, 1.0, -1.0], [-1.0, -1.0, -1.0]],
        [[0.0, 1.0, 1.0], [0.0, 1.0, -1.0], [-1.0, -1.0, -1.0]],
    ], dtype=torch.float32)
    tetrahedron_bprimitive.control_point = base_control_points
    tetrahedron_bprimitive.control_points = base_control_points
    tetrahedron_bprimitive.num_primitives = base_control_points.size(0)
    all_points, all_indices = tetrahedron_bprimitive.generate_regular_mesh(num_samples_per_cube_edge)
    all_points = all_points.reshape(-1, 3)
    all_indices = all_indices.reshape(-1, 3)

    uvw = BPrimitiveSubdivision.generate_uvw(order)
    control_points = torch.einsum("sxd, nx-> snd", all_points[all_indices], uvw) # [s, (n + 2) * (n + 1) // 2, 3]

    return control_points

def experimental_triangle_get_bprimitive_gaussian(
    num_samples_per_cube_edge: int,
    order: int
):
    """
    Attach BP on a surface of triangle.
    Attach GS on the BP.

    Args:
        num_samples_per_cube_edge (int): Number of samples per cube edge.
        order (int): Order of the B-primitive.

    Returns:
        bp_gs (GaussianModel): Gaussian model (6 * num_samples_per_cube_edge ** 2 * (num_samples + 1) * (num_samples + 2) // 2).
    """

    # tetrahedron_bprimitive = BPrimitiveSubdivision(3, 1)
    # tetrahedron_bprimitive.control_points = torch.Tensor([
    #     [
    #         [1.0, 1.0, 0.0],
    #         [2/3, 4/3, 0.0], [1/3, 0.0, 0.0],
    #         [1/3, 4/3, 0.0], [0.0, 1/3, 0.0], [-1/3, -2/3, 0.0],
    #         [0.0, 1.0, 0.0], [-2/3, 1/3, 0.0], [-1, -1/3, 0.0], [-1.0, -1.0, 0.0]
    #     ],
    # ])
    tetrahedron_bprimitive = BPrimitiveSubdivision(1, sh_degree=0)
    base_control_points = torch.tensor([
        [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, -1.0, 0.0]],
    ], dtype=torch.float32)
    tetrahedron_bprimitive.control_point = base_control_points
    tetrahedron_bprimitive.control_points = base_control_points
    tetrahedron_bprimitive.num_primitives = base_control_points.size(0)
    all_points, all_indices = tetrahedron_bprimitive.generate_regular_mesh(num_samples_per_cube_edge)
    all_points = all_points.reshape(-1, 3)
    all_indices = all_indices.reshape(-1, 3)

    uvw = BPrimitiveSubdivision.generate_uvw(order)
    control_points = torch.einsum("sxd, nx-> snd", all_points[all_indices], uvw) # [s, (n + 2) * (n + 1) // 2, 3]

    return control_points

def experimental_triangle_get_bprimitive_random(
    num_samples_per_cube_edge: int,
    order: int,
):
    """
    Attach BP on a surface of triangle.
    Attach GS on the BP.

    Args:
        num_samples_per_cube_edge (int): Number of samples per cube edge.
        order (int): Order of the B-primitive.
        num_samples (int): Number of samples per B-primitive's edge.
        sh_degree (int): Degree of spherical harmonics.

    Returns:
        bp_gs (GaussianModel): Gaussian model (6 * num_samples_per_cube_edge ** 2 * (num_samples + 1) * (num_samples + 2) // 2).
    """

    

    tetrahedron_bprimitive = BPrimitiveSubdivision(1, sh_degree=0)
    base_control_points = torch.tensor([
        [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, -1.0, 0.0]],
    ], dtype=torch.float32)
    tetrahedron_bprimitive.control_point = base_control_points
    tetrahedron_bprimitive.control_points = base_control_points
    tetrahedron_bprimitive.num_primitives = base_control_points.size(0)
    all_points, all_indices = tetrahedron_bprimitive.generate_regular_mesh(num_samples_per_cube_edge)
    all_points = all_points.reshape(-1, 3)
    all_indices = all_indices.reshape(-1, 3)




    # B-primitive


    num_primitives = all_indices.shape[0]
    bprimitive = BPrimitiveSubdivision(order, sh_degree=0)
    bprimitive.control_points = torch.Tensor([
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0], [-1.0, 0, 0.0], [-1.0, -1.0, 0.0]
        ],
    ]).repeat(num_primitives,1,1) * 0.1 + torch.randn(num_primitives, 1, 3).repeat(1,bprimitive.num_control_points,1)/2

    return bprimitive

def experimental_triangle_gt_image(cameras: FoVPerspectiveCameras):
    vertex = torch.Tensor([
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
    ])
    faces = torch.LongTensor([
        [0, 1, 2],
    ])
    colors = torch.Tensor([
        [0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0],
    ])
    textures = TexturesVertex(verts_features=colors.unsqueeze(0))
    mesh = Meshes(verts=[vertex], faces=[faces], textures=textures)

    raster_settings = RasterizationSettings(
        image_size=[1000, 1000],
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(
            device=vertex.device,
            cameras=cameras
        )
    )

    images = renderer(mesh)
    rgba_image = images[0, ..., :4]
    rgb_image = rgba_image[..., :3]
    alpha_channel = rgba_image[..., 3]
    
    black_background_image = torch.where(
        alpha_channel[..., None] == 0, torch.zeros_like(rgb_image), rgb_image
    ).permute(2, 0, 1)
    return black_background_image
