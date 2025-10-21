"""Subdivision-surface based BG-Triangle primitive implementation.

This module replaces the former Bézier triangle primitive with a subdivision
surface evaluator.  The original code evolved organically from the SIGGRAPH
2025 paper implementation and the control flow scattered the important ideas
across several helper utilities.  The goal of this rewrite is two-fold:

1. Provide a drop-in replacement that keeps the public interface intact so the
   training, evaluation, and viewer scripts continue to operate without any
   further changes.  We keep the legacy ``BPrimitiveBezier`` alias exposed
   through ``model.__init__`` for backwards compatibility.
2. Document the mathematical steps of the new primitive so future contributors
   can understand *why* the evaluator is structured in this fashion.  Most of
   the computation happens in barycentric space and relies on cached
   subdivision connectivity, hence the extensive commentary below.

The class itself is still heavily optimized for vectorized execution on the
GPU.  The surrounding documentation is deliberately verbose because the user
requested a detailed explanation of the change set.  Each helper method now has
docstrings that describe its role in the evaluation pipeline and how the caches
interact with one another.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from plyfile import PlyData

from .bprimitive_base import BPrimitiveBase
from utils.graphics_utils import BasicPointCloud
from utils.point_utils import generate_random_unit_triangles


def fetchPly(path):
    """Load a PLY file into :class:`BasicPointCloud`.

    The legacy Bézier implementation exposed this helper and several parts of
    the training scripts still import it.  Although unrelated to subdivision
    surfaces directly, keeping the function here allows existing experiment
    setups to keep running.  The function simply parses the vertex positions and
    returns a :class:`BasicPointCloud` with empty color/normal fields.
    """

    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return BasicPointCloud(points=positions, colors=None, normals=None)


class BPrimitiveSubdivision(BPrimitiveBase):
    """Triangle subdivision surface primitive.

    The primitive stores its learnable control points directly in barycentric
    coordinates.  Rather than evaluating a closed-form Bézier basis (which is
    numerically sensitive for high orders), we subdivide the reference triangle
    into a regular lattice and perform piecewise-linear interpolation inside the
    local triangle that contains the query.  This turns the geometry evaluation
    into a fast gather + dot-product routine that is easy to differentiate.

    A quick summary of the internal data:

    * ``control_point_uvw`` – canonical barycentric coordinates of the control
      lattice that match the learnable positions stored in ``self.control_point``.
    * ``subdivision_faces`` – integer triplets describing how the lattice is
      tessellated into micro-triangles.  These indices never change and only
      depend on the chosen order.
    * ``face_lookup_*`` – cached matrices used to locate which micro-triangle a
      query falls into and to compute barycentric derivatives.  They are shared
      across all primitives in the batch for maximum reuse.
    * ``face_cache``/``uvw_cache`` – reusable caches for mesh extraction where a
      different tessellation density may be requested.
    """

    def __init__(self, order: int, sh_degree: int, optimizer_type: str = "default") -> None:
        super().__init__(order, sh_degree, optimizer_type)

        # Control lattice in barycentric space for the optimized parameters.
        self.control_point_uvw = self.generate_uvw(order)  # [num_control_points, 3]
        self.control_point_uv = self.control_point_uvw[:, 1:]

        # Pre-compute subdivision connectivity and evaluation helpers.
        self.subdivision_faces = self.generate_regular_face(order)
        self._prepare_face_lookup()

        # Regular mesh caches for different evaluation resolutions.
        num_caches = 100
        self.face_cache = {cache: self.generate_regular_face(cache) for cache in range(1, num_caches + 1)}
        self.uvw_cache = {cache: self.generate_uvw(cache) for cache in range(1, num_caches + 1)}

    def _prepare_face_lookup(self) -> None:
        """Pre-compute lookup tables for barycentric queries.

        The lookup resembles the structure used in rasterization: we express any
        point in the reference triangle as ``p = p0 + a * e0 + b * e1``.  By
        storing the inverse edge matrix we can recover ``(a, b)`` for any query
        and thus obtain the barycentric coordinates.  The derivative helpers are
        derived analytically from the transformation so that gradient-based
        optimizers can access first-order information without finite differences.
        """

        face_vertices_uv = self.control_point_uv[self.subdivision_faces]
        p0 = face_vertices_uv[:, 0]
        edges = torch.stack([
            face_vertices_uv[:, 1] - p0,
            face_vertices_uv[:, 2] - p0,
        ], dim=-1)  # [num_faces, 2, 2]

        inv_edges = torch.inverse(edges)

        # Derivative helpers for barycentric coordinates with respect to (u, v).
        dtype = face_vertices_uv.dtype
        device = face_vertices_uv.device
        du_vec = torch.tensor([0.0, -1.0], dtype=dtype, device=device)
        dv_vec = torch.tensor([1.0, -1.0], dtype=dtype, device=device)

        du_local = torch.einsum("fij,j->fi", inv_edges, du_vec)
        dv_local = torch.einsum("fij,j->fi", inv_edges, dv_vec)

        self.face_lookup_p0 = p0
        self.face_lookup_inv_edges = inv_edges
        self.face_dbary_du = torch.stack([
            -(du_local[:, 0] + du_local[:, 1]),
            du_local[:, 0],
            du_local[:, 1],
        ], dim=-1)
        self.face_dbary_dv = torch.stack([
            -(dv_local[:, 0] + dv_local[:, 1]),
            dv_local[:, 0],
            dv_local[:, 1],
        ], dim=-1)

    def to(self, device: torch.device) -> "BPrimitiveSubdivision":
        """Move all cached tensors to ``device``.

        ``BPrimitiveBase.to`` only transfers the learnable parameters.  The new
        caches introduced by the subdivision evaluator also need to migrate, so
        we mirror the API and return ``self`` for fluent chaining (matching
        PyTorch modules).
        """
        self.control_point_uvw = self.control_point_uvw.to(device)
        self.control_point_uv = self.control_point_uv.to(device)
        self.subdivision_faces = self.subdivision_faces.to(device)
        self.face_lookup_p0 = self.face_lookup_p0.to(device)
        self.face_lookup_inv_edges = self.face_lookup_inv_edges.to(device)
        self.face_dbary_du = self.face_dbary_du.to(device)
        self.face_dbary_dv = self.face_dbary_dv.to(device)

        self.face_cache = {k: v.to(device) for k, v in self.face_cache.items()}
        self.uvw_cache = {k: v.to(device) for k, v in self.uvw_cache.items()}
        return self

    @classmethod
    def generate_uvw(cls, n: int) -> torch.Tensor:
        """Generate regularly spaced barycentric coordinates.

        The points are ordered from the top vertex downwards which matches the
        convention used in the historical Bézier implementation.  Maintaining the
        order avoids touching any of the learnable parameter initializers that
        expect the same layout.
        """
        return torch.tensor([[i / n, j / n, (n - i - j) / n] for i in range(n, -1, -1) for j in range(n - i, -1, -1)])

    def _locate_faces(self, uvw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identify subdivision triangle indices and barycentric weights.

        Args:
            uvw: ``(N, 3)`` tensor containing barycentric coordinates relative to
                the *coarse* control lattice.  The values are expected to sum to
                one, but we do not assume they are normalized because the caller
                might provide the ``(v, w)`` projection.

        Returns:
            A tuple ``(face_ids, barycentric)`` where ``face_ids`` indexes into
            ``self.subdivision_faces`` and ``barycentric`` gives the local
            barycentric coordinates inside that micro-triangle.
        """

        device = uvw.device
        face_p0 = self.face_lookup_p0.to(device)
        inv_edges = self.face_lookup_inv_edges.to(device)

        uv = uvw[:, 1:]  # convert to 2D coordinates (v, w)
        diff = uv.unsqueeze(1) - face_p0.unsqueeze(0)
        local = torch.einsum("nmd,mdk->nmk", diff, inv_edges)
        barycentric = torch.cat([
            1.0 - local.sum(dim=-1, keepdim=True),
            local,
        ], dim=-1)

        mask = (barycentric >= -1e-6).all(dim=-1)
        valid = mask.any(dim=1)
        if not valid.all():
            raise RuntimeError("Subdivision barycentric lookup failed for some samples")

        face_ids = torch.argmax(mask.float(), dim=1)
        barycentric = barycentric[torch.arange(uvw.size(0), device=device), face_ids]
        return face_ids, barycentric

    def _gather_vertices(self, control_points: torch.Tensor, face_ids: torch.Tensor) -> torch.Tensor:
        """Collect per-face control vertices for ``face_ids``.

        The helper encapsulates the slightly verbose ``torch.gather`` call and
        keeps shape handling centralized in one place.  It works for both the
        learnable control point tensor as well as any derived attributes.
        """
        faces = self.subdivision_faces
        if faces.device != control_points.device:
            faces = faces.to(control_points.device)
        indices = faces[face_ids]
        return torch.gather(
            control_points,
            1,
            indices.unsqueeze(-1).expand(-1, -1, control_points.size(-1)),
        )

    def generate_regular_mesh(self, num_segments_per_edge: int, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate uniformly subdivided triangle meshes.

        When exporting meshes the caller can specify how many segments to place
        on each edge of the canonical triangle.  Because the evaluation is fully
        vectorized we can compute the vertices for an entire batch of primitives
        in a single call.  The method returns both the evaluated vertex positions
        and the triangle indices so that downstream utilities (PLY export,
        marching cubes, etc.) can stay unchanged.
        """

        uvw = self.uvw_cache[num_segments_per_edge].to(self.control_point.device)

        if mask is None:
            primitive_indices = torch.arange(self.num_primitives, device=self.control_point.device)
        else:
            primitive_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)

        num_primitives = primitive_indices.numel()
        if num_primitives == 0:
            face_template = self.face_cache[num_segments_per_edge]
            return (
                torch.empty(0, uvw.size(0), 3, dtype=self.control_point.dtype, device=self.control_point.device),
                torch.empty(0, face_template.size(0), 3, dtype=torch.long, device=self.control_point.device),
            )

        repeated_uvw = uvw.unsqueeze(0).expand(num_primitives, -1, -1).reshape(-1, 3)
        repeated_ids = primitive_indices.repeat_interleave(uvw.size(0))
        evaluated = self.evaluate(repeated_ids, repeated_uvw).view(num_primitives, uvw.size(0), 3)

        faces = self.face_cache[num_segments_per_edge].to(self.control_point.device)
        faces = faces.unsqueeze(0).expand(num_primitives, -1, -1)
        offset = torch.arange(num_primitives, device=self.control_point.device).view(-1, 1, 1) * uvw.size(0)
        faces = faces + offset
        return evaluated, faces

    def generate_regular_face(self, m: int) -> torch.Tensor:
        """Enumerate the micro-triangle connectivity for a tessellation level.

        ``m`` corresponds to the number of subdivisions along each edge.  The
        faces are generated in the same order as the original implementation so
        the renderer and differentiable rasterizer receive identical topology.
        """
        face = []
        for i in range(m):
            l, r = i * (i + 1) // 2, (i + 1) * (i + 2) // 2
            for j in range(l, r - 1):
                face.append([j, j + i + 1, j + i + 2])
                face.append([j, j + i + 2, j + 1])
            face.append([r - 1, r + i, r + i + 1])
        return torch.tensor(face) # [m * m, 3]

    def generate_from_triangles(self, trangles: torch.Tensor) -> torch.Tensor:
        """
        Generate control points from trangles.

        Args:
            trangles (torch.Tensor): Trangles vertices, which can be regarded as order 1 B-primitive, shape [N, 3, 3].

        Returns:
            control_points (torch.Tensor): Shape [N, num_control_points, 3].
        """
        # Here, ijk = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # [3, 3]
        uvw = self.generate_uvw(self.order).to(trangles.device) # [num_control_points, 3]
        control_points = torch.einsum("ncd, mc -> nmd", trangles, uvw) # [N, num_control_points, 3]
        return control_points

    def evaluate(self, bprimitive_id: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bprimitive_id (torch.Tensor): Boundary Primitive ID [N].
            uvw (torch.Tensor): Boundary UVW coordinates [N, 3].

        Returns:
            vertices (torch.Tensor): Vertices in UVW space [N, 3].
        """
        face_ids, barycentric = self._locate_faces(uvw)
        control_points = self.control_point[bprimitive_id]
        gathered = self._gather_vertices(control_points, face_ids)
        vertices = torch.sum(gathered * barycentric.unsqueeze(-1), dim=1)
        return vertices

    def evaluate_dc(self, bprimitive_id: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bprimitive_id (torch.Tensor): Boundary Primitive ID [N].
            uvw (torch.Tensor): Boundary UVW coordinates [N, 3].

        Returns:
            vertices (torch.Tensor): Vertices in UVW space [N, 3].
        """
        face_ids, barycentric = self._locate_faces(uvw)
        control_points = self.control_point_dc[bprimitive_id]
        gathered = self._gather_vertices(control_points, face_ids)
        vertices = torch.sum(gathered * barycentric.unsqueeze(-1), dim=1)
        return vertices

    def evaluate_rest(self, bprimitive_id: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bprimitive_id (torch.Tensor): Boundary Primitive ID [N].
            uvw (torch.Tensor): Boundary UVW coordinates [N, 3].

        Returns:
            vertices (torch.Tensor): Vertices in UVW space [N, 3].
        """
        face_ids, barycentric = self._locate_faces(uvw)
        control_points = self.control_point_rest[bprimitive_id]
        gathered = self._gather_vertices(control_points, face_ids)
        vertices = torch.sum(gathered * barycentric.unsqueeze(-1), dim=1)
        return vertices

    def evaluate_u_derivative(self, bprimitive_id: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bprimitive_id (torch.Tensor): Boundary B-primitive ID [N].
            uvw (torch.Tensor): Boundary UVW coordinates [N, 3].

        Returns:
            d_vertices_d_u (torch.Tensor): Derivative of the vertices w.r.t. U [N, 3].
        """
        face_ids, _ = self._locate_faces(uvw)
        control_points = self.control_point[bprimitive_id]
        gathered = self._gather_vertices(control_points, face_ids)
        dbary_du = self.face_dbary_du[face_ids].to(control_points.device)
        d_vertices_d_u = torch.sum(gathered * dbary_du.unsqueeze(-1), dim=1)
        return d_vertices_d_u

    def evaluate_v_derivative(self, bprimitive_id: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bprimitive_id (torch.Tensor): Boundary B-primitive ID [N].
            uvw (torch.Tensor): Boundary UVW coordinates [N, 3].

        Returns:
            d_vertices_d_v (torch.Tensor): Derivative of the vertices w.r.t. V [N, 3].
        """
        face_ids, _ = self._locate_faces(uvw)
        control_points = self.control_point[bprimitive_id]
        gathered = self._gather_vertices(control_points, face_ids)
        dbary_dv = self.face_dbary_dv[face_ids].to(control_points.device)
        d_vertices_d_v = torch.sum(gathered * dbary_dv.unsqueeze(-1), dim=1)
        return d_vertices_d_v

    def evaluate_normal(self, bprimitive_id: torch.Tensor, uvw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bprimitive_id (torch.Tensor): Boundary B-primitive ID [N].
            uvw (torch.Tensor): Boundary UVW coordinates [N, 3].

        Returns:
            normals (torch.Tensor): Normals [N, 3].
        """
        d_vertices_d_u = self.evaluate_u_derivative(bprimitive_id, uvw)
        d_vertices_d_v = self.evaluate_v_derivative(bprimitive_id, uvw)
        normals = torch.cross(d_vertices_d_u, d_vertices_d_v, dim=-1)
        return normals

    def clone(self) -> "BPrimitiveSubdivision":
        bprimitive = BPrimitiveSubdivision(self.order, self.max_sh_degree)

        bprimitive.control_point.data = self.control_point.data.clone()
        bprimitive.features_dc.data = self.features_dc.data.clone()
        bprimitive.features_rest.data = self.features_rest.data.clone()
        bprimitive.features_mlp.data = self.features_mlp.data.clone()
        bprimitive.scaling.data = self.scaling.data.clone()
        bprimitive.opacity.data = self.opacity.data.clone()
        bprimitive.gradient_accum.data = self.gradient_accum.data.clone()
        bprimitive.denom.data = self.denom.data.clone()
        return bprimitive

    def mode(self, id = 0):
        '''
        mode:
        0 -> non_boundary_mode
        1 -> boundary_mode
        '''

        if  self.boundary_mode==id:
            return False
        self.boundary_mode =  id


        return True

    def create_from_cube(self, spatial_lr_scale: float) -> None:
        self.spatial_lr_scale = spatial_lr_scale

        from model import experimental_cube_get_bprimitive_gaussian
        control_points = experimental_cube_get_bprimitive_gaussian(
            num_samples_per_cube_edge=2,
            order=self.order
        ).cuda()
        print("Shape of control points points at initialization: ", control_points.shape)
        self.num_primitives = control_points.shape[0]

        resolution = 2

        self.control_point = nn.Parameter(control_points, requires_grad=True)

        print("Shape of control points points at CUBE initialization: ", control_points.shape)
        self.num_primitives = control_points.shape[0]
        # self.features_dc = nn.Parameter(torch.ones(self.num_primitives, 1, 3, device="cuda"), requires_grad=True)
        # self.features_rest = nn.Parameter(torch.zeros(self.num_primitives, (self.max_sh_degree + 1) ** 2 - 1, 3, device="cuda"), requires_grad=True)

        self.control_point_dc = nn.Parameter(torch.ones(self.num_primitives, self.control_point.size(1),  3, device="cuda"), requires_grad=True)
        self.control_point_rest = nn.Parameter(torch.zeros(self.num_primitives, self.control_point.size(1),  45, device="cuda"), requires_grad=True)

        self.create_feature_texture(resolution)
        #self.features_dc = nn.Parameter(torch.ones(self.num_primitives, 1, 3, device="cuda"), requires_grad=True)
        self.single_features_rest = nn.Parameter(torch.zeros(self.num_primitives, (self.max_sh_degree + 1) ** 2 - 1, 3, device="cuda"), requires_grad=True)
        #self.features_dc = nn.Parameter(torch.ones(self.num_primitives, 3, resolution, resolution, device="cuda"), requires_grad=True)
        #self.features_rest = nn.Parameter(torch.zeros(self.num_primitives, ((self.max_sh_degree + 1) ** 2 - 1) * 3, resolution, resolution, device="cuda"), requires_grad=True)
        self.features_mlp = nn.Parameter(torch.randn(self.num_primitives, self.num_feat_channel, device="cuda"), requires_grad=True)
        self.scaling = nn.Parameter(torch.full((self.num_primitives, 3), -5.0, device="cuda"), requires_grad=True)
        self.opacity = nn.Parameter(torch.logit(torch.full((self.num_primitives, 1), 0.9, device="cuda")), requires_grad=True)
        self.to("cuda")


    def create_from_pc(self, points, init_triangle_size: float, spatial_lr_scale: float, resolution: int, resolution_res: int = 1) -> None:
        self.spatial_lr_scale = spatial_lr_scale

     


        points = points.unsqueeze(1).repeat(1,3,1)

        triangles  = generate_random_unit_triangles(points.size(0))*init_triangle_size

        triangles = triangles + points

        control_points = torch.from_numpy(triangles.detach().cpu().numpy()).float().cuda()
        control_points = self.generate_from_triangles(control_points)
        print("Shape of control points points at initialization: ", control_points.shape)
        self.num_primitives = control_points.shape[0]

        self.control_point = nn.Parameter(control_points, requires_grad=True)

        self.control_point_dc = nn.Parameter(torch.ones(self.num_primitives, self.control_point.size(1),  3, device="cuda"), requires_grad=True)
        self.control_point_rest = nn.Parameter(torch.zeros(self.num_primitives, self.control_point.size(1),  45, device="cuda"), requires_grad=True)

        self.create_feature_texture(resolution, resolution_res)

        self.single_features_rest = nn.Parameter(torch.zeros(self.num_primitives, (self.max_sh_degree + 1) ** 2 - 1, 3, device="cuda"), requires_grad=True)
       
        self.features_mlp = nn.Parameter(torch.randn(self.num_primitives, self.num_feat_channel, device="cuda"), requires_grad=True)
        self.scaling = nn.Parameter(torch.full((self.num_primitives, 3), -5.0, device="cuda"), requires_grad=True)
        self.opacity = nn.Parameter(torch.logit(torch.full((self.num_primitives, 1), 0.9, device="cuda")), requires_grad=True)
        self.to("cuda")





    def create_from_obj(self, init_path: str, spatial_lr_scale: float) -> None:
        self.spatial_lr_scale = spatial_lr_scale

        # Load obj file
        import trimesh
        mesh = trimesh.load_mesh(init_path, force='mesh')

        vertices = mesh.vertices
        faces = mesh.faces
        # 对于blender 导出的模型需要使用以下变换！！！
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
        vertices[:, 1] *= -1

        resolution = 4

        control_points = torch.from_numpy(vertices[faces]).float().cuda()
        control_points = self.generate_from_triangles(control_points)
        print("Shape of control points points at initialization: ", control_points.shape)
        self.num_primitives = control_points.shape[0]

        self.control_point = nn.Parameter(control_points, requires_grad=True)

        self.control_point_dc = nn.Parameter(torch.ones(self.num_primitives, self.control_point.size(1),  3, device="cuda"), requires_grad=True)
        self.control_point_rest = nn.Parameter(torch.zeros(self.num_primitives, self.control_point.size(1),  45, device="cuda"), requires_grad=True)

        self.create_feature_texture(resolution)
        self.features_mlp = nn.Parameter(torch.randn(self.num_primitives, self.num_feat_channel, device="cuda"), requires_grad=True)
        self.scaling = nn.Parameter(torch.full((self.num_primitives, 3), -5.0, device="cuda"), requires_grad=True)
        self.opacity = nn.Parameter(torch.logit(torch.full((self.num_primitives, 1), 0.9, device="cuda")), requires_grad=True)
        self.to("cuda")

    def create_from_random(self, spatial_lr_scale: float) -> None:
        self.spatial_lr_scale = spatial_lr_scale

        from model import experimental_triangle_get_bprimitive_random

        bprimitive_object = experimental_triangle_get_bprimitive_random(50,self.order)

        resolution = 3


        self.control_point = nn.Parameter(bprimitive_object.control_points.clone().cuda(), requires_grad=True)
        print("Shape of control points points at initialization: ", self.control_point.shape)
        self.num_primitives = self.control_point.shape[0]
        self.features_mlp = nn.Parameter(torch.randn(self.num_primitives, self.num_feat_channel, device="cuda"), requires_grad=True)
        self.features_dc = nn.Parameter(torch.ones(self.num_primitives, 3, resolution, resolution, device="cuda"), requires_grad=True)
        self.features_rest = nn.Parameter(torch.zeros(self.num_primitives, ((self.max_sh_degree + 1) ** 2 - 1) * 3, resolution, resolution, device="cuda"), requires_grad=True)
        self.scaling = nn.Parameter(torch.full((self.num_primitives, 3), 0.004, device="cuda"), requires_grad=True)
        self.opacity = nn.Parameter(torch.logit(torch.full((self.num_primitives, 1), 0.9, device="cuda")), requires_grad=True)
        self.to("cuda")


    #----------------- Version 1 ---------------------------

    @torch.no_grad()
    def add_densification_stats(self, viewspace_point_tensor, bprimitive_ids):
        unique_bprimitive_ids, counts = torch.unique(bprimitive_ids, return_counts=True)

        src = torch.norm(viewspace_point_tensor.grad[:, :2], dim=-1, keepdim=True)
        src[src>1e-6] = 1e-9

        tmp = torch.zeros_like(self.gradient_accum).scatter_add(
            dim=0,
            index=bprimitive_ids.unsqueeze(-1), 
            src=src
        )
        tmp[unique_bprimitive_ids, 0] /= counts
        self.gradient_accum += tmp
        self.denom[unique_bprimitive_ids] += 1

    def split_bprimitive(self, edge_threshold = 5, grad_threshold = 5e-4, area_threshold = 0.00003):
        metrics = self.gradient_accum / (self.denom + 1)
        mask = metrics >= grad_threshold


        metrics = self.edge_accum / (self.denom_edge + 1)
        mask = mask | (metrics >= edge_threshold)


        mask = mask.squeeze(1)

        vertices, faces = self.generate_regular_mesh(2)
        vertices= vertices.reshape(-1, 3)
        faces = faces.reshape(-1, 3)

        v0 = vertices[faces[:, 0]]  # A
        v1 = vertices[faces[:, 1]]  # B
        v2 = vertices[faces[:, 2]]  # C

        AB = v1 - v0
        AC = v2 - v0
        cross_product = torch.cross(AB, AC, dim=-1)
        areas = 0.5 * torch.norm(cross_product, dim=-1)
        areas = areas.reshape(-1,4).sum(dim=1)


        mask = mask & (areas > area_threshold*2)


        num_primitive_to_split = mask.sum()

        if num_primitive_to_split != 0:
            split_points = self.generate_uvw(self.order).repeat(num_primitive_to_split, 1).cuda()
            bids = torch.arange(0, mask.size(0), device="cuda")
            bids = torch.repeat_interleave(bids[mask], self.num_control_points)

            new_trangle_points = self.evaluate(bids, split_points)
            new_trangle_points = new_trangle_points.reshape(num_primitive_to_split,-1,3)


            vertices2, faces2 = self.generate_regular_mesh(2, mask)
            vertices2 = vertices2.reshape(-1, 3)
            faces2 = faces2.reshape(-1, 3)
            new_trangles = vertices2[faces2]





            new_control_points = self.generate_from_triangles(new_trangles)

            if new_control_points.size(0)+(~mask).sum() < 202000:  # maximum number of primitives
                center = new_control_points.mean(dim=1).unsqueeze(1)
                new_control_points = (new_control_points-center)*1.1+center
                self.update_primitives_add(new_control_points, mask)
                print(f'New primitives: {new_control_points.size(0)}')

    def update_primitives_add(self, new_control_points, primitives_to_remove):
        self.control_point = nn.Parameter(torch.cat([
            self.control_point.data[~primitives_to_remove], new_control_points
        ], dim=0))

        self.control_point_dc = nn.Parameter(torch.cat([
            self.control_point_dc.data[~primitives_to_remove], self.control_point_dc.data[primitives_to_remove].repeat_interleave(4, dim=0)
        ], dim=0))


        self.control_point_rest = nn.Parameter(torch.cat([
            self.control_point_rest.data[~primitives_to_remove], self.control_point_rest.data[primitives_to_remove].repeat_interleave(4, dim=0)
        ], dim=0))

        
        self.num_primitives = self.control_point.size(0)

        self.gradient_accum = torch.zeros((self.control_point.size(0), 1), device="cuda")
        self.denom = torch.zeros((self.control_point.size(0), 1), device="cuda")


        def resize_split(X):
            
            to_process = X  
            h = X.size(-1)

            processed = None

            
            if to_process.shape[0] == 0:
                pass
            else:
               
                h_half = h // 2
                top_left = to_process[:, :, :h_half, :h_half]  # (M, 3, h/2, h/2)
                top_right = to_process[:, :, :h_half, h_half:]  # (M, 3, h/2, h/2)
                bottom_left = to_process[:, :, h_half:, :h_half]  # (M, 3, h/2, h/2)
                bottom_right = to_process[:, :, h_half:, h_half:]  # (M, 3, h/2, h/2)

                
                top_left_upscaled = F.interpolate(top_left, size=(h, h), mode='bilinear', align_corners=True)  # (M, 3, h, h)
                top_right_upscaled = F.interpolate(top_right, size=(h, h), mode='bilinear', align_corners=True)  # (M, 3, h, h)
                bottom_left_upscaled = F.interpolate(bottom_left, size=(h, h), mode='bilinear', align_corners=True)  # (M, 3, h, h)
                bottom_right_upscaled = F.interpolate(bottom_right, size=(h, h), mode='bilinear', align_corners=True)  # (M, 3, h, h)

                processed = torch.stack((top_left_upscaled, bottom_left_upscaled, top_left_upscaled,top_left_upscaled ), dim=1)  # (M, 4, 3, h, h)

                processed = processed.view(-1, X.size(1), h, h)  # (4 * M, 3, h, h)
            return processed




        self.single_features_rest =  nn.Parameter(torch.cat([
            self.single_features_rest.data[~primitives_to_remove], self.single_features_rest.data[primitives_to_remove].repeat_interleave(4, dim=0)
        ], dim=0).contiguous())



        self.scaling = nn.Parameter(torch.cat([
            self.scaling.data[~primitives_to_remove], self.scaling.data[primitives_to_remove].repeat_interleave(4, dim=0)
        ], dim=0).contiguous())
        self.opacity = nn.Parameter(torch.cat([
            self.opacity.data[~primitives_to_remove], self.opacity.data[primitives_to_remove].repeat_interleave(4, dim=0)
        ], dim=0).contiguous())

        self.vis_accum = torch.cat([
                        self.vis_accum[~primitives_to_remove], self.vis_accum[primitives_to_remove].repeat_interleave(4, dim=0)
                         ], dim=0)
        self.denom_vis = torch.cat([
                        self.denom_vis[~primitives_to_remove], self.denom_vis[primitives_to_remove].repeat_interleave(4, dim=0)
                         ], dim=0)

        self.vis_map = torch.cat([
                        self.vis_map[~primitives_to_remove], self.vis_map[primitives_to_remove].repeat_interleave(4, dim=0)
                         ], dim=0)



        


    def update_primitives_del(self,  primitives_to_remove):
        self.control_point = nn.Parameter(self.control_point.data[~primitives_to_remove])

        self.control_point_dc = nn.Parameter(self.control_point_dc.data[~primitives_to_remove])
        self.control_point_rest = nn.Parameter(self.control_point_rest.data[~primitives_to_remove])
        
        self.num_primitives = self.control_point.size(0)

        self.gradient_accum = self.gradient_accum[~primitives_to_remove]
        self.denom = self.denom[~primitives_to_remove]

        self.single_features_rest = nn.Parameter(self.single_features_rest.data[~primitives_to_remove])


        self.scaling = nn.Parameter(self.scaling.data[~primitives_to_remove].contiguous())
        self.opacity = nn.Parameter(self.opacity.data[~primitives_to_remove].contiguous())

    def prune(self, min_opacity, vis_threshold = 0.02, area_threshold = 0.00003):
        # Prune primitives with small opacity
        mask = self.opacity.data < min_opacity

        metrics = self.vis_accum / (self.denom_vis + 1)
        mask = mask | ( metrics < vis_threshold)

        mask = mask.squeeze(1)

        # Prune primitives with small areas
        vertices, faces = self.generate_regular_mesh(2)
        vertices= vertices.reshape(-1, 3)
        faces = faces.reshape(-1, 3)

        v0 = vertices[faces[:, 0]]  # A
        v1 = vertices[faces[:, 1]]  # B
        v2 = vertices[faces[:, 2]]  # C

        AB = v1 - v0
        AC = v2 - v0
        cross_product = torch.cross(AB, AC, dim=-1)
        areas = 0.5 * torch.norm(cross_product, dim=-1)
        areas = areas.reshape(-1,4).sum(dim=1)

        # Prune primitives with large aspect ratios
        vertices, faces = self.generate_regular_mesh(2)
        bbox_max = vertices.max(dim=1)[0]
        bbox_min = vertices.min(dim=1)[0]

        lengths = bbox_max - bbox_min
        lengths = lengths.sort(dim=1)[0]

        ratio = lengths[:, 2] / lengths[:, 1]

        # Update mask
        area_mask = areas < area_threshold
        ratio_mask = (ratio > 20)
        seen_mask = (self.vis_map.view(-1,36)>0).sum(dim=1) <8

        masks = mask | area_mask | ratio_mask | seen_mask
        print(f'Prune {masks.sum()} primitives, {mask.sum().item()} visibility, {area_mask.sum().item()} areas, {ratio_mask.sum()} ratio, {seen_mask.sum()} unseen')

        self.update_primitives_del(masks)

    @torch.no_grad()
    def densify_and_prune(self, edge_threshold = 5, grad_threshold = 5e-4, vis_threshold = 0.02, area_threshold = 0.00003):
        self.split_bprimitive(edge_threshold = edge_threshold,  grad_threshold = grad_threshold, area_threshold = area_threshold)
        self.prune(0.3, vis_threshold =vis_threshold, area_threshold = area_threshold)

        self.num_primitives = self.control_point.shape[0]
        print(f'Total #Primitives: {self.num_primitives}')



    #----------------- Version 2 ---------------------------

    @torch.no_grad()
    def add_densification_stats_v2(self, edge_map, bprimitive_image):

        mask = bprimitive_image != -1
        bprimitive_ids = bprimitive_image[mask]
        unique_bprimitive_ids, counts = torch.unique(bprimitive_ids, return_counts=True)


        src = edge_map.mean(dim=0)
        src = src[mask]





        tmp = torch.zeros_like(self.gradient_accum).scatter_add(
            dim=0,
            index=bprimitive_ids.unsqueeze(-1).long(), 
            src=src.unsqueeze(-1)
        )

        #mp[unique_bprimitive_ids, 0] /= counts
        self.edge_accum += tmp
        self.denom_edge[unique_bprimitive_ids] += 1


    @torch.no_grad()
    def add_densification_stats_v3(self):
        grads = self.control_point.grad

        grads = grads.norm(dim=-1).sum(dim=-1).unsqueeze(-1)
        self.gradient_accum += grads
        self.denom =  self.denom+ 1


    @torch.no_grad()
    def add_densification_stats_v4(self, bprimitive_image):

        mask = bprimitive_image != -1
        bprimitive_ids = bprimitive_image[mask]
        unique_bprimitive_ids, counts = torch.unique(bprimitive_ids, return_counts=True)

        
        self.vis_accum[unique_bprimitive_ids] += 1
        self.denom_vis +=  1