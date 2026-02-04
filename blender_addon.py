"""Blender add-on: Biharmonic deformation transfer using libigl.

Select three meshes: original (low-res), deformed (low-res), and transfer-to (high-res).
The add-on transfers the deformation from original → deformed onto transfer-to.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

import bpy
import bmesh
from bpy.types import Operator, Panel
from bpy.props import BoolProperty, EnumProperty
from mathutils import Vector

try:
	import igl  # type: ignore
except Exception:  # pragma: no cover - Blender environment dependency
	igl = None


bl_info = {
	"name": "Biharmonic Deformation Transfer",
	"blender": (4, 0, 0),
	"category": "Scene",
}


# -----------------------------
# Utility functions (IGL)
# -----------------------------

def closest_point_handles(
	source_vertices: np.ndarray,
	target_vertices: np.ndarray,
	target_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Compute closest points on a target mesh for each source vertex.

	Parameters
	----------
	source_vertices
		Source vertices (N, 3).
	target_vertices
		Target mesh vertices (V, 3).
	target_faces
		Target mesh faces (F, 3).

	Returns
	-------
	face_indices
		Face index in the target mesh for each closest point.
	points
		Closest points on the target mesh surface.
	barycentric
		Barycentric coordinates of the closest points within their faces.
	"""
	_, face_indices, points = igl.point_mesh_squared_distance(
		source_vertices, target_vertices, target_faces
	)
	tri = target_faces[face_indices]
	v0 = target_vertices[tri[:, 0]]
	v1 = target_vertices[tri[:, 1]]
	v2 = target_vertices[tri[:, 2]]
	barycentric = igl.barycentric_coordinates(points, v0, v1, v2)
	return face_indices, points, barycentric


def vertex_handle_indices(
	face_indices: np.ndarray,
	barycentric: np.ndarray,
	faces: np.ndarray,
) -> np.ndarray:
	"""Pick a single face vertex as the handle for each closest point.

	Selects the vertex with the largest barycentric weight.
	"""
	tri = faces[face_indices]
	max_idx = np.argmax(barycentric, axis=1)
	handle_vertices = tri[np.arange(tri.shape[0]), max_idx]
	return handle_vertices


def aggregate_vertex_constraints(
	n_vertices: int,
	handle_vertices: np.ndarray,
	handle_disp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Average handle displacements for possibly repeated vertices."""
	accum = np.zeros((n_vertices, 3), dtype=handle_disp.dtype)
	counts = np.zeros((n_vertices, 1), dtype=np.int32)
	accum[handle_vertices] += handle_disp
	counts[handle_vertices] += 1
	b = np.where(counts[:, 0] > 0)[0]
	bc = accum[b] / counts[b]
	return b, bc


def face_to_vertex_constraints(
	n_vertices: int,
	face_indices: np.ndarray,
	faces: np.ndarray,
	handle_disp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Convert face-based handles to vertex constraints by averaging.

	Each handle displacement is applied to the three face vertices and
	averaged per vertex. This is a simple approximation that preserves
	the handle point's displacement in expectation.
	"""
	tri = faces[face_indices]
	accum = np.zeros((n_vertices, 3), dtype=handle_disp.dtype)
	counts = np.zeros((n_vertices, 1), dtype=np.int32)
	for k in range(3):
		verts = tri[:, k]
		accum[verts] += handle_disp
		counts[verts] += 1
	b = np.where(counts[:, 0] > 0)[0]
	bc = accum[b] / counts[b]
	return b, bc


def solve_biharmonic(
	vertices: np.ndarray,
	faces: np.ndarray,
	b: np.ndarray,
	bc: np.ndarray,
	k: int = 2,
) -> np.ndarray:
	"""Solve a biharmonic displacement field with boundary constraints."""
	disp = igl.harmonic(vertices, faces, b, bc, k)
	return disp


def build_biharmonic_Q(vertices: np.ndarray, faces: np.ndarray) -> sparse.csr_matrix:
	"""Build the bi-Laplacian quadratic form Q = L * M^{-1} * L."""
	L = igl.cotmatrix(vertices, faces)
	M = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_VORONOI)
	m_diag = M.diagonal()
	m_diag_safe = np.where(m_diag > 0, m_diag, 1.0)
	Minv = sparse.diags(1.0 / m_diag_safe)
	Q = L @ Minv @ L
	return Q.tocsr()


def aggregate_barycentric_constraints(
	face_indices: np.ndarray,
	barycentric: np.ndarray,
	handle_disp: np.ndarray,
	decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Merge duplicate barycentric handles by averaging displacements."""
	rounded = np.round(barycentric, decimals=decimals)
	key = np.concatenate([face_indices[:, None], rounded], axis=1)
	dtype = np.dtype(
		[("f", np.int64), ("b0", np.float64), ("b1", np.float64), ("b2", np.float64)]
	)
	keys = np.rec.fromarrays(key.T, dtype=dtype)
	unique_keys, inv = np.unique(keys, return_inverse=True)
	m = unique_keys.shape[0]
	disp_accum = np.zeros((m, 3), dtype=handle_disp.dtype)
	counts = np.zeros((m, 1), dtype=np.int32)
	np.add.at(disp_accum, inv, handle_disp)
	np.add.at(counts, inv, 1)
	disp_avg = disp_accum / counts
	face_unique = np.array([uk[0] for uk in unique_keys], dtype=np.int32)
	bary_unique = np.vstack([(uk[1], uk[2], uk[3]) for uk in unique_keys]).astype(
		barycentric.dtype
	)
	return face_unique, bary_unique, disp_avg


def build_barycentric_Aeq(
	faces: np.ndarray,
	face_indices: np.ndarray,
	barycentric: np.ndarray,
) -> sparse.csr_matrix:
	"""Construct Aeq for barycentric handle constraints."""
	tri = faces[face_indices]
	rows = np.repeat(np.arange(face_indices.shape[0]), 3)
	cols = tri.reshape(-1)
	data = barycentric.reshape(-1)
	Aeq = sparse.coo_matrix(
		(data, (rows, cols)),
		shape=(face_indices.shape[0], faces.max() + 1),
	)
	return Aeq.tocsr()


def solve_biharmonic_barycentric(
	vertices: np.ndarray,
	faces: np.ndarray,
	face_indices: np.ndarray,
	barycentric: np.ndarray,
	handle_disp: np.ndarray,
) -> np.ndarray:
	"""Solve biharmonic displacement with barycentric equality constraints."""
	Q = build_biharmonic_Q(vertices, faces)
	n = vertices.shape[0]
	Aeq = build_barycentric_Aeq(faces, face_indices, barycentric)
	B = np.zeros((n, 3), dtype=np.float64)
	Beq = handle_disp.astype(np.float64, copy=False)
	b = np.array([], dtype=np.int64)
	bc = np.zeros((0, 3), dtype=np.float64)
	disp = igl.min_quad_with_fixed(Q.tocsc(), B, b, bc, Aeq.tocsc(), Beq, True)
	return disp


def transfer_with_vertex_handles(
	low_vertices: np.ndarray,
	low_def_vertices: np.ndarray,
	high_vertices: np.ndarray,
	high_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Transfer deformation using vertex handles.

	Returns
	-------
	high_def_vertices
		Deformed high-res vertices.
	disp
		Per-vertex displacement on the high-res mesh.
	handle_vertices
		Vertex handle indices on the high-res mesh.
	handle_disp
		Displacement at each handle.
	"""
	face_indices, _, barycentric = closest_point_handles(
		low_vertices, high_vertices, high_faces
	)
	handle_vertices = vertex_handle_indices(face_indices, barycentric, high_faces)
	handle_disp = low_def_vertices - low_vertices
	b, bc = aggregate_vertex_constraints(
		high_vertices.shape[0], handle_vertices, handle_disp
	)
	disp = solve_biharmonic(high_vertices, high_faces, b, bc, k=2)
	high_def_vertices = high_vertices + disp
	return high_def_vertices, disp, handle_vertices, handle_disp


def transfer_with_barycentric_handles(
	low_vertices: np.ndarray,
	low_def_vertices: np.ndarray,
	high_vertices: np.ndarray,
	high_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Transfer deformation using barycentric face handles."""
	face_indices, _, barycentric = closest_point_handles(
		low_vertices, high_vertices, high_faces
	)
	handle_disp = low_def_vertices - low_vertices
	face_u, bary_u, disp_u = aggregate_barycentric_constraints(
		face_indices, barycentric, handle_disp
	)
	disp = solve_biharmonic_barycentric(
		high_vertices, high_faces, face_u, bary_u, disp_u
	)
	high_def_vertices = high_vertices + disp
	return high_def_vertices, disp, face_u, disp_u, bary_u


# -----------------------------
# Blender <-> NumPy conversion
# -----------------------------

def mesh_to_numpy_world(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
	"""Convert a Blender mesh object to triangulated NumPy arrays in world space."""
	bm = bmesh.new()
	bm.from_mesh(obj.data)
	bmesh.ops.triangulate(bm, faces=bm.faces)
	bm.verts.ensure_lookup_table()
	bm.faces.ensure_lookup_table()

	verts = np.zeros((len(bm.verts), 3), dtype=np.float64)
	for i, v in enumerate(bm.verts):
		world_v = obj.matrix_world @ v.co
		verts[i] = (world_v.x, world_v.y, world_v.z)

	faces = np.array(
		[[v.index for v in f.verts] for f in bm.faces], dtype=np.int32
	)
	bm.free()
	return verts, faces


def create_mesh_object(
	name: str,
	vertices_world: np.ndarray,
	faces: np.ndarray,
	reference_obj: bpy.types.Object,
	context: bpy.types.Context,
) -> bpy.types.Object:
	"""Create a new mesh object from world-space vertices and faces."""
	inv = reference_obj.matrix_world.inverted()
	vertices_local = [inv @ Vector(v) for v in vertices_world]

	mesh = bpy.data.meshes.new(name)
	mesh.from_pydata([v[:] for v in vertices_local], [], faces.tolist())
	mesh.update()

	obj = bpy.data.objects.new(name, mesh)
	obj.matrix_world = reference_obj.matrix_world.copy()
	(context.collection or context.scene.collection).objects.link(obj)
	return obj


# -----------------------------
# UI and Operator
# -----------------------------

def _report_and_cancel(op: Operator, message: str):
	op.report({'ERROR'}, message)
	return {'CANCELLED'}


def _mesh_enum_items(self, context: bpy.types.Context):
	items = []
	for obj in context.scene.objects:
		if obj.type == 'MESH':
			items.append((obj.name, obj.name, ""))
	return items


def _same_topology(obj_a: bpy.types.Object, obj_b: bpy.types.Object) -> bool:
	if len(obj_a.data.vertices) != len(obj_b.data.vertices):
		return False
	if len(obj_a.data.polygons) != len(obj_b.data.polygons):
		return False
	polys_a = [tuple(p.vertices) for p in obj_a.data.polygons]
	polys_b = [tuple(p.vertices) for p in obj_b.data.polygons]
	return polys_a == polys_b


class TransferBiharmonicOperator(Operator):
	"""Transfer deformation from low-res to high-res using libigl."""

	bl_idname = "scene.transfer_biharmonic"
	bl_label = "Transfer Deformation (IGL)"
	bl_description = (
		"Select original/deformed/transfer-to meshes and transfer deformation. Can take ~1min for large meshes."
	)

	use_barycentric: BoolProperty(
		name="Use barycentric handles",
		description="Use barycentric face handles (more accurate, slower)",
		default=False,
	)

	def execute(self, context: bpy.types.Context):
		if igl is None:
			return _report_and_cancel(
				self,
				"libigl (python bindings) not available. Install it in Blender's Python environment.",
			)

		orig_name = context.scene.biharmonic_original_mesh
		def_name = context.scene.biharmonic_deformed_mesh
		transfer_name = context.scene.biharmonic_transfer_to_mesh
		orig_obj = context.scene.objects.get(orig_name) if orig_name else None
		def_obj = context.scene.objects.get(def_name) if def_name else None
		transfer_obj = context.scene.objects.get(transfer_name) if transfer_name else None
		if orig_obj is None or def_obj is None or transfer_obj is None:
			return _report_and_cancel(
				self, "Select original, deformed, and transfer-to meshes."
			)
		if orig_obj == def_obj:
			return _report_and_cancel(
				self, "Original and deformed meshes must be different objects."
			)
		if orig_obj == transfer_obj or def_obj == transfer_obj:
			return _report_and_cancel(
				self, "Transfer-to mesh must be different from original/deformed meshes."
			)

		if not _same_topology(orig_obj, def_obj):
			return _report_and_cancel(
				self,
				"Original and deformed meshes must have identical topology (same faces).",
			)

		low_v, _ = mesh_to_numpy_world(orig_obj)
		low_def_v, _ = mesh_to_numpy_world(def_obj)
		high_v, high_f = mesh_to_numpy_world(transfer_obj)

		if low_v.shape[0] != low_def_v.shape[0]:
			return _report_and_cancel(
				self,
				"Original and deformed meshes must have the same vertex count.",
			)

		self.report({'INFO'}, "Running biharmonic transfer...")
		if self.use_barycentric:
			high_def_v, _, _, _, _ = transfer_with_barycentric_handles(
				low_v, low_def_v, high_v, high_f
			)
		else:
			high_def_v, _, _, _ = transfer_with_vertex_handles(
				low_v, low_def_v, high_v, high_f
			)

		new_name = f"{transfer_obj.name}_deformed"
		create_mesh_object(new_name, high_def_v, high_f, transfer_obj, context)
		self.report({'INFO'}, f"Created {new_name}.")
		return {'FINISHED'}


class BiharmonicTransferPanel(Panel):
	"""Scene panel for deformation transfer."""

	bl_label = "Biharmonic Deformation Transfer"
	bl_idname = "SCENE_PT_biharmonic_transfer"
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = "scene"

	def draw(self, context: bpy.types.Context):
		layout = self.layout
		layout.prop(context.scene, "biharmonic_original_mesh")
		layout.prop(context.scene, "biharmonic_deformed_mesh")
		layout.prop(context.scene, "biharmonic_transfer_to_mesh")
		layout.prop(context.scene, "biharmonic_use_barycentric")
		op = layout.operator(
			TransferBiharmonicOperator.bl_idname, text="Transfer Deformation"
		)
		op.use_barycentric = context.scene.biharmonic_use_barycentric


def register():
	bpy.types.Scene.biharmonic_original_mesh = EnumProperty(
		name="Original",
		description="Original low-res mesh (must match deformed topology)",
		items=_mesh_enum_items,
	)
	bpy.types.Scene.biharmonic_deformed_mesh = EnumProperty(
		name="Deformed",
		description="Deformed low-res mesh (must match original topology)",
		items=_mesh_enum_items,
	)
	bpy.types.Scene.biharmonic_transfer_to_mesh = EnumProperty(
		name="Transfer-to",
		description="High-res mesh to deform (must be spatially aligned with original)",
		items=_mesh_enum_items,
	)
	bpy.types.Scene.biharmonic_use_barycentric = BoolProperty(
		name="Use barycentric handles",
		description="Use barycentric face handles (more accurate, slower)",
		default=False,
	)
	bpy.utils.register_class(TransferBiharmonicOperator)
	bpy.utils.register_class(BiharmonicTransferPanel)


def unregister():
	if hasattr(bpy.types.Scene, "biharmonic_original_mesh"):
		del bpy.types.Scene.biharmonic_original_mesh
	if hasattr(bpy.types.Scene, "biharmonic_deformed_mesh"):
		del bpy.types.Scene.biharmonic_deformed_mesh
	if hasattr(bpy.types.Scene, "biharmonic_transfer_to_mesh"):
		del bpy.types.Scene.biharmonic_transfer_to_mesh
	if hasattr(bpy.types.Scene, "biharmonic_use_barycentric"):
		del bpy.types.Scene.biharmonic_use_barycentric
	bpy.utils.unregister_class(BiharmonicTransferPanel)
	bpy.utils.unregister_class(TransferBiharmonicOperator)


if __name__ == "__main__":
	register()
