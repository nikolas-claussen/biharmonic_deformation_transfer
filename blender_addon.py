"""Blender add-on: Biharmonic deformation transfer using libigl.

Select three meshes: low-res (B), high-res (A), and low-res deformed (C).
The active object must be the deformed low-res mesh (C). The add-on infers
low/high by vertex count and creates a deformed high-res mesh.
"""

from __future__ import annotations

import numpy as np

import bpy
import bmesh
from bpy.types import Operator, Panel
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


def transfer_with_barycentric_handles(
	low_vertices: np.ndarray,
	low_def_vertices: np.ndarray,
	high_vertices: np.ndarray,
	high_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Transfer deformation using barycentric face handles.

	Returns
	-------
	high_def_vertices
		Deformed high-res vertices.
	disp
		Per-vertex displacement on the high-res mesh.
	face_indices
		Face index for each handle on the high-res mesh.
	handle_disp
		Displacement at each handle.
	barycentric
		Barycentric weights for each handle.
	"""
	face_indices, _, barycentric = closest_point_handles(
		low_vertices, high_vertices, high_faces
	)
	handle_disp = low_def_vertices - low_vertices
	b, bc = face_to_vertex_constraints(
		high_vertices.shape[0], face_indices, high_faces, handle_disp
	)
	disp = solve_biharmonic(high_vertices, high_faces, b, bc, k=2)
	high_def_vertices = high_vertices + disp
	return high_def_vertices, disp, face_indices, handle_disp, barycentric


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


def _infer_mesh_roles(
	op: Operator,
	context: bpy.types.Context,
) -> tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object] | set:
	selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
	if len(selected) != 3:
		return _report_and_cancel(op, "Select exactly three mesh objects.")

	active = context.view_layer.objects.active
	if active is None or active.type != 'MESH' or active not in selected:
		return _report_and_cancel(op, "Active object must be the deformed low-res mesh.")

	remaining = [obj for obj in selected if obj != active]
	if len(remaining) != 2:
		return _report_and_cancel(op, "Select exactly three meshes with one active.")

	active_count = len(active.data.vertices)
	rem_counts = [len(obj.data.vertices) for obj in remaining]

	low = None
	high = None
	for obj in remaining:
		if len(obj.data.vertices) == active_count:
			low = obj
		elif len(obj.data.vertices) > active_count:
			high = obj

	if low is None or high is None:
		return _report_and_cancel(
			op,
			"Could not infer low/high meshes. Ensure active is low-res deformed and the other two are low/high.",
		)

	return low, active, high


class TransferBiharmonicOperator(Operator):
	"""Transfer deformation from low-res to high-res using libigl."""

	bl_idname = "scene.transfer_biharmonic"
	bl_label = "Transfer Deformation (IGL)"

	def execute(self, context: bpy.types.Context):
		if igl is None:
			return _report_and_cancel(
				self,
				"libigl (python bindings) not available. Install it in Blender's Python environment.",
			)

		result = _infer_mesh_roles(self, context)
		if isinstance(result, set):
			return result
		low_obj, low_def_obj, high_obj = result

		low_v, _ = mesh_to_numpy_world(low_obj)
		low_def_v, _ = mesh_to_numpy_world(low_def_obj)
		high_v, high_f = mesh_to_numpy_world(high_obj)

		if low_v.shape[0] != low_def_v.shape[0]:
			return _report_and_cancel(
				self,
				"Low-res and deformed low-res meshes must have the same vertex count.",
			)

		self.report({'INFO'}, "Running biharmonic transfer...")
		high_def_v, _, _, _, _ = transfer_with_barycentric_handles(
			low_v, low_def_v, high_v, high_f
		)

		new_name = f"{high_obj.name}_deformed"
		create_mesh_object(new_name, high_def_v, high_f, high_obj, context)
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
		layout.operator(TransferBiharmonicOperator.bl_idname, text="Transfer Deformation (IGL)")


def register():
	bpy.utils.register_class(TransferBiharmonicOperator)
	bpy.utils.register_class(BiharmonicTransferPanel)


def unregister():
	bpy.utils.unregister_class(BiharmonicTransferPanel)
	bpy.utils.unregister_class(TransferBiharmonicOperator)


if __name__ == "__main__":
	register()
