# Biharmonic Deformation Transfer (Blender Add-on)

Transfers deformations from a low-res mesh to a high-res mesh using libigl’s biharmonic solve.

## Install
1. In Blender, open **Edit → Preferences → Extensions**.
2. Click **Install from Disk…** and select the platform-specific zip:
   - macOS (Intel): `biharmonic_deformation_transfer-0.1.0-macos_x64.zip`
   - macOS (Apple Silicon): `biharmonic_deformation_transfer-0.1.0-macos_arm64.zip`
   - Windows: `biharmonic_deformation_transfer-0.1.0-windows_x64.zip`
3. Enable the add-on.

## Use
1. Select three mesh objects:
   - Low-res mesh (original)
   - High-res mesh (target)
   - Low-res deformed mesh (**active object**)
2. Go to **Scene Properties → Biharmonic Deformation Transfer**.
3. Click **Transfer Deformation (IGL)**.

A new mesh named `<HighResName>_deformed` will be created in the scene.
