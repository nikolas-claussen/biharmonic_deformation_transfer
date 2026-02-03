# Build instructions for the Blender extension zips

If you edit the add-on code, rebuild the platform zips using Blender’s CLI. The extension source lives in the `biharmonic_deformation_transfer/` folder and the output zips go to `dist/`.

## 1) Update Python code
Edit the add-on in:
- `biharmonic_deformation_transfer/__init__.py`

## 2) (Optional) Refresh wheels
If you need to update dependencies, download wheels into `biharmonic_deformation_transfer/wheels/`:

```zsh
WHEELS=/Users/nc1333/Documents/Personal/blender_cecile/biharmonic_deformation_transfer/wheels

# macOS arm64
python -m pip download --only-binary=:all: --no-deps --platform macosx_11_0_arm64 --python-version 311 --implementation cp --abi cp311 -d "$WHEELS" libigl
python -m pip download --only-binary=:all: --no-deps --platform macosx_11_0_arm64 --python-version 311 --implementation cp --abi cp311 -d "$WHEELS" numpy
python -m pip download --only-binary=:all: --no-deps --platform macosx_12_0_arm64 --python-version 311 --implementation cp --abi cp311 -d "$WHEELS" scipy

# macOS x64
python -m pip download --only-binary=:all: --no-deps --platform macosx_10_9_x86_64 --python-version 311 --implementation cp --abi cp311 -d "$WHEELS" libigl
python -m pip download --only-binary=:all: --no-deps --platform macosx_10_9_x86_64 --python-version 311 --implementation cp --abi cp311 -d "$WHEELS" numpy
python -m pip download --only-binary=:all: --no-deps --platform macosx_10_9_x86_64 --python-version 311 --implementation cp --abi cp311 -d "$WHEELS" scipy

# Windows x64
python -m pip download --only-binary=:all: --no-deps --platform win_amd64 --python-version 311 --implementation cp --abi cp311 -d "$WHEELS" libigl
python -m pip download --only-binary=:all: --no-deps --platform win_amd64 --python-version 311 --implementation cp --abi cp311 -d "$WHEELS" numpy
python -m pip download --only-binary=:all: --no-deps --platform win_amd64 --python-version 311 --implementation cp --abi cp311 -d "$WHEELS" scipy
```

## 3) Build zips

```zsh
/Applications/Blender.app/Contents/MacOS/Blender \
  --command extension build \
  --source-dir /Users/nc1333/Documents/Personal/blender_cecile/biharmonic_deformation_transfer \
  --split-platforms \
  --output-dir /Users/nc1333/Documents/Personal/blender_cecile/dist \
  --verbose
```

## 4) Validate zips

```zsh
/Applications/Blender.app/Contents/MacOS/Blender --command extension validate /Users/nc1333/Documents/Personal/blender_cecile/dist/biharmonic_deformation_transfer-0.1.0-macos_x64.zip
/Applications/Blender.app/Contents/MacOS/Blender --command extension validate /Users/nc1333/Documents/Personal/blender_cecile/dist/biharmonic_deformation_transfer-0.1.0-macos_arm64.zip
/Applications/Blender.app/Contents/MacOS/Blender --command extension validate /Users/nc1333/Documents/Personal/blender_cecile/dist/biharmonic_deformation_transfer-0.1.0-windows_x64.zip
```
