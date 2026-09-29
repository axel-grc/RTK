#!/usr/bin/env bash

# Build ITK with RTK as a remote module and install the Python bindings into a
# self-contained folder usable via PYTHONPATH.
# Edit the configuration variables below, then run it.

exec > >(tee "$(basename "$0" .sh).log")
exec 2> >(tee "$(basename "$0" .sh).err" >&2)

set -e
set -x

# ==================== Config (edit these) ====================
ITK_SRC=/path/to/ITK
RTK_SRC=/path/to/RTK
DEST_DIR=/path/to/bindings
NTHREADS=${NTHREADS:-24}
RTK_USE_CUDA=OFF
EDITABLE=OFF       # symlink the RTK Python files back to $RTK_SRC
# =============================================================

BUILD_DIR=PythonWrapping
INSTALL_PATH=${BUILD_DIR}-install

echo "ITK source:    $ITK_SRC"
echo "RTK source:    $RTK_SRC"
echo "Install folder: $DEST_DIR"
echo "Editable:      $EDITABLE"
echo "RTK_USE_CUDA:  $RTK_USE_CUDA"

mkdir -p "$ITK_SRC/Modules/Remote"
ln -sfn "$RTK_SRC" "$ITK_SRC/Modules/Remote/RTK"

CMAKE_EXTRA=
if [ "$RTK_USE_CUDA" = "ON" ]; then
  CMAKE_EXTRA=-DModule_CudaCommon:BOOL=ON
fi

mkdir -p "$BUILD_DIR" "$DEST_DIR"

cmake "$ITK_SRC" \
  -B"$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTING=OFF \
  -DITK_BUILD_DEFAULT_MODULES=ON \
  -DITK_WRAP_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
  -DPY_SITE_PACKAGES_PATH:STRING="$DEST_DIR" \
  -DITK_WRAP_unsigned_short:BOOL=ON \
  -DITK_WRAP_double:BOOL=ON \
  -DITK_WRAP_complex_double:BOOL=ON \
  -DITK_WRAP_IMAGE_DIMS:STRING="2;3;4" \
  -DModule_RTK:BOOL=ON \
  -DModule_RTK_GIT_TAG:STRING="" \
  -DRTK_USE_CUDA:BOOL="$RTK_USE_CUDA" \
  -DRTK_BUILD_APPLICATIONS:BOOL=OFF \
  -DRTK_EDITABLE_PYTHON_PACKAGES:BOOL="$EDITABLE" \
  $CMAKE_EXTRA

cmake --build "$BUILD_DIR" --parallel "$NTHREADS" -- -k
cmake --install "$BUILD_DIR"

echo "Done. Export $DEST_DIR for this session, or add it permanently to your shell profile:"
echo "  export PYTHONPATH=$DEST_DIR:\$PYTHONPATH"
echo "Then run the RTK applications as Python modules, e.g.:"
echo "  python3 -m itk.rtkfdk -g geometry.xml --path . --regexp *.mha -o output.mha"
