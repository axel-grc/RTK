#!/usr/bin/env python
import os
import sys
import itk
from itk import RTK as rtk

if len(sys.argv) < 3:
    print("Usage: FirstReconstruction <outputimage> <outputgeometry>")
    sys.exit(1)

# Add CUDA path for DLLs on Windows (required for Python ≥3.8)
if sys.platform == "win32":
    os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))

# Image types
GPUImageType = rtk.CudaImage[itk.F, 3]
CPUImageType = itk.Image[itk.F, 3]

# Parameters
numberOfProjections = 360
firstAngle = 0.0
angularArc = 360.0
sid = 600
sdd = 1200

# Defines the RTK geometry object
geometry = rtk.ThreeDCircularProjectionGeometry.New()
for x in range(numberOfProjections):
    angle = firstAngle + x * angularArc / numberOfProjections
    geometry.AddProjection(sid, sdd, angle)
rtk.write_geometry(geometry, sys.argv[2])

# Create a stack of empty projection images
constantImageSource = rtk.constant_image_source(
    ttype=[GPUImageType],
    origin=[-127, -127, 0.0],
    spacing=[2.0, 2.0, 2.0],
    size=[128, 128, numberOfProjections],
    constant=0.0,
)

# Create simulated ellipsoid projections (on CPU)
rei = rtk.ray_ellipsoid_intersection_image_filter(
    ttype=[CPUImageType, CPUImageType],
    density=2,
    angle=0,
    center=[0, 0, 10],
    axis=[50, 50, 50],
    geometry=geometry,
    input=constantImageSource,
)
# Create reconstructed image
constantImageSource2 = rtk.constant_image_source(
    ttype=[GPUImageType],
    origin=[-63.5, -63.5, -63.5],
    spacing=[1.0, 1.0, 1.0],
    size=[128, 128, 128],
    constant=0.0,
)

# Graft the projections to an itk::CudaImage
projections = GPUImageType.New()
rei.Update()
projections.SetPixelContainer(rei.GetPixelContainer())
projections.CopyInformation(rei)
projections.SetBufferedRegion(rei.GetBufferedRegion())
projections.SetRequestedRegion(rei.GetRequestedRegion())

# FDK reconstruction
# Cannot use fully functional style here due to direct access to ramp filter
print("Reconstructing...")
feldkamp = rtk.CudaFDKConeBeamReconstructionFilter.New()
feldkamp.SetInput(0, constantImageSource2)
feldkamp.SetInput(1, projections)
feldkamp.SetGeometry(geometry)
feldkamp.GetRampFilter().SetTruncationCorrection(0.0)
feldkamp.GetRampFilter().SetHannCutFrequency(0.0)

# Field-of-view masking
fieldofview = rtk.field_of_view_image_filter(
    input=feldkamp, projections_stack=rei, geometry=geometry
)

# Save result
print("Writing output image...")
itk.imwrite(fieldofview, sys.argv[1])
print("Done!")
