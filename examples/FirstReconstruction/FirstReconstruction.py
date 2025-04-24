#!/usr/bin/env python
import sys
import itk
from itk import RTK as rtk

if len(sys.argv) < 3:
    print("Usage: FirstReconstruction <outputimage> <outputgeometry>")
    sys.exit(1)

# Defines the image type
ImageType = itk.Image[itk.F, 3]

# Defines the RTK geometry object
geometry = rtk.ThreeDCircularProjectionGeometry.New()
numberOfProjections = 360
firstAngle = 0.0
angularArc = 360.0
sid = 600  # source to isocenter distance
sdd = 1200  # source to detector distance
for x in range(0, numberOfProjections):
    angle = firstAngle + x * angularArc / numberOfProjections
    geometry.AddProjection(sid, sdd, angle)

# Writing the geometry to disk
rtk.write_geometry(geometry, sys.argv[2])

# Create a stack of empty projection images
constantImageSource = rtk.constant_image_source(
    ttype=[ImageType],
    origin=[-127, -127, 0.0],
    spacing=[2.0, 2.0, 2.0],
    size=[128, 128, numberOfProjections],
    constant=0.0,
)

# Create projections of an ellipse
rei = rtk.ray_ellipsoid_intersection_image_filter(
    ttype=[ImageType, ImageType],
    input=constantImageSource,
    geometry=geometry,
    density=2,
    angle=0,
    center=[0, 0, 10],
    axis=[50, 50, 50],
)

# Create reconstructed image
constantImageSource2 = rtk.constant_image_source(
    ttype=[ImageType],
    origin=[-63.5, -63.5, -63.5],
    spacing=[1.0, 1.0, 1.0],
    size=[128, 128, 128],
    constant=0.0,
)

# FDK reconstruction
# Cannot use fully functional style here due to direct access to ramp filter
print("Reconstructing...")
feldkamp = rtk.FDKConeBeamReconstructionFilter[ImageType].New()
feldkamp.SetInput(0, constantImageSource2)
feldkamp.SetInput(1, rei)
feldkamp.SetGeometry(geometry)
feldkamp.GetRampFilter().SetTruncationCorrection(0.0)
feldkamp.GetRampFilter().SetHannCutFrequency(0.0)

# Field-of-view masking
fieldofview = rtk.field_of_view_image_filter(
    input=feldkamp.GetOutput(), projections_stack=rei, geometry=geometry
)

# Writer
print("Writing output image...")
itk.imwrite(fieldofview, sys.argv[1])

print("Done!")
