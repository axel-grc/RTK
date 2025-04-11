#!/usr/bin/env python
import sys
import argparse
import itk
from itk import RTK as rtk

def main():
  parser = argparse.ArgumentParser(
    description="Performs an iterative 3D reconstruction with Daubechies wavelets regularization.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  # GGO options
  parser.add_argument("--verbose", "-v", help="Verbose execution", action="store_true")
  parser.add_argument("--config", help="Config file", type=str)
  parser.add_argument("--geometry", "-g", help="XML geometry file name", type=str, required=True)
  parser.add_argument("--output", "-o", help="Output file name", type=str, required=True)
  parser.add_argument("--niterations", "-n", help="Number of iterations", type=int, default=1)
  parser.add_argument("--alpha", help="Regularization parameter", type=float, default=0.1)
  parser.add_argument("--beta", help="Augmented Lagrangian constraint multiplier", type=float, default=1)
  parser.add_argument("--CGiter", help="Number of nested iterations of conjugate gradient", type=int, default=5)
  parser.add_argument("--order", help="The order of the Daubechies wavelets", type=int, default=3)
  parser.add_argument("--levels", help="The number of decomposition levels in the wavelets transform", type=int, default=5)
  parser.add_argument("--input", "-i", help="Input volume", type=str)
  parser.add_argument("--nodisplaced", help="Disable the displaced detector filter", action="store_true")

  # Add additional RTK output image options if needed
  rtk.add_rtk3Doutputimage_group(parser)
  rtk.add_rtkinputprojections_group(parser)
  rtk.add_rtkiterations_group(parser)
  rtk.add_rtkprojectors_group(parser)

  args_info = parser.parse_args()

  # Define output pixel type and dimension
  OutputPixelType = itk.F
  Dimension = 3
  OutputImageType = itk.Image[OutputPixelType, Dimension]

  # Projections reader
  projectionsReader = rtk.ProjectionsReader[OutputImageType].New()
  rtk.SetProjectionsReaderFromArgParse(projectionsReader, args_info)

  if args_info.verbose:
    print(f"Reading projections...")
  projectionsReader.Update()

  # Geometry
  if args_info.verbose:
    print(f"Reading geometry information from {args_info.geometry}...")

  geometry = rtk.read_geometry(args_info.geometry)

  # Displaced detector weighting
  ddf = rtk.DisplacedDetectorImageFilter[OutputImageType].New()
  ddf.SetInput(projectionsReader.GetOutput())
  ddf.SetGeometry(geometry)

  # Create input volume: either an existing volume read from a file or a blank image
  if args_info.input:
    inputFilter = itk.imread(args_info.input)
  else:
    inputFilter = rtk.ConstantImageSource[OutputImageType].New()
    rtk.SetConstantImageSourceFromArgParse(inputFilter, args_info)

  # Setup the ADMM wavelets reconstruction filter
  admmFilter = rtk.ADMMWaveletsConeBeamReconstructionFilter[OutputImageType].New()
  if hasattr(itk, 'CudaImage'):
    admmFilter.SetInput(0, itk.cuda_image_from_image(inputFilter.GetOutput()))
    admmFilter.SetInput(1, itk.cuda_image_from_image(projectionsReader.GetOutput()))
  else:
    admmFilter.SetInput(0, inputFilter.GetOutput())
    admmFilter.SetInput(1, projectionsReader.GetOutput())

  # Set the forward and back projection filters to be used inside admmFilte
  rtk.SetForwardProjectionFromArgParse(admmFilter, args_info)
  rtk.SetBackProjectionFromArgParse(admmFilter, args_info)

  # Set the geometry and interpolation weights
  admmFilter.SetGeometry(geometry)

  # Set numerical parameters
  admmFilter.SetCG_iterations(args_info.CGiter)
  admmFilter.SetAL_iterations(args_info.niterations)
  admmFilter.SetAlpha(args_info.alpha)
  admmFilter.SetBeta(args_info.beta)
  admmFilter.SetNumberOfLevels(args_info.levels)
  admmFilter.SetOrder(args_info.order)

  admmFilter.SetDisableDisplacedDetectorFilter(args_info.nodisplaced)

  rtk.SetIterationsReportFromArgParse(args_info,admmFilter)

  admmFilter.Update()

  # Write the output volume
  itk.imwrite(admmFilter.GetOutput(), args_info.output)

  if args_info.verbose:
      print("Processing completed successfully.")

if __name__ == "__main__":
    main()