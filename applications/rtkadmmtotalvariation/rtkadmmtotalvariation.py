#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Perform ADMM total variation reconstruction on cone-beam projections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General options
    parser.add_argument('--verbose', '-v', help="Verbose execution", action='store_true')
    parser.add_argument('--output', '-o', help="Output reconstructed volume file name", type=str, required=True)
    parser.add_argument('--input', '-i', help="Input initial volume file name (optional)", type=str)
    parser.add_argument('--projections', '-p', help="Projections file name", type=str, required=True)
    parser.add_argument('--geometry', '-g', help="XML geometry file name", type=str, required=True)
    parser.add_argument('--phases', help="Phase gating file", type=str)
    parser.add_argument('--windowwidth', help="Gating window width", type=float, default=1.0)
    parser.add_argument('--windowcenter', help="Gating window center", type=float, default=0.5)
    parser.add_argument('--windowshape', help="Gating window shape", type=str, default="hann")
    parser.add_argument('--CGiter', help="Number of CG iterations", type=int, default=10)
    parser.add_argument('--niterations', help="Number of AL iterations", type=int, default=10)
    parser.add_argument('--alpha', help="Regularization parameter alpha", type=float, default=0.1)
    parser.add_argument('--beta', help="Regularization parameter beta", type=float, default=0.1)
    parser.add_argument('--nodisplaced', help="Disable displaced detector correction", action='store_true')

    # RTK specific groups
    rtk.add_rtkprojectors_group(parser)
    rtk.add_rtk3Doutputimage_group(parser)

    # Parse arguments
    args_info = parser.parse_args()

    # Define output pixel type and dimension
    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.CudaImage[OutputPixelType, Dimension]
    GradientOutputImageType = itk.CudaImage[itk.CovariantVector[OutputPixelType, Dimension], Dimension]

    # Projections reader
    projectionsReader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(projectionsReader, args_info)

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry from {args_info.geometry}...")

    geometry = rtk.ReadGeometry(args_info.geometry)

    # Phase gating weights reader
    phaseGating = rtk.PhaseGatingImageFilter[OutputImageType].New()
    if args_info.phases:
        phaseGating.SetPhasesFileName(args_info.phases)
        phaseGating.SetGatingWindowWidth(args_info.windowwidth)
        phaseGating.SetGatingWindowCenter(args_info.windowcenter)
        phaseGating.SetGatingWindowShape(args_info.windowshape)
        phaseGating.SetInputProjectionStack(projectionsReader.GetOutput())
        phaseGating.SetInputGeometry(geometry)

        phaseGating.Update()

    # Create input: either an existing volume read from a file or a blank image
    if args_info.input:
        # Read an existing image to initialize the volume
        inputReader = itk.ImageFileReader[OutputImageType].New()
        inputReader.SetFileName(args_info.input)
        inputFilter = inputReader
    else:
        # Create new empty volume
        constantImageSource = rtk.ConstantImageSource[OutputImageType].New()
        inputFilter = constantImageSource

    # Setup the ADMM filter and run it
    admmFilter = rtk.ADMMTotalVariationConeBeamReconstructionFilter[OutputImageType, GradientOutputImageType].New()

    # Set the forward and back projection filters to be used inside admmFilter
    rtk.SetForwardProjectionFromArgParse(args_info, admmFilter)
    rtk.SetBackProjectionFromArgParse(args_info, admmFilter)

    # Set all four numerical parameters
    admmFilter.SetCG_iterations(args_info.CGiter)
    admmFilter.SetAL_iterations(args_info.niterations)
    admmFilter.SetAlpha(args_info.alpha)
    admmFilter.SetBeta(args_info.beta)
    admmFilter.SetDisableDisplacedDetectorFilter(args_info.nodisplaced)

    # Set inputs
    admmFilter.SetInput(0, inputFilter.GetOutput())
    if args_info.phases:
        admmFilter.SetInput(1, phaseGating.GetOutput())
        admmFilter.SetGeometry(phaseGating.GetOutputGeometry())
        admmFilter.SetGatingWeights(phaseGating.GetGatingWeightsOnSelectedProjections())
    else:
        admmFilter.SetInput(1, projectionsReader.GetOutput())
        admmFilter.SetGeometry(geometry)

    # Run the reconstruction
    if args_info.verbose:
        print("Running ADMM total variation reconstruction...")

    try:
        admmFilter.Update()
    except Exception as e:
        print(f"Error during reconstruction: {e}", file=sys.stderr)
        sys.exit(1)

    # Write the output
    if args_info.verbose:
        print(f"Writing output to {args_info.output}...")

    try:
        itk.imwrite(admmFilter.GetOutput(), args_info.output)
    except Exception as e:
        print(f"Error while writing image: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
