import argparse
import itk
from itk import RTK as rtk

def build_parser():
    parser = argparse.ArgumentParser(
        description="Computes expected photon counts from incident spectrum, material attenuations, detector response and material-decomposed projections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose execution")
    parser.add_argument("--output", "-o", type=str, help="Output file name (high and low energy projections)", required=True)
    parser.add_argument("--input", "-i", type=str, help="Material-decomposed projections", required=True)
    parser.add_argument("--high", type=str, help="Incident spectrum image, high energy", required=True)
    parser.add_argument("--low", type=str, help="Incident spectrum image, low energy", required=True)
    parser.add_argument("--detector", "-d", type=str, help="Detector response file")
    parser.add_argument("--attenuations", "-a", type=str, help="Material attenuations file", required=True)
    parser.add_argument("--variances", type=str, help="Output variances of measured energies, file name")
    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    PixelType = itk.F
    Dimension = 3

    # Read all inputs
    if args.verbose:
        print(f"Reading decomposed projections: {args.input}")
    decomposedProjectionReader = itk.imread(args.input, itk.VectorImage[PixelType, Dimension])

    if args.verbose:
        print(f"Reading high energy spectrum: {args.high}")
    incidentSpectrumReaderHighEnergy = itk.imread(args.high, PixelType)

    if args.verbose:
        print(f"Reading low energy spectrum: {args.low}")
    incidentSpectrumReaderLowEnergy = itk.imread(args.low, PixelType)

    if args.verbose:
        print(f"Reading material attenuations: {args.attenuations}")
    materialAttenuationsReader = itk.imread(args.attenuations, PixelType)

    # Get parameters from the images
    MaximumEnergy = incidentSpectrumReaderHighEnergy.GetLargestPossibleRegion().GetSize()[0]

    detectorResponseReader = rtk.DetectorResponseReaderType.New()

    # If the detector response is given by the user, use it. Otherwise, assume it is included in the
    # incident spectrum, and fill the response with ones
    if args.detector:
        detectorResponseReader.SetXMLFileName(args.detector)
        detectorResponseReader.Update()
        detectorImage = detectorResponseReader.GetOutput()
    else:
        detectorSource = rtk.ConstantImageSource[rtk.Image[PixelType, Dimension - 1]].New()
        detectorSource.SetSize(itk.make_size(1, MaximumEnergy))
        detectorSource.SetConstant(1.0)
        detectorSource.Update()
        detectorImage = detectorSource.GetOutput()

    # Generate a set of zero-filled intensity projections
    dualEnergyProjections = itk.VectorImage[PixelType, Dimension].New()
    dualEnergyProjections.CopyInformation(decomposedProjectionReader)
    dualEnergyProjections.SetVectorLength(2)
    dualEnergyProjections.Allocate()

    # Check that the inputs have the expected size
    if decomposedProjectionReader.GetNumberOfComponentsPerPixel() != 2:
        raise RuntimeError(f"Decomposed projections image has vector length {decomposedProjectionReader.GetNumberOfComponentsPerPixel()}, should be 2")
    if materialAttenuationsReader.GetLargestPossibleRegion().GetSize()[1] != MaximumEnergy:
        raise RuntimeError(f"Material attenuations image has {materialAttenuationsReader.GetLargestPossibleRegion().GetSize()[1]} energies, should have {MaximumEnergy}")

    # Create and set the filter
    forward = rtk.SpectralForwardModelImageFilter[
        type(decomposedProjectionReader), type(dualEnergyProjections).New()
    ]

    forward.SetInputDecomposedProjections(decomposedProjectionReader)
    forward.SetInputMeasuredProjections(dualEnergyProjections)
    forward.SetInputIncidentSpectrum(incidentSpectrumReaderHighEnergy)
    forward.SetInputSecondIncidentSpectrum(incidentSpectrumReaderLowEnergy)
    forward.SetDetectorResponse(detectorImage)
    forward.SetMaterialAttenuations(materialAttenuationsReader)
    forward.SetIsSpectralCT(False)
    forward.SetComputeVariances(bool(args.variances))

    if args.verbose:
        print("Running forward model filter...")
    forward.Update()

    # Write output
    if args.verbose:
        print(f"Writing output projections: {args.output}")
    itk.imwrite(forward.GetOutput(), args.output)

    # If requested, write the variances
    if args.variances:
        if args.verbose:
            print(f"Writing output variances: {args.variances}")
        itk.imwrite(forward.GetOutputVariances(), args.variances)

    if args.verbose:
        print("Done.")

if __name__ == "__main__":
    main()