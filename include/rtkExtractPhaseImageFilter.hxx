/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkExtractPhaseImageFilter_hxx
#define rtkExtractPhaseImageFilter_hxx


#include "rtkHilbertImageFilter.h"

#include <itkImageRegionIteratorWithIndex.h>
#include <itkComplexToPhaseImageFilter.h>
#include <itkConvolutionImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkConvolutionImageFilter.h>
#include <itkNumericTraits.h>
namespace rtk
{

template <class TImage>
ExtractPhaseImageFilter<TImage>::ExtractPhaseImageFilter()
  : m_MovingAverageSize(1)
  , m_UnsharpMaskSize(55)
  , m_Model(LINEAR_BETWEEN_MINIMA)
{}


// Helper function to compute padded region for FFT (product of 2,3,5)
template <typename TImage>
typename TImage::RegionType
GetPaddedImageRegion(const typename TImage::RegionType & inputRegion)
{
  typename TImage::RegionType paddedRegion = inputRegion;
  using SizeType = typename TImage::SizeType;
  SizeType size = inputRegion.GetSize();
  int      xPaddedSize = size[0];
  auto     is_valid = [](int x) {
    while (x % 2 == 0)
      x /= 2;
    while (x % 3 == 0)
      x /= 3;
    while (x % 5 == 0)
      x /= 5;
    return x == 1;
  };
  while (!is_valid(xPaddedSize))
    ++xPaddedSize;
  paddedRegion.SetSize(0, xPaddedSize);
  paddedRegion.SetIndex(0, inputRegion.GetIndex(0));
  return paddedRegion;
}

template <class TImage>
void
ExtractPhaseImageFilter<TImage>::GenerateData()
{
  // Pad input signal to valid FFT size only if FFTW is not available
  const TImage *              inputRaw = this->GetInput();
  typename TImage::Pointer    input = const_cast<TImage *>(inputRaw);
  int                         orig_size = input->GetLargestPossibleRegion().GetSize()[0];
  typename TImage::RegionType region = input->GetLargestPossibleRegion();
#if !defined(USE_FFTWF) && !defined(USE_FFTWD)
  typename TImage::RegionType padded_region = GetPaddedImageRegion<TImage>(region);
  int                         padded_size = padded_region.GetSize()[0];
  if (padded_size != orig_size)
  {
    typename TImage::Pointer padded = TImage::New();
    padded->SetRegions(padded_region);
    padded->Allocate();
    padded->FillBuffer(0);
    // Copy original data
    for (int i = 0; i < orig_size; ++i)
      padded->SetPixel({ i }, input->GetPixel({ i }));
    input = padded;
  }
#endif

  // Moving average
  typename TImage::SizeType kernelSz;
  kernelSz[0] = m_MovingAverageSize;

  auto kernel = TImage::New();
  kernel->SetRegions(kernelSz);
  kernel->Allocate();
  kernel->FillBuffer(1. / m_MovingAverageSize);

  using ConvolutionType = itk::ConvolutionImageFilter<TImage, TImage>;
  auto conv = ConvolutionType::New();
  conv->SetInput(input);
  conv->SetKernelImage(kernel);

  // Unsharp mask
  kernelSz[0] = m_UnsharpMaskSize;
  auto kernel2 = TImage::New();
  kernel2->SetRegions(kernelSz);
  kernel2->Allocate();
  kernel2->FillBuffer(1. / m_UnsharpMaskSize);

  auto conv2 = ConvolutionType::New();
  conv2->SetInput(conv->GetOutput());
  conv2->SetKernelImage(kernel2);

  // Define real image type for FFT
  using RealOutputPixelType = typename itk::NumericTraits<typename TImage::PixelType>::RealType;
  using RealOutputImageType = itk::Image<RealOutputPixelType, 1>;

  auto sub = itk::SubtractImageFilter<TImage, TImage, RealOutputImageType>::New();
  sub->SetInput1(conv->GetOutput());
  sub->SetInput2(conv2->GetOutput());
  sub->InPlaceOff();
  sub->Update();

  // Define FFT output type (complex)
  using ComplexSignalType = itk::Image<std::complex<RealOutputPixelType>, 1>;

  // Hilbert transform
  auto hilbert = rtk::HilbertImageFilter<RealOutputImageType, ComplexSignalType>::New();
  hilbert->SetInput(sub->GetOutput());

  // Take the linear phase of this signal
  auto phase = itk::ComplexToPhaseImageFilter<ComplexSignalType, TImage>::New();
  phase->SetInput(hilbert->GetOutput());
  phase->Update();

  // Find extrema at 0 and pi
  m_MinimaPositions.clear();
  m_MaximaPositions.clear();
  using IteratorType = typename itk::ImageRegionIteratorWithIndex<TImage>;
  IteratorType               it(phase->GetOutput(), phase->GetOutput()->GetLargestPossibleRegion());
  typename TImage::PixelType curr, prev = it.Get();
  ++it;
  int prevType = 0; // 1 if maximum, 2 if minimum
  while (!it.IsAtEnd())
  {
    curr = it.Get();
    if (curr * prev <= 0.)
    {
      if (prevType != 1 && curr >= 0. && curr - prev <= itk::Math::pi)
      {
        m_MaximaPositions.push_back(it.GetIndex()[0]);
        prevType = 1;
      }
      else if (prevType != 2 && curr < 0. && prev - curr > itk::Math::pi)
      {
        m_MinimaPositions.push_back(it.GetIndex()[0]);
        prevType = 2;
      }
    }
    ++it;
    prev = curr;
  }

  if (m_MinimaPositions.empty() || m_MaximaPositions.empty())
    itkExceptionMacro(<< "Problem detecting extremas");

  const typename TImage::PixelType * sig = input->GetBufferPointer();
  int                                nsig = orig_size;

  // Find minimum between two maxima
  int currMinPos = 0;
  if (m_MinimaPositions[currMinPos] < m_MaximaPositions.front())
  {
    // Before first maximum
    for (int i = 0; i < m_MaximaPositions.front(); i++)
      if (sig[i] < sig[m_MinimaPositions[currMinPos]])
        m_MinimaPositions[currMinPos] = i;
    currMinPos++;
  }
  for (unsigned int j = 1; j < m_MaximaPositions.size(); j++, currMinPos++)
  {
    // Between maxima
    for (int i = m_MaximaPositions[j - 1]; i < m_MaximaPositions[j]; i++)
      if (sig[i] < sig[m_MinimaPositions[currMinPos]])
        m_MinimaPositions[currMinPos] = i;
  }
  if (m_MinimaPositions.back() > m_MaximaPositions.back())
  {
    // After last maximum
    for (int i = m_MaximaPositions.back(); i < nsig; i++)
      if (sig[i] < sig[m_MinimaPositions[currMinPos]])
        m_MinimaPositions[currMinPos] = i;
  }

  // Find maximum between two minima
  int currMaxPos = 0;
  if (m_MaximaPositions[currMaxPos] < m_MinimaPositions.front())
  {
    // Before first minimum
    for (int i = 0; i < m_MinimaPositions.front(); i++)
      if (sig[i] > sig[m_MaximaPositions[currMaxPos]])
        m_MaximaPositions[currMaxPos] = i;
    currMaxPos++;
  }
  for (unsigned int j = 1; j < m_MinimaPositions.size(); j++, currMaxPos++)
  {
    // Between minima
    for (int i = m_MinimaPositions[j - 1]; i < m_MinimaPositions[j]; i++)
      if (sig[i] > sig[m_MaximaPositions[currMaxPos]])
        m_MaximaPositions[currMaxPos] = i;
  }
  if (m_MaximaPositions.back() > m_MinimaPositions.back())
  {
    // After last minimum
    for (int i = m_MinimaPositions.back(); i < nsig; i++)
      if (sig[i] > sig[m_MaximaPositions[currMaxPos]])
        m_MaximaPositions[currMaxPos] = i;
  }

  switch (m_Model)
  {
    case (LOCAL_PHASE):
      it.GoToBegin();
      while (!it.IsAtEnd())
      {
        curr = it.Get() / (2. * itk::Math::pi);
        curr -= itk::Math::Floor<typename TImage::PixelType>(curr);
        it.Set(curr);
        ++it;
      }
      // Crop output to original size
      {
        typename TImage::RegionType crop_region;
        crop_region.SetIndex(0, 0);
        crop_region.SetSize(0, orig_size);
        auto extract = itk::ExtractImageFilter<TImage, TImage>::New();
        extract->SetInput(phase->GetOutput());
        extract->SetExtractionRegion(crop_region);
        extract->SetDirectionCollapseToIdentity();
        extract->Update();
        this->GetOutput()->Graft(extract->GetOutput());
      }
      break;
    case (LINEAR_BETWEEN_MINIMA):
      ComputeLinearPhaseBetweenPositions(m_MinimaPositions);
      break;
    case (LINEAR_BETWEEN_MAXIMA):
      ComputeLinearPhaseBetweenPositions(m_MaximaPositions);
      break;
  }
}

template <class TImage>
void
ExtractPhaseImageFilter<TImage>::ComputeLinearPhaseBetweenPositions(const PositionsListType & positions)
{
  if (positions.size() < 2)
    itkExceptionMacro(<< "Cannot compute linear phase with only one position");

  this->AllocateOutputs();

  using IteratorType = typename itk::ImageRegionIteratorWithIndex<TImage>;
  IteratorType               it(this->GetOutput(), this->GetOutput()->GetLargestPossibleRegion());
  typename TImage::PixelType phase = 0.;

  // Before the first position
  double slope = 1. / (positions[1] - positions[0]);
  for (int i = 0; i < positions[0]; i++, ++it)
  {
    phase = slope * (i - positions[0]);
    phase -= itk::Math::Floor<typename TImage::PixelType>(phase);
    it.Set(phase);
  }

  // Between positions
  for (unsigned int j = 1; j < positions.size(); j++)
  {
    slope = 1. / (positions[j] - positions[j - 1]);
    for (int i = positions[j - 1]; i < positions[j]; i++, ++it)
    {
      phase = slope * (i - positions[j - 1]);
      it.Set(phase);
    }
  }

  // After the last position
  slope = 1. / (positions.back() - positions[positions.size() - 2]);
  for (unsigned int i = positions.back(); i < this->GetOutput()->GetLargestPossibleRegion().GetSize(0); i++, ++it)
  {
    phase = slope * (i - positions[positions.size() - 2]);
    phase -= itk::Math::Floor<typename TImage::PixelType>(phase);
    it.Set(phase);
  }
}

} // end of namespace rtk
#endif
