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

#ifndef rtkCudaExternTemplates_h
#define rtkCudaExternTemplates_h

#include "rtkConfiguration.h"
#ifdef RTK_USE_CUDA

#  include <itkCudaImage.h>
#  include <itkImageSource.h>
#  include <itkImageToImageFilter.h>
#  include <itkInPlaceImageFilter.h>
#  include <itkCovariantVector.h>
#  include "RTKExport.h"

#  if defined(_MSC_VER)
#    if defined(RTK_EXPORTS)
#      define RTK_EXPORT_EXPLICIT
#    else
#      define RTK_EXPORT_EXPLICIT RTK_EXPORT
#    endif
#  else
#    define RTK_EXPORT_EXPLICIT
#  endif


// Suppress implicit instantiation of ITK base class templates specialized
// with CudaImage types. Without these declarations, MSVC produces LNK2005
// (duplicate symbol) errors when linking test executables against the RTK
// shared library, because the concrete CUDA filter classes (marked with
// RTK_EXPORT / __declspec(dllexport)) cause these base class methods to be
// exported from the DLL, conflicting with implicit instantiations in the
// consumer translation units.
//
// Explicit instantiation definitions are in rtkCudaExternTemplates.cxx.
ITK_GCC_PRAGMA_DIAG_PUSH()
ITK_GCC_PRAGMA_DIAG(ignored "-Wattributes")
extern template class RTK_EXPORT_EXPLICIT itk::ImageSource<itk::CudaImage<float, 3>>;
extern template class RTK_EXPORT_EXPLICIT itk::ImageToImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>;
extern template class RTK_EXPORT_EXPLICIT itk::InPlaceImageFilter<itk::CudaImage<float, 3>, itk::CudaImage<float, 3>>;
extern template class RTK_EXPORT_EXPLICIT itk::ImageSource<itk::CudaImage<itk::CovariantVector<float, 3>, 3>>;
ITK_GCC_PRAGMA_DIAG_POP()

namespace rtk
{
// Empty namespace block required by KWStyle
} // namespace rtk

#endif // RTK_USE_CUDA
#endif // rtkCudaExternTemplates_h
