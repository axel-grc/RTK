itk_wrap_module(RTK)

set(
  WRAPPER_SUBMODULE_ORDER
  itkFixedArrayRTK
  itkVectorRTK
  itkCovariantVectorRTK
  itkPointRTK
  itkContinuousIndexRTK
  itkMatrixRTK
  itkVariableLengthVectorRTK
  itkImageBaseRTK
  itkImageRTK
  itkCudaImageRTK
  itkVectorImageRTK
  itkImageDuplicatorRTK
  itkImageSourceRTK
  itkImageToImageFilterRTK
  itkInPlaceImageFilterRTK
  rtkFFTRampImageFilter
  rtkParkerShortScanImageFilter
  rtkScatterGlareCorrectionImageFilter
  rtkPolynomialGainCorrectionImageFilter
  rtkLagCorrectionImageFilter
  rtkTotalVariationDenoisingBPDQImageFilter
  itkCudaImageToImageFilterRTK
  itkCudaInPlaceImageFilterRTK
  rtkForwardProjectionImageFilter
  itkPyBufferRTK
)

itk_auto_load_submodules()
itk_end_wrap_module()

configure_file(
  ${RTK_SOURCE_DIR}/wrapping/rtkConfig.py.in
  ${RTK_BINARY_DIR}/Wrapping/Generators/Python/rtkConfig.py
)

file(
  GLOB RTK_GROUP_SCRIPTS
  CONFIGURE_DEPENDS
  "${RTK_SOURCE_DIR}/applications/rtk*_group.py"
)
file(
  GLOB RTK_APP_SCRIPTS
  CONFIGURE_DEPENDS
  "${RTK_SOURCE_DIR}/applications/rtk*/rtk*.py"
)

set(
  RTK_PYTHON_FILES
  "${RTK_SOURCE_DIR}/wrapping/__init_rtk__.py"
  ${RTK_GROUP_SCRIPTS}
  ${RTK_APP_SCRIPTS}
  "${RTK_BINARY_DIR}/Wrapping/Generators/Python/rtkConfig.py"
  "${RTK_SOURCE_DIR}/wrapping/rtkExtras.py"
  "${RTK_SOURCE_DIR}/applications/rtkargumentparser.py"
)

# Editable dev mode: install the Python files above as symbolic links back to
# the source tree instead of copies, so edits are picked up without
# reinstalling. Off by default: keep plain copies.
option(
  RTK_EDITABLE_PYTHON_PACKAGES
  "Install RTK Python scripts into the itk package as symbolic links (editable dev mode)."
  OFF
)

if(RTK_EDITABLE_PYTHON_PACKAGES)
  set(_rtk_editable_dir "${RTK_BINARY_DIR}/rtk_editable_python")
  file(REMOVE_RECURSE "${_rtk_editable_dir}")
  file(MAKE_DIRECTORY "${_rtk_editable_dir}")
  foreach(_rtk_src IN LISTS RTK_PYTHON_FILES)
    get_filename_component(_rtk_name "${_rtk_src}" NAME)
    file(
      CREATE_LINK
        "${_rtk_src}"
        "${_rtk_editable_dir}/${_rtk_name}"
      SYMBOLIC
      COPY_ON_ERROR
    )
  endforeach()
  install(
    DIRECTORY
      "${_rtk_editable_dir}/"
    DESTINATION "${PY_SITE_PACKAGES_PATH}/itk"
    COMPONENT RuntimeLibraries
  )
else()
  wrap_itk_python_bindings_install(/itk "RTK" ${RTK_PYTHON_FILES})
endif()
