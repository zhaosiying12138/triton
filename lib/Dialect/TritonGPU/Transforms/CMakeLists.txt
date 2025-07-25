add_triton_library(TritonGPUTransforms
  Coalesce.cpp
  CombineTensorSelectAndIf.cpp
  DecomposeScaledBlocked.cpp
  ReduceDataDuplication.cpp
  OptimizeThreadLocality.cpp
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  CoalesceAsyncCopy.cpp
  Utility.cpp
  DEPENDS
  TritonGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonTransforms
  TritonGPUIR
  TritonToTritonGPU
  MLIRTransformUtils
)
