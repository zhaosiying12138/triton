add_triton_library(TritonGPUToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandFMA.cpp
    DotOpToLLVM/FMA.cpp
    GlobalScratchMemoryAllocation.cpp
    TypeConverter.cpp
    Utility.cpp
    ElementwiseOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    AssertOpToLLVM.cpp
    ViewOpToLLVM.cpp
    MakeRangeOpToLLVM.cpp
    HistogramOpToLLVM.cpp
    AllocateSharedMemory.cpp
    ReduceOpToLLVM.cpp
    ScanOpToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    ControlFlowOpToLLVM.cpp
    FuncOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    DecomposeUnsupportedConversions.cpp
    PrintOpToLLVM.cpp
    CuteJitOrc.cpp

    DEPENDS
    TritonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGPUDialect
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
    LLVMExecutionEngine
    LLVMRuntimeDyld
    LLVMOrcJIT
    LLVMTransformUtils
    LLVMCore
    LLVMIRReader
    LLVMSupport
    LLVMTarget
    LLVMMC
    LLVMMCParser
    LLVMObject
    LLVMOption
    # ---- X86 目标相关 ----
    LLVMX86Info           # 提供 LLVMInitializeX86TargetInfo
    LLVMX86Desc           # 提供 LLVMInitializeX86TargetMC
    LLVMX86CodeGen        # 提供 LLVMInitializeX86Target
    # LLVMX86AsmPrinter     # 提供 LLVMInitializeX86AsmPrinter
    LLVMX86AsmParser      # 如果用到了 AsmParser
    LLVMX86Disassembler   # 如果需要反汇编支持
)

set(CUDA_INCLUDE_DIRS "$ENV{CUDA_INCLUDE_DIRS}")
set(CUTLASS_INCLUDE_DIRS "$ENV{CUTLASS_INCLUDE_DIRS}")

target_include_directories(TritonGPUToLLVM
  PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CUTLASS_INCLUDE_DIRS}
)
