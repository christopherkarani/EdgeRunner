import Metal
import EdgeRunnerSharedTypes

/// GPU-accelerated General Matrix Multiply (GEMM) kernel.
///
/// Computes C = A * B where A is MxK, B is KxN, and C is MxN.
/// Supports Float32 (`gemm_f32`) and Float16 (`gemm_f16`) precision.
struct GEMMKernel {
    private let pipelineF32: MTLComputePipelineState
    private let pipelineF16: MTLComputePipelineState

    /// Initializes the GEMM kernel by loading pipelines from the given registry.
    init(registry: KernelRegistry) throws {
        self.pipelineF32 = try registry.pipeline(for: "gemm_f32")
        self.pipelineF16 = try registry.pipeline(for: "gemm_f16")
    }

    /// The Float32 compute pipeline.
    var f32Pipeline: MTLComputePipelineState { pipelineF32 }

    /// The Float16 compute pipeline.
    var f16Pipeline: MTLComputePipelineState { pipelineF16 }
}
