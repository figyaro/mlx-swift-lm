//
//  BailingMoeV3.swift
//  mlx-swift-lm
//
//  Port of the Bailing MoE V3 hybrid KDA/MLA architecture used by
//  inclusionAI/Ling-3.0-tiny.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum BailingMoeV3AttentionKind: String, Sendable {
    case kda
    case mla
}

public enum BailingMoeV3ConfigurationError: Error, Equatable, LocalizedError {
    case invalidIdentity(modelType: String, architecture: String)
    case invalidValue(String)
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case .invalidIdentity(let modelType, let architecture):
            "Unsupported Bailing MoE V3 identity: model_type=\(modelType), architecture=\(architecture)"
        case .invalidValue(let message), .unsupported(let message):
            message
        }
    }
}

public struct BailingMoeV3Configuration: Codable, Sendable {
    private let architectures: [String]
    public var architecture: String { architectures.first ?? "" }
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let vocabularySize: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let tieWordEmbeddings: Bool
    public let layerGroupSize: Int
    public let shortConvKernelSize: Int
    public let noKDALora: Bool
    public let kdaSafeGate: Bool
    public let kdaLowerBound: Float
    public let gatedAttentionType: String
    public let qkHeadDim: Int
    public let qkNopeHeadDim: Int
    public let qkRopeHeadDim: Int
    public let vHeadDim: Int
    public let qLoraRank: Int
    public let kvLoraRank: Int
    public let ropeInterleave: Bool
    public let useQKNorm: Bool
    public let firstKDenseReplace: Int
    public let numExperts: Int
    public let numExpertsPerToken: Int
    public let numSharedExperts: Int
    public let moeIntermediateSize: Int
    public let moeSharedExpertIntermediateSize: Int
    public let nGroup: Int
    public let topkGroup: Int
    public let normTopkProb: Bool
    public let routedScalingFactor: Float
    public let moeRouterEnableExpertBias: Bool
    public let ropeScaling: [String: StringOrNumber]?

    public var projectionSize: Int { numAttentionHeads * headDim }

    enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerGroupSize = "layer_group_size"
        case shortConvKernelSize = "short_conv_kernel_size"
        case noKDALora = "no_kda_lora"
        case kdaSafeGate = "kda_safe_gate"
        case kdaLowerBound = "kda_lower_bound"
        case gatedAttentionType = "gated_attention_proj_granularity_type"
        case qkHeadDim = "qk_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case ropeInterleave = "rope_interleave"
        case useQKNorm = "use_qk_norm"
        case firstKDenseReplace = "first_k_dense_replace"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
        case ropeScaling = "rope_scaling"
    }

    public init(jsonData: Data) throws {
        let decoded = try JSONDecoder.json5().decode(Self.self, from: jsonData)
        try decoded.validateModelConfiguration()
        self = decoded
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let architectures = try container.decode([String].self, forKey: .architectures)
        guard let architecture = architectures.first else {
            throw BailingMoeV3ConfigurationError.invalidValue(
                "Bailing MoE V3 config must declare architectures"
            )
        }
        let modelType = try container.decode(String.self, forKey: .modelType)
        guard modelType == "bailing_hybrid", architecture == "BailingMoeV3ForCausalLM" else {
            throw BailingMoeV3ConfigurationError.invalidIdentity(
                modelType: modelType, architecture: architecture)
        }

        self.architectures = [architecture]
        self.modelType = modelType
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.tieWordEmbeddings = try container.decode(Bool.self, forKey: .tieWordEmbeddings)
        self.layerGroupSize = try container.decode(Int.self, forKey: .layerGroupSize)
        self.shortConvKernelSize = try container.decode(Int.self, forKey: .shortConvKernelSize)
        self.noKDALora = try container.decode(Bool.self, forKey: .noKDALora)
        self.kdaSafeGate = try container.decode(Bool.self, forKey: .kdaSafeGate)
        self.kdaLowerBound = try container.decode(Float.self, forKey: .kdaLowerBound)
        self.gatedAttentionType = try container.decode(String.self, forKey: .gatedAttentionType)
        self.qkHeadDim = try container.decode(Int.self, forKey: .qkHeadDim)
        self.qkNopeHeadDim = try container.decode(Int.self, forKey: .qkNopeHeadDim)
        self.qkRopeHeadDim = try container.decode(Int.self, forKey: .qkRopeHeadDim)
        self.vHeadDim = try container.decode(Int.self, forKey: .vHeadDim)
        self.qLoraRank = try container.decode(Int.self, forKey: .qLoraRank)
        self.kvLoraRank = try container.decode(Int.self, forKey: .kvLoraRank)
        self.ropeInterleave = try container.decode(Bool.self, forKey: .ropeInterleave)
        self.useQKNorm = try container.decode(Bool.self, forKey: .useQKNorm)
        self.firstKDenseReplace = try container.decode(Int.self, forKey: .firstKDenseReplace)
        self.numExperts = try container.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerToken = try container.decode(Int.self, forKey: .numExpertsPerToken)
        self.numSharedExperts = try container.decode(Int.self, forKey: .numSharedExperts)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.moeSharedExpertIntermediateSize = try container.decode(
            Int.self, forKey: .moeSharedExpertIntermediateSize)
        self.nGroup = try container.decode(Int.self, forKey: .nGroup)
        self.topkGroup = try container.decode(Int.self, forKey: .topkGroup)
        self.normTopkProb = try container.decode(Bool.self, forKey: .normTopkProb)
        self.routedScalingFactor = try container.decode(Float.self, forKey: .routedScalingFactor)
        self.moeRouterEnableExpertBias = try container.decode(
            Bool.self, forKey: .moeRouterEnableExpertBias)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
    }

    public func validateModelConfiguration() throws {
        guard modelType == "bailing_hybrid", architecture == "BailingMoeV3ForCausalLM" else {
            throw BailingMoeV3ConfigurationError.invalidIdentity(
                modelType: modelType, architecture: architecture)
        }
        guard hiddenSize > 0, headDim > 0, numAttentionHeads > 0 else {
            throw BailingMoeV3ConfigurationError.invalidValue(
                "hidden_size, head_dim, and num_attention_heads must be positive")
        }
        guard qkNopeHeadDim + qkRopeHeadDim == qkHeadDim else {
            throw BailingMoeV3ConfigurationError.invalidValue(
                "qk_nope_head_dim + qk_rope_head_dim must equal qk_head_dim")
        }
        guard layerGroupSize > 0, numHiddenLayers > 0 else {
            throw BailingMoeV3ConfigurationError.invalidValue(
                "layer_group_size and num_hidden_layers must be positive")
        }
        guard numExperts > 0, numExperts % nGroup == 0 else {
            throw BailingMoeV3ConfigurationError.invalidValue(
                "num_experts must be positive and divisible by n_group")
        }
        guard topkGroup > 0, topkGroup <= nGroup else {
            throw BailingMoeV3ConfigurationError.invalidValue(
                "topk_group must be within the available expert groups")
        }
        guard numExpertsPerToken > 0,
            numExpertsPerToken <= topkGroup * (numExperts / nGroup)
        else {
            throw BailingMoeV3ConfigurationError.invalidValue(
                "num_experts_per_tok exceeds the selected expert groups")
        }
        guard noKDALora else {
            throw BailingMoeV3ConfigurationError.unsupported(
                "Bailing MoE V3 KDA LoRA projections are not supported")
        }
        guard gatedAttentionType == "head_wise" else {
            throw BailingMoeV3ConfigurationError.unsupported(
                "Only head_wise gated attention is supported")
        }
        guard ropeInterleave else {
            throw BailingMoeV3ConfigurationError.unsupported(
                "Only interleaved MLA RoPE is supported")
        }
        guard ropeScaling == nil else {
            throw BailingMoeV3ConfigurationError.unsupported(
                "Ling-3.0-tiny rope scaling is not supported")
        }
    }

    public func attentionKind(forLayer layer: Int) -> BailingMoeV3AttentionKind {
        let finalCompleteGroup = (numHiddenLayers / layerGroupSize) * layerGroupSize
        if (layer + 1) % layerGroupSize == 0 || layer >= finalCompleteGroup {
            return .mla
        }
        return .kda
    }
}

extension BailingMoeV3Configuration: ModelConfigurationValidating {}

private func bailingMoeV3L2Normalize(_ x: MLXArray) -> MLXArray {
    let x = x.asType(.float32)
    let meanSquares = (x * x).sum(axis: -1, keepDims: true)
    return x * (meanSquares + 1e-6).rsqrt()
}

/// Decode-only compression for the wide dense projections around the routed
/// experts. Ollama's latest Bailing path quantizes float passthrough tensors
/// on request because lm_head and attention projections otherwise dominate
/// single-token memory traffic. Keep the checkpoint and prefill path in its
/// native dtype; only B=L=1 calls use the cached MXFP8 representation.
private enum BailingMoeV3DecodeQuantization {
    static let groupSize = 32
    static let bits = 8
    static let mode: QuantizationMode = .mxfp8
    static let enabled = ProcessInfo.processInfo.environment["DENOJU_LING_DENSE_MXFP8"] != "0"

    static func supports(_ weight: MLXArray) -> Bool {
        guard enabled, weight.ndim == 2,
            weight.dim(-1).isMultiple(of: groupSize),
            weight.dim(0) >= 512, weight.dim(-1) >= 512
        else {
            return false
        }
        return weight.dtype == .bfloat16 || weight.dtype == .float16 || weight.dtype == .float32
    }
}

private final class BailingMoeV3DecodeLinear: Linear {
    private var decodeQuantizedWeight:
        (weight: MLXArray, scales: MLXArray, biases: MLXArray?)?
    private let decodeQuantizedWeightLock = NSLock()

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard bias == nil, x.ndim >= 2, x.dim(0) == 1, x.dim(1) == 1,
            BailingMoeV3DecodeQuantization.supports(weight)
        else {
            return super.callAsFunction(x)
        }

        decodeQuantizedWeightLock.lock()
        if decodeQuantizedWeight == nil {
            let quantized = MLX.quantized(
                weight,
                groupSize: BailingMoeV3DecodeQuantization.groupSize,
                bits: BailingMoeV3DecodeQuantization.bits,
                mode: BailingMoeV3DecodeQuantization.mode)
            var arrays = [quantized.0, quantized.1]
            if let biases = quantized.2 {
                arrays.append(biases)
            }
            // Publish only evaluated immutable tensors. MLXArray lazy graphs
            // must not be initialized concurrently by separate chat sessions.
            eval(arrays)
            decodeQuantizedWeight = (
                weight: quantized.0, scales: quantized.1, biases: quantized.2)
        }
        let quantized = decodeQuantizedWeight!
        decodeQuantizedWeightLock.unlock()

        return MLX.quantizedMM(
            x, quantized.weight, scales: quantized.scales, biases: quantized.biases,
            transpose: true,
            groupSize: BailingMoeV3DecodeQuantization.groupSize,
            bits: BailingMoeV3DecodeQuantization.bits,
            mode: BailingMoeV3DecodeQuantization.mode)
    }
}

/// Build the single-token KDA graph used during autoregressive decode.
///
/// Prefill still uses the dedicated SIMD recurrence kernel below.  Decode is a
/// different workload: every layer receives exactly one token, so the cost is
/// dominated by launching the normalization, decay gate, delta update, and
/// reductions as separate MLX operations.  MLX's compiler can fuse this fixed
/// graph while retaining the same FP32 recurrent state as the reference path.
private func makeBailingMoeV3KDADecodeStep(
    safeGate: Bool
) -> @Sendable ([MLXArray]) -> [MLXArray] {
    compile(shapeless: true) { inputs in
        let q = inputs[0]
        let k = inputs[1]
        let v = inputs[2].asType(.float32)
        let decayInput = inputs[3].asType(.float32)
        let betaLogits = inputs[4].asType(.float32)
        var state = inputs[5]
        let aExp = inputs[6].reshaped(1, q.dim(1), 1)
        let dt = inputs[7].reshaped(1, q.dim(1), q.dim(2))
        let qScale = inputs[8]
        let lowerBound = inputs[9]

        let normalizedQ = bailingMoeV3L2Normalize(q) * qScale
        let normalizedK = bailingMoeV3L2Normalize(k)
        let beta = MLXNN.sigmoid(betaLogits)
        let gateInput = decayInput + dt
        let logDecay: MLXArray
        if safeGate {
            logDecay = MLXNN.sigmoid(aExp * gateInput) * lowerBound
        } else {
            logDecay = -(aExp * MLXNN.softplus(gateInput))
        }
        let decay = exp(logDecay)

        let keyExpanded = expandedDimensions(normalizedK, axis: -2)
        state = state * expandedDimensions(decay, axis: -2)
        let memory = (state * keyExpanded).sum(axis: -1)
        let delta = (v - memory) * expandedDimensions(beta, axis: -1)
        state = state + keyExpanded * expandedDimensions(delta, axis: -1)
        let output = (state * expandedDimensions(normalizedQ, axis: -2)).sum(axis: -1)
        return [output.asType(q.dtype), state]
    }
}

private let bailingMoeV3KDADecodeStepSafe = makeBailingMoeV3KDADecodeStep(safeGate: true)
private let bailingMoeV3KDADecodeStepSoftplus = makeBailingMoeV3KDADecodeStep(safeGate: false)

/// Compile the head-wise output gate used after the recurrent KDA state.
///
/// This is deliberately kept separate from the recurrence graph: the output
/// gate consumes the normalized state and is also used by the prefill path.
/// Keeping it as a tiny compiled graph removes the per-token sigmoid,
/// broadcast, multiply, and cast launch sequence without changing the FP32
/// accumulation semantics of the recurrent state.
private let bailingMoeV3KDAOutputGate: @Sendable (MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { output, gate in
        (output.asType(.float32) * MLXNN.sigmoid(gate.asType(.float32))).asType(output.dtype)
    }

private func bailingMoeV3InterleavedToHalf(_ x: MLXArray) -> MLXArray {
    let dimension = x.dim(-1)
    precondition(dimension.isMultiple(of: 2), "Rotary dimensions must be even")

    let shape = x.shape
    let paired = x.reshaped(Array(shape.dropLast()) + [dimension / 2, 2])
    let axes = Array(0 ..< paired.ndim - 2) + [paired.ndim - 1, paired.ndim - 2]
    return paired.transposed(axes: axes).reshaped(shape)
}

// MARK: - Ling KDA decode kernel

/// Fuses a single KDA recurrence step into one Metal dispatch.
///
/// Ling-3.0-tiny has eighteen KDA layers.  During token-by-token decoding the
/// previous implementation expressed each one as a Swift loop over many small
/// MLX operations (broadcasts, reductions, and state writes).  The operations
/// are mathematically simple, but their individual kernel launches dominate
/// decode time on Apple silicon.  This kernel keeps the state in FP32 exactly
/// as the reference path does, and only changes how the one-token recurrence is
/// scheduled.
private func makeBailingMoeV3KDAUpdateKernel() -> MLXFast.MLXFastKernel {
    let source = """
        uint index = thread_position_in_grid.x;
        if (index >= B * H * D) {
            return;
        }

        uint valueIndex = index % D;
        uint headIndex = (index / D) % H;
        uint batchIndex = index / (H * D);
        uint vectorOffset = (batchIndex * H + headIndex) * D;
        uint stateOffset = index * D;

        // state is [batch, head, value, key].  Decay applies to its key axis.
        float memory = 0.0f;
        for (uint key = 0; key < D; ++key) {
            memory += state[stateOffset + key] * k[vectorOffset + key];
        }
        float delta = (v[vectorOffset + valueIndex] - memory)
            * beta[batchIndex * H + headIndex];

        float result = 0.0f;
        for (uint key = 0; key < D; ++key) {
            float updated = state[stateOffset + key] * decay[vectorOffset + key]
                + k[vectorOffset + key] * delta;
            next_state[stateOffset + key] = updated;
            result += updated * q[vectorOffset + key];
        }
        output[index] = result;
        """

    return MLXFast.metalKernel(
        name: "bailing_moe_v3_kda_decode",
        inputNames: ["q", "k", "v", "decay", "beta", "state"],
        outputNames: ["output", "next_state"],
        source: source
    )
}

private final class BailingMoeV3KDAKernelManager: Sendable {
    static let shared = BailingMoeV3KDAKernelManager()

    let updateKernel: MLXFast.MLXFastKernel

    private init() {
        updateKernel = makeBailingMoeV3KDAUpdateKernel()
    }
}

private func bailingMoeV3KDAUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    decay: MLXArray,
    beta: MLXArray,
    state: MLXArray
) -> (output: MLXArray, state: MLXArray) {
    let (batch, heads, dimension) = q.shape3
    let outputs = BailingMoeV3KDAKernelManager.shared.updateKernel(
        [q, k, v, decay, beta, state],
        template: [
            ("B", batch),
            ("H", heads),
            ("D", dimension),
        ],
        grid: (batch * heads * dimension, 1, 1),
        threadGroup: (min(dimension, 256), 1, 1),
        outputShapes: [[batch, heads, dimension], state.shape],
        outputDTypes: [.float32, .float32]
    )
    return (outputs[0], outputs[1])
}

private class BailingMoeV3DenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ configuration: BailingMoeV3Configuration, intermediateSize: Int? = nil) {
        let intermediateSize = intermediateSize ?? configuration.intermediateSize
        _gateUpProj.wrappedValue = BailingMoeV3DecodeLinear(
            configuration.hiddenSize, 2 * intermediateSize, bias: false)
        _downProj.wrappedValue = BailingMoeV3DecodeLinear(
            intermediateSize, configuration.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateUp = split(gateUpProj(x), parts: 2, axis: -1)
        return downProj(compiledSiluProduct(gateUp[0], gateUp[1]))
    }
}

private func bailingMoeV3RouteTopK(
    scores: MLXArray,
    routingScores: MLXArray,
    configuration: BailingMoeV3Configuration
) -> (indices: MLXArray, weights: MLXArray) {
    let expertsPerGroup = configuration.numExperts / configuration.nGroup
    let grouped = routingScores.reshaped(
        1, 1, configuration.nGroup, expertsPerGroup)

    // Keep the decode router in the compiled graph without Slice primitives:
    // MLX 0.29/0.31 cannot infer Slice output shapes for symbolic sources.
    let top2Positions = repeated(
        MLXArray([0, 1] as [Int32]).reshaped(1, 1, 1, 2),
        count: configuration.nGroup,
        axis: 2
    )
    let groupTop2 = takeAlong(
        argPartition(-grouped, kth: 1, axis: -1), top2Positions, axis: -1)
    let groupScores = takeAlong(grouped, groupTop2, axis: -1).sum(axis: -1)
    let groupPositions = MLXArray(
        Array(0 ..< configuration.topkGroup).map(Int32.init)
    ).reshaped(1, 1, configuration.topkGroup)
    let groupIndices = takeAlong(
        argPartition(-groupScores, kth: configuration.topkGroup - 1, axis: -1),
        groupPositions,
        axis: -1
    )

    let groupGather = repeated(
        expandedDimensions(groupIndices, axis: -1), count: expertsPerGroup, axis: -1)
    let candidateScores = takeAlong(grouped, groupGather, axis: 2).flattened(
        start: -2, end: -1)
    let localIDs = MLXArray(0 ..< expertsPerGroup).reshaped(1, 1, 1, expertsPerGroup)
    let globalIDs = (
        groupGather * expertsPerGroup + localIDs
    ).flattened(start: -2, end: -1)
    let candidatePositions = MLXArray(
        Array(0 ..< configuration.numExpertsPerToken).map(Int32.init)
    ).reshaped(1, 1, configuration.numExpertsPerToken)
    let selectedCandidates = takeAlong(
        argPartition(-candidateScores, kth: configuration.numExpertsPerToken - 1, axis: -1),
        candidatePositions,
        axis: -1
    )
    let indices = takeAlong(globalIDs, selectedCandidates, axis: -1)
    var weights = takeAlong(scores, indices, axis: -1)
    if configuration.normTopkProb, configuration.numExpertsPerToken > 1 {
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
    }
    return (indices, weights * configuration.routedScalingFactor)
}

private func makeBailingMoeV3DecodeRouter(
    _ configuration: BailingMoeV3Configuration
) -> @Sendable ([MLXArray]) -> [MLXArray] {
    compile(shapeless: true) { inputs in
        let scores = MLXNN.sigmoid(inputs[0].asType(.float32))
        let routingScores: MLXArray
        if configuration.moeRouterEnableExpertBias {
            routingScores = scores + inputs[1].asType(.float32).reshaped(
                1, 1, configuration.numExperts)
        } else {
            routingScores = scores
        }
        let route = bailingMoeV3RouteTopK(
            scores: scores,
            routingScores: routingScores,
            configuration: configuration
        )
        return [route.indices, route.weights]
    }
}

private class BailingMoeV3Gate: Module {
    let configuration: BailingMoeV3Configuration
    private let fusedDecodeRouter: @Sendable ([MLXArray]) -> [MLXArray]

    // The outer sparse-MoE module owns `mlp.gate`; the projection itself is
    // nested below it and receives the converted Hugging Face weight name.
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray

    init(_ configuration: BailingMoeV3Configuration) {
        self.configuration = configuration
        self.fusedDecodeRouter = makeBailingMoeV3DecodeRouter(configuration)
        _gateProj.wrappedValue = BailingMoeV3DecodeLinear(
            configuration.hiddenSize, configuration.numExperts, bias: false)
        _expertBias.wrappedValue = MLXArray.zeros([configuration.numExperts])
    }

    func select(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let logits = gateProj(x)

        if x.dim(0) == 1, x.dim(1) == 1,
            configuration.headDim.isMultiple(of: 32) {
            let inputs = configuration.moeRouterEnableExpertBias
                ? [logits, expertBias]
                : [logits]
            let outputs = fusedDecodeRouter(inputs)
            return (indices: outputs[0], weights: outputs[1])
        }

        let expertsPerGroup = configuration.numExperts / configuration.nGroup
        let scores = MLXNN.sigmoid(logits.asType(.float32))
        let routingScores = configuration.moeRouterEnableExpertBias
            ? scores + expertBias.asType(.float32)
            : scores
        let grouped = routingScores.reshaped(
            x.dim(0), x.dim(1), configuration.nGroup, expertsPerGroup)

        let groupTop2 = argPartition(-grouped, kth: 1, axis: -1)[.ellipsis, ..<2]
        let groupScores = takeAlong(grouped, groupTop2, axis: -1).sum(axis: -1)
        let groupIndices = argPartition(
            -groupScores, kth: configuration.topkGroup - 1, axis: -1
        )[
            .ellipsis, ..<configuration.topkGroup
        ]

        let groupGather = repeated(
            expandedDimensions(groupIndices, axis: -1), count: expertsPerGroup, axis: -1)
        let candidateScores = takeAlong(grouped, groupGather, axis: 2).flattened(
            start: -2, end: -1)
        let localIDs = MLXArray(0 ..< expertsPerGroup).reshaped(1, 1, 1, expertsPerGroup)
        let globalIDs = (
            groupGather * expertsPerGroup + localIDs
        ).flattened(start: -2, end: -1)
        let selectedCandidates = argPartition(
            -candidateScores, kth: configuration.numExpertsPerToken - 1, axis: -1
        )[
            .ellipsis, ..<configuration.numExpertsPerToken
        ]
        let indices = takeAlong(globalIDs, selectedCandidates, axis: -1)
        var weights = takeAlong(scores, indices, axis: -1)
        if configuration.normTopkProb, configuration.numExpertsPerToken > 1 {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, weights * configuration.routedScalingFactor)
    }
}

private class BailingMoeV3SparseMoeBlock: Module, UnaryLayer {
    let configuration: BailingMoeV3Configuration

    @ModuleInfo(key: "switch_mlp") var switchMLP: FusedGateUpSwitchGLU
    @ModuleInfo(key: "gate") var gate: BailingMoeV3Gate
    @ModuleInfo(key: "shared_experts") var sharedExperts: BailingMoeV3DenseMLP?

    init(_ configuration: BailingMoeV3Configuration) {
        self.configuration = configuration
        _switchMLP.wrappedValue = FusedGateUpSwitchGLU(
            inputDims: configuration.hiddenSize,
            hiddenDims: configuration.moeIntermediateSize,
            numExperts: configuration.numExperts,
            bias: false,
            // Ling's BF16 checkpoint is imported as-is, but single-token
            // expert decode benefits from the same MXFP8 path used by the
            // native MLX runner. Prefill keeps the original BF16 projections.
            decodeQuantization: SwitchDecodeQuantization(
                groupSize: 32, bits: 8, mode: .mxfp8)
        )
        _gate.wrappedValue = BailingMoeV3Gate(configuration)
        if configuration.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = BailingMoeV3DenseMLP(
                configuration,
                intermediateSize: configuration.moeSharedExpertIntermediateSize
                    * configuration.numSharedExperts
            )
        } else {
            _sharedExperts.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = gate.select(x)
        var output = weightedExpertSum(switchMLP(x, indices), weights)
        if let sharedExperts {
            output = output + sharedExperts(x)
        }
        return output
    }
}

private protocol BailingMoeV3Attention: Module {
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray
}

private class BailingMoeV3KDAAttention: Module, BailingMoeV3Attention {
    let configuration: BailingMoeV3Configuration

    // Ling's decode path is dominated by six small input projections.  Keep
    // the checkpoint-compatible projections as one QKV and one FBG linear so
    // a single-token decode uses two matmuls instead of six.
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "fbg_proj") var fbgProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_conv1d") var qConv: Conv1d
    @ModuleInfo(key: "k_conv1d") var kConv: Conv1d
    @ModuleInfo(key: "v_conv1d") var vConv: Conv1d
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

    init(_ configuration: BailingMoeV3Configuration) {
        self.configuration = configuration
        let projectionSize = configuration.projectionSize
        _qkvProj.wrappedValue = BailingMoeV3DecodeLinear(
            configuration.hiddenSize, 3 * projectionSize, bias: false)
        _fbgProj.wrappedValue = BailingMoeV3DecodeLinear(
            configuration.hiddenSize,
            2 * projectionSize + configuration.numAttentionHeads,
            bias: false
        )
        _oProj.wrappedValue = BailingMoeV3DecodeLinear(
            projectionSize, configuration.hiddenSize, bias: false)
        _qConv.wrappedValue = Conv1d(
            inputChannels: projectionSize, outputChannels: projectionSize,
            kernelSize: configuration.shortConvKernelSize, groups: projectionSize, bias: false)
        _kConv.wrappedValue = Conv1d(
            inputChannels: projectionSize, outputChannels: projectionSize,
            kernelSize: configuration.shortConvKernelSize, groups: projectionSize, bias: false)
        _vConv.wrappedValue = Conv1d(
            inputChannels: projectionSize, outputChannels: projectionSize,
            kernelSize: configuration.shortConvKernelSize, groups: projectionSize, bias: false)
        _oNorm.wrappedValue = RMSNorm(dimensions: configuration.headDim, eps: configuration.rmsNormEps)
        _aLog.wrappedValue = MLXArray.zeros([configuration.numAttentionHeads])
        _dtBias.wrappedValue = MLXArray.zeros([projectionSize])
    }

    func callAsFunction(
        _ x: MLXArray, mask _: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let cache = cache as? MambaCache
        let (batch, length) = (x.dim(0), x.dim(1))
        let projectionSize = configuration.projectionSize
        let kernelTail = configuration.shortConvKernelSize - 1
        let projected = qkvProj(x)
        let convState = cache?[0] ?? MLXArray.zeros(
            [batch, kernelTail, 3 * projectionSize], dtype: x.dtype)
        let paddedInput = concatenated([convState, projected], axis: 1)
        let projectedParts = split(paddedInput, indices: [projectionSize, 2 * projectionSize], axis: -1)
        let q = MLXNN.silu(qConv(projectedParts[0])).reshaped(
            batch, length, configuration.numAttentionHeads, configuration.headDim)
        let k = MLXNN.silu(kConv(projectedParts[1])).reshaped(
            batch, length, configuration.numAttentionHeads, configuration.headDim)
        let v = MLXNN.silu(vConv(projectedParts[2])).reshaped(
            batch, length, configuration.numAttentionHeads, configuration.headDim)

        let fbg = fbgProj(x)
        let fbgParts = split(
            fbg,
            indices: [projectionSize, projectionSize + configuration.numAttentionHeads],
            axis: -1
        )
        let decayInput = fbgParts[0].reshaped(
            batch, length, configuration.numAttentionHeads, configuration.headDim).asType(.float32)
        let aExp = exp(aLog.asType(.float32)).reshaped(1, 1, configuration.numAttentionHeads, 1)
        let dt = dtBias.asType(.float32).reshaped(
            1, 1, configuration.numAttentionHeads, configuration.headDim)

        // KDA keeps a [value, key] state matrix. Ling's feature-wise decay
        // applies to the key axis, never to the value axis. Use the common
        // SIMD Metal recurrence for both prefill and decode; it preserves the
        // FP32 state while avoiding a Swift loop and per-token MLX launches.
        let state = cache?[1] ?? MLXArray.zeros(
            [batch, configuration.numAttentionHeads, configuration.headDim, configuration.headDim],
            dtype: .float32
        )
        let recurrence: (output: MLXArray, state: MLXArray)
        if length == 1, configuration.headDim.isMultiple(of: 32) {
            let decodeStep = configuration.kdaSafeGate
                ? bailingMoeV3KDADecodeStepSafe
                : bailingMoeV3KDADecodeStepSoftplus
            let outputs = decodeStep([
                q.squeezed(axis: 1),
                k.squeezed(axis: 1),
                v.squeezed(axis: 1),
                decayInput.squeezed(axis: 1),
                fbgParts[1].reshaped(batch, length, configuration.numAttentionHeads)
                    .squeezed(axis: 1),
                state,
                aExp.reshaped(configuration.numAttentionHeads),
                dt.reshaped(configuration.numAttentionHeads, configuration.headDim),
                MLXArray(Float(1.0 / sqrt(Double(configuration.headDim)))),
                MLXArray(configuration.kdaLowerBound),
            ])
            recurrence = (
                output: outputs[0].expandedDimensions(axis: 1),
                state: outputs[1]
            )
        } else {
            let normalizedQ = bailingMoeV3L2Normalize(q)
                * Float(1.0 / sqrt(Double(configuration.headDim)))
            let normalizedK = bailingMoeV3L2Normalize(k)
            let beta = MLXNN.sigmoid(
                fbgParts[1].reshaped(batch, length, configuration.numAttentionHeads)
                    .asType(.float32))
            let logDecay: MLXArray
            if configuration.kdaSafeGate {
                logDecay = MLXNN.sigmoid(aExp * (decayInput + dt)) * configuration.kdaLowerBound
            } else {
                logDecay = -(aExp * MLXNN.softplus(decayInput + dt))
            }
            let decay = exp(logDecay)
            if configuration.headDim.isMultiple(of: 32) {
                recurrence = gatedDeltaFeatureDecay(
                    q: normalizedQ,
                    k: normalizedK,
                    v: v,
                    decay: decay,
                    beta: beta,
                    state: state
                )
            } else {
                // Keep the reference path for small synthetic/test configurations
                // that cannot satisfy the SIMD kernel's 32-wide lane contract.
                var fallbackState = state
                var outputs: [MLXArray] = []
                outputs.reserveCapacity(length)
                for token in 0 ..< length {
                    let qToken = normalizedQ[0..., token, 0..., 0...]
                    let kToken = normalizedK[0..., token, 0..., 0...]
                    let vToken = v[0..., token, 0..., 0...].asType(.float32)
                    let decayToken = decay[0..., token, 0..., 0...]
                    let betaToken = beta[0..., token, 0...]
                    let keyExpanded = expandedDimensions(kToken, axis: -2)
                    fallbackState = fallbackState * expandedDimensions(decayToken, axis: -2)
                    let memory = (fallbackState * keyExpanded).sum(axis: -1)
                    let delta = (vToken - memory) * expandedDimensions(betaToken, axis: -1)
                    fallbackState = fallbackState + keyExpanded * expandedDimensions(delta, axis: -1)
                    let output = (fallbackState * expandedDimensions(qToken, axis: -2)).sum(axis: -1)
                    outputs.append(expandedDimensions(output.asType(x.dtype), axis: 1))
                }
                recurrence = (output: concatenated(outputs, axis: 1), state: fallbackState)
            }
        }

        if let cache {
            let start = max(0, paddedInput.dim(1) - kernelTail)
            cache[0] = contiguous(paddedInput[0..., start..., 0...])
            cache[1] = recurrence.state
            cache.advance(length)
        }

        var output = recurrence.output
        output = oNorm(output)
        if length == 1, configuration.headDim.isMultiple(of: 32) {
            output = bailingMoeV3KDAOutputGate(
                output,
                fbgParts[2].reshaped(
                    batch, length, configuration.numAttentionHeads, configuration.headDim
                )
            )
        } else {
            let gate = MLXNN.sigmoid(fbgParts[2].reshaped(
                batch, length, configuration.numAttentionHeads, configuration.headDim).asType(.float32))
            output = (output.asType(.float32) * gate).asType(x.dtype)
        }
        return oProj(output.reshaped(batch, length, projectionSize))
    }
}

private class BailingMoeV3MLAAttention: Module, BailingMoeV3Attention {
    let configuration: BailingMoeV3Configuration
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear
    @ModuleInfo(key: "dense") var dense: Linear

    init(_ configuration: BailingMoeV3Configuration) {
        self.configuration = configuration
        self.scale = pow(Float(configuration.qkHeadDim), -0.5)
        self.rope = initializeRope(
            dims: configuration.qkRopeHeadDim, base: configuration.ropeTheta,
            traditional: false, scalingConfig: nil,
            maxPositionEmbeddings: configuration.maxPositionEmbeddings)

        if configuration.qLoraRank > 0 {
            _qProj.wrappedValue = nil
            _qAProj.wrappedValue = BailingMoeV3DecodeLinear(
                configuration.hiddenSize, configuration.qLoraRank, bias: false)
            _qALayerNorm.wrappedValue = RMSNorm(
                dimensions: configuration.qLoraRank, eps: configuration.rmsNormEps)
            _qBProj.wrappedValue = BailingMoeV3DecodeLinear(
                configuration.qLoraRank,
                configuration.numAttentionHeads * configuration.qkHeadDim,
                bias: false
            )
        } else {
            _qProj.wrappedValue = BailingMoeV3DecodeLinear(
                configuration.hiddenSize,
                configuration.numAttentionHeads * configuration.qkHeadDim,
                bias: false
            )
            _qAProj.wrappedValue = nil
            _qALayerNorm.wrappedValue = nil
            _qBProj.wrappedValue = nil
        }
        _kvAProjWithMqa.wrappedValue = BailingMoeV3DecodeLinear(
            configuration.hiddenSize, configuration.kvLoraRank + configuration.qkRopeHeadDim,
            bias: false)
        _kvALayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.kvLoraRank, eps: configuration.rmsNormEps)
        _kvBProj.wrappedValue = BailingMoeV3DecodeLinear(
            configuration.kvLoraRank,
            configuration.numAttentionHeads * (configuration.qkNopeHeadDim + configuration.vHeadDim),
            bias: false)
        _gProj.wrappedValue = BailingMoeV3DecodeLinear(
            configuration.hiddenSize, configuration.numAttentionHeads, bias: false)
        _dense.wrappedValue = BailingMoeV3DecodeLinear(
            configuration.numAttentionHeads * configuration.vHeadDim, configuration.hiddenSize,
            bias: false)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let q: MLXArray
        if let qProj {
            q = qProj(x)
        } else {
            q = qBProj!(qALayerNorm!(qAProj!(x)))
        }
        let qParts = split(
            q.reshaped(batch, length, configuration.numAttentionHeads, configuration.qkHeadDim)
                .transposed(0, 2, 1, 3),
            indices: [configuration.qkNopeHeadDim], axis: -1)
        let qNope = qParts[0]
        var qPE = qParts[1]

        let compressed = kvAProjWithMqa(x)
        let compressedParts = split(compressed, indices: [configuration.kvLoraRank], axis: -1)
        let kvLatent = compressedParts[0]
        var kPE = compressedParts[1].reshaped(
            batch, length, 1, configuration.qkRopeHeadDim).transposed(0, 2, 1, 3)
        let kvExpanded = kvBProj(kvALayerNorm(kvLatent)).reshaped(
            batch, length, configuration.numAttentionHeads,
            configuration.qkNopeHeadDim + configuration.vHeadDim
        ).transposed(0, 2, 1, 3)
        let kvParts = split(kvExpanded, indices: [configuration.qkNopeHeadDim], axis: -1)
        let kNope = kvParts[0]
        let values = kvParts[1]

        let offset = cache?.ropeOffset
        qPE = applyRotaryPosition(rope, to: bailingMoeV3InterleavedToHalf(qPE), offset: offset)
        kPE = applyRotaryPosition(rope, to: bailingMoeV3InterleavedToHalf(kPE), offset: offset)
        kPE = repeated(kPE, count: configuration.numAttentionHeads, axis: 1)

        let queries = concatenated([qNope, qPE], axis: -1)
        let keys = concatenated([kNope, kPE], axis: -1)
        var output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask)
        let gate = MLXNN.sigmoid(gProj(x).asType(.float32)).transposed(0, 2, 1)
            .expandedDimensions(axis: -1)
        output = (output.asType(.float32) * gate).asType(x.dtype)
        return dense(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

private class BailingMoeV3TransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: Module & BailingMoeV3Attention
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ configuration: BailingMoeV3Configuration, layerIndex: Int) {
        if configuration.attentionKind(forLayer: layerIndex) == .kda {
            _attention.wrappedValue = BailingMoeV3KDAAttention(configuration)
        } else {
            _attention.wrappedValue = BailingMoeV3MLAAttention(configuration)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        if layerIndex < configuration.firstKDenseReplace {
            _mlp.wrappedValue = BailingMoeV3DenseMLP(configuration)
        } else {
            _mlp.wrappedValue = BailingMoeV3SparseMoeBlock(configuration)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let hidden = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return hidden + mlp(postAttentionLayerNorm(hidden))
    }
}

private class BailingMoeV3ModelInner: Module {
    let configuration: BailingMoeV3Configuration
    let firstMLALayerIndex: Int

    @ModuleInfo(key: "word_embeddings") var embedTokens: Embedding
    fileprivate let layers: [BailingMoeV3TransformerBlock]
    let norm: RMSNorm

    init(_ configuration: BailingMoeV3Configuration) {
        self.configuration = configuration
        self.firstMLALayerIndex = (0 ..< configuration.numHiddenLayers).first {
            configuration.attentionKind(forLayer: $0) == .mla
        } ?? 0
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize, dimensions: configuration.hiddenSize)
        self.layers = (0 ..< configuration.numHiddenLayers).map {
            BailingMoeV3TransformerBlock(configuration, layerIndex: $0)
        }
        self.norm = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embedTokens(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache?[firstMLALayerIndex])
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[index])
        }
        return norm(hidden)
    }
}

public class BailingMoeV3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: BailingMoeV3Configuration

    @ModuleInfo(key: "model") fileprivate var model: BailingMoeV3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: BailingMoeV3Configuration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(repeating: configuration.numAttentionHeads, count: configuration.numHiddenLayers)
        _model.wrappedValue = BailingMoeV3ModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = BailingMoeV3DecodeLinear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.numHiddenLayers).map { layer in
            configuration.attentionKind(forLayer: layer) == .kda ? MambaCache() : KVCacheSimple()
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let output = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(output)
        }
        return model.embedTokens.asLinear(output)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights
        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        for layer in 0 ..< configuration.numHiddenLayers {
            let prefix = "model.layers.\(layer)"

            // The upstream checkpoint stores these projections separately.  The
            // runtime modules intentionally use row-fused weights so decode can
            // replace several tiny matmuls with one wider matmul.  This runs
            // before optional quantization, so the fused tensor is quantized by
            // the normal model conversion path when a quantized variant is made.
            func fuseRows(_ keys: [String], into outputKey: String) {
                guard keys.allSatisfy({ sanitized[$0] != nil }) else { return }
                let tensors = keys.compactMap { sanitized[$0] }
                sanitized[outputKey] = concatenated(tensors, axis: 0)
                for key in keys {
                    sanitized[key] = nil
                }
            }

            if let gateWeight = sanitized.removeValue(forKey: "\(prefix).mlp.gate.weight") {
                sanitized["\(prefix).mlp.gate.gate_proj.weight"] = gateWeight
            }

            if configuration.attentionKind(forLayer: layer) == .kda {
                let attentionPrefix = "\(prefix).attention"
                fuseRows(
                    [
                        "\(attentionPrefix).q_proj.weight",
                        "\(attentionPrefix).k_proj.weight",
                        "\(attentionPrefix).v_proj.weight",
                    ],
                    into: "\(attentionPrefix).qkv_proj.weight"
                )
                fuseRows(
                    [
                        "\(attentionPrefix).f_proj.weight",
                        "\(attentionPrefix).b_proj.weight",
                        "\(attentionPrefix).g_proj.weight",
                    ],
                    into: "\(attentionPrefix).fbg_proj.weight"
                )
                for projection in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                    let key = "\(prefix).attention.\(projection).weight"
                    if let weight = sanitized[key], weight.ndim == 3, weight.dim(1) == 1 {
                        sanitized[key] = weight.movedAxis(source: 2, destination: 1)
                    }
                }
            }

            fuseRows(
                ["\(prefix).mlp.gate_proj.weight", "\(prefix).mlp.up_proj.weight"],
                into: "\(prefix).mlp.gate_up_proj.weight"
            )

            guard layer >= configuration.firstKDenseReplace else { continue }
            let expertGateKeys = (0 ..< configuration.numExperts).map {
                "\(prefix).mlp.experts.\($0).gate_proj.weight"
            }
            let expertUpKeys = (0 ..< configuration.numExperts).map {
                "\(prefix).mlp.experts.\($0).up_proj.weight"
            }
            let expertDownKeys = (0 ..< configuration.numExperts).map {
                "\(prefix).mlp.experts.\($0).down_proj.weight"
            }
            if expertGateKeys.allSatisfy({ sanitized[$0] != nil })
                && expertUpKeys.allSatisfy({ sanitized[$0] != nil }) {
                let fusedExperts = zip(expertGateKeys, expertUpKeys).map { gateKey, upKey in
                    concatenated([sanitized[gateKey]!, sanitized[upKey]!], axis: 0)
                }
                sanitized["\(prefix).mlp.switch_mlp.gate_up_proj.weight"] = stacked(fusedExperts)
                for key in expertGateKeys + expertUpKeys {
                    sanitized[key] = nil
                }
            }
            if expertDownKeys.allSatisfy({ sanitized[$0] != nil }) {
                sanitized["\(prefix).mlp.switch_mlp.down_proj.weight"] = stacked(
                    expertDownKeys.compactMap { sanitized[$0] })
                for key in expertDownKeys {
                    sanitized[key] = nil
                }
            }

            if configuration.numSharedExperts > 0 {
                fuseRows(
                    [
                        "\(prefix).mlp.shared_experts.gate_proj.weight",
                        "\(prefix).mlp.shared_experts.up_proj.weight",
                    ],
                    into: "\(prefix).mlp.shared_experts.gate_up_proj.weight"
                )
            }
        }
        return sanitized
    }
}

extension BailingMoeV3Model: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}
