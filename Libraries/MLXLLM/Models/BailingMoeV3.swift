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
