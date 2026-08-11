import Foundation
import Testing

import MLX
@testable import MLXLLM
import MLXLMCommon

@Suite("BailingMoeV3 configuration")
struct BailingMoeV3Tests {
    @Test("decodes the Ling-3.0-tiny configuration")
    func decodesLingConfiguration() throws {
        let configuration = try BailingMoeV3Configuration(
            jsonData: Data(
                """
                {
                  "architectures": ["BailingMoeV3ForCausalLM"],
                  "model_type": "bailing_hybrid",
                  "hidden_size": 1536,
                  "intermediate_size": 4608,
                  "num_hidden_layers": 24,
                  "num_attention_heads": 16,
                  "num_key_value_heads": 16,
                  "head_dim": 128,
                  "vocab_size": 157184,
                  "max_position_embeddings": 131072,
                  "rms_norm_eps": 0.000001,
                  "rope_theta": 6000000,
                  "tie_word_embeddings": false,
                  "layer_group_size": 4,
                  "short_conv_kernel_size": 4,
                  "no_kda_lora": true,
                  "kda_safe_gate": true,
                  "kda_lower_bound": -5,
                  "gated_attention_proj_granularity_type": "head_wise",
                  "qk_head_dim": 192,
                  "qk_nope_head_dim": 128,
                  "qk_rope_head_dim": 64,
                  "v_head_dim": 128,
                  "q_lora_rank": 256,
                  "kv_lora_rank": 512,
                  "rope_interleave": true,
                  "use_qk_norm": true,
                  "first_k_dense_replace": 1,
                  "num_experts": 128,
                  "num_experts_per_tok": 8,
                  "num_shared_experts": 1,
                  "moe_intermediate_size": 512,
                  "moe_shared_expert_intermediate_size": 512,
                  "n_group": 8,
                  "topk_group": 4,
                  "norm_topk_prob": true,
                  "routed_scaling_factor": 2.5,
                  "moe_router_enable_expert_bias": true
                }
                """.utf8
            )
        )

        // Ling uses KDA for the first three layers of each group and MLA on the fourth.
        // The last layer is also MLA by the architecture's tail rule.
        // These assertions intentionally cover the values used by Ling-3.0-tiny.
        // The public surface is small so DENOJU can check support without constructing weights.
        // Keep the first RED assertion focused on model identity and dimensions.
        #expect(configuration.modelType == "bailing_hybrid")
        #expect(configuration.architecture == "BailingMoeV3ForCausalLM")
        #expect(configuration.hiddenSize == 1536)
        #expect(configuration.numHiddenLayers == 24)
        #expect(configuration.attentionKind(forLayer: 0) == .kda)
        #expect(configuration.attentionKind(forLayer: 3) == .mla)
        #expect(configuration.attentionKind(forLayer: 23) == .mla)
    }

    @Test("rejects non-Ling model identities")
    func rejectsNonLingModelIdentity() {
        #expect(throws: BailingMoeV3ConfigurationError.self) {
            try BailingMoeV3Configuration(
                jsonData: Data(
                    """
                    {
                      "architectures": ["LlamaForCausalLM"],
                      "model_type": "llama",
                      "hidden_size": 4096,
                      "num_hidden_layers": 32,
                      "num_attention_heads": 32,
                      "head_dim": 128,
                      "vocab_size": 32000,
                      "layer_group_size": 4,
                      "qk_head_dim": 128,
                      "qk_nope_head_dim": 64,
                      "qk_rope_head_dim": 64
                    }
                    """.utf8
                )
            )
        }
    }

    @Test("registers bailing_hybrid with the LLM factory")
    func registersBailingHybridModelType() async {
        #expect(await LLMTypeRegistry.shared.contains("bailing_hybrid"))
    }

    @Test("creates recurrent and attention caches for the hybrid layer schedule")
    func createsHybridCaches() throws {
        let configuration = try BailingMoeV3Configuration(
            jsonData: Data(
                """
                {
                  "architectures": ["BailingMoeV3ForCausalLM"],
                  "model_type": "bailing_hybrid",
                  "hidden_size": 8,
                  "intermediate_size": 16,
                  "num_hidden_layers": 4,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 2,
                  "head_dim": 4,
                  "vocab_size": 16,
                  "max_position_embeddings": 64,
                  "rms_norm_eps": 0.000001,
                  "rope_theta": 10000,
                  "tie_word_embeddings": false,
                  "layer_group_size": 4,
                  "short_conv_kernel_size": 4,
                  "no_kda_lora": true,
                  "kda_safe_gate": true,
                  "kda_lower_bound": -5,
                  "gated_attention_proj_granularity_type": "head_wise",
                  "qk_head_dim": 4,
                  "qk_nope_head_dim": 2,
                  "qk_rope_head_dim": 2,
                  "v_head_dim": 4,
                  "q_lora_rank": 2,
                  "kv_lora_rank": 4,
                  "rope_interleave": true,
                  "use_qk_norm": true,
                  "first_k_dense_replace": 1,
                  "num_experts": 4,
                  "num_experts_per_tok": 1,
                  "num_shared_experts": 1,
                  "moe_intermediate_size": 4,
                  "moe_shared_expert_intermediate_size": 4,
                  "n_group": 2,
                  "topk_group": 1,
                  "norm_topk_prob": true,
                  "routed_scaling_factor": 1,
                  "moe_router_enable_expert_bias": true
                }
                """.utf8
            )
        )

        let caches = BailingMoeV3Model(configuration).newCache(parameters: nil)

        #expect(caches.count == 4)
        #expect(caches[0] is MambaCache)
        #expect(caches[1] is MambaCache)
        #expect(caches[2] is MambaCache)
        #expect(caches[3] is KVCacheSimple)

        let model = BailingMoeV3Model(configuration)
        let logits = model(
            MLXArray([1, 2] as [Int32]).reshaped(1, 2), cache: model.newCache(parameters: nil))
        eval(logits)
        #expect(logits.shape == [1, 2, 16])
    }
}
