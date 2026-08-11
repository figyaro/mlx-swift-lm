import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, up in
    MLXNN.silu(gate) * up
}

public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        let activated =
            if let activationProduct {
                activationProduct(xGate, xUp)
            } else {
                activation(xGate) * xUp
            }
        x = downProj(
            activated,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - FusedGateUpSwitchGLU

/// Optional decode-only weight compression for routed expert projections.
///
/// Single-token MoE decode is launch-bound when every selected expert is kept
/// in BF16. A model can opt into a cached quantized representation without
/// changing its prefill path or checkpoint format.
public struct SwitchDecodeQuantization: Sendable {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(groupSize: Int, bits: Int, mode: QuantizationMode) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }
}

/// SwitchGLU variant for models that ship a single fused `gate_up_proj` weight
/// of shape `[numExperts, 2*hiddenDims, inputDims]` instead of separate
/// `gate_proj` / `up_proj`. Used by Gemma 4 26B MoE.
public class FusedGateUpSwitchGLU: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?
    let decodeQuantization: SwitchDecodeQuantization?
    private var decodeQuantizedWeights:
        (gateWeight: MLXArray, gateScales: MLXArray, gateBiases: MLXArray?,
        downWeight: MLXArray, downScales: MLXArray, downBiases: MLXArray?)?
    private let decodeQuantizedWeightsLock = NSLock()

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false,
        decodeQuantization: SwitchDecodeQuantization? = nil
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct
        self.decodeQuantization = decodeQuantization

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: 2 * hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false,
        decodeQuantization: SwitchDecodeQuantization? = nil
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil
        self.decodeQuantization = decodeQuantization

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: 2 * hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        if let decodeQuantization,
            !(gateUpProj is QuantizedSwitchLinear),
            !(downProj is QuantizedSwitchLinear),
            !doSort, gateUpProj.bias == nil, downProj.bias == nil,
            indices.dim(0) == 1, indices.dim(1) == 1 {
            decodeQuantizedWeightsLock.lock()
            if decodeQuantizedWeights == nil {
                let gate = MLX.quantized(
                    gateUpProj.weight,
                    groupSize: decodeQuantization.groupSize,
                    bits: decodeQuantization.bits,
                    mode: decodeQuantization.mode)
                let down = MLX.quantized(
                    downProj.weight,
                    groupSize: decodeQuantization.groupSize,
                    bits: decodeQuantization.bits,
                    mode: decodeQuantization.mode)
                // MLXArray graphs are lazy and are not safe to publish for
                // concurrent evaluation. Materialize the shared weights while
                // the initialization lock is held so each session only reads
                // immutable, evaluated tensors during decode.
                var quantizedArrays = [gate.0, gate.1, down.0, down.1]
                if let gateBiases = gate.2 {
                    quantizedArrays.append(gateBiases)
                }
                if let downBiases = down.2 {
                    quantizedArrays.append(downBiases)
                }
                eval(quantizedArrays)
                decodeQuantizedWeights = (
                    gateWeight: gate.0, gateScales: gate.1, gateBiases: gate.2,
                    downWeight: down.0, downScales: down.1, downBiases: down.2
                )
            }
            let quantized = decodeQuantizedWeights!
            decodeQuantizedWeightsLock.unlock()
            let gateUp = MLX.gatherQuantizedMM(
                x, quantized.gateWeight, scales: quantized.gateScales,
                biases: quantized.gateBiases, rhsIndices: idx, transpose: true,
                groupSize: decodeQuantization.groupSize,
                bits: decodeQuantization.bits,
                mode: decodeQuantization.mode,
                sortedIndices: false)
            let parts = MLX.split(gateUp, parts: 2, axis: -1)
            let activated =
                if let activationProduct {
                    activationProduct(parts[0], parts[1])
                } else {
                    activation(parts[0]) * parts[1]
                }
            let down = MLX.gatherQuantizedMM(
                activated, quantized.downWeight, scales: quantized.downScales,
                biases: quantized.downBiases, rhsIndices: idx, transpose: true,
                groupSize: decodeQuantization.groupSize,
                bits: decodeQuantization.bits,
                mode: decodeQuantization.mode,
                sortedIndices: false)
            return MLX.squeezed(down, axis: -2)
        }

        let gateUp = gateUpProj(x, idx, sortedIndices: doSort)
        let parts = MLX.split(gateUp, parts: 2, axis: -1)
        let activated =
            if let activationProduct {
                activationProduct(parts[0], parts[1])
            } else {
                activation(parts[0]) * parts[1]
            }
        x = downProj(
            activated,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - SwitchLinear

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}
