// Load and save FontDiffuser weights as .safetensors, the format the Python package writes.
//
// To start from the official PyTorch checkpoints, convert them once with the Python package
// (`fdmlx convert`, no PyTorch needed); the parameter names and layouts are the same in both versions.

import Foundation
import MLX
import MLXNN

public enum WeightsError: Error, CustomStringConvertible {
    case notSafetensors(URL)

    public var description: String {
        switch self {
        case .notSafetensors(let url):
            return "\(url.path) is not a .safetensors file. Convert PyTorch checkpoints first with `fdmlx convert`."
        }
    }
}

extension FontDiffuser {
    /// Load weights. Strict: every parameter must be in the file, with the right shape, and nothing else.
    public func load(weights url: URL) throws {
        guard url.pathExtension == "safetensors" else { throw WeightsError.notSafetensors(url) }
        let arrays = try loadArrays(url: url)
        try update(parameters: ModuleParameters.unflattened(arrays), verify: [.all])
        eval(self)
    }

    /// Save all parameters, including the spectral-norm vectors, so the file loads in Swift or Python.
    public func save(weights url: URL) throws {
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: parameters().flattened()), url: url)
    }
}
