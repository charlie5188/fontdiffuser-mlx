// swift-tools-version:5.9
// FontDiffuser in MLX Swift: the same model, sampler and fine-tuning as the Python package (fdmlx/), for apps.
import PackageDescription

let package = Package(
    name: "FontDiffuserMLX",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "FontDiffuserMLX", targets: ["FontDiffuserMLX"]),
        .executable(name: "fdmlx-swift", targets: ["fdmlx-swift"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.33.0")),
    ],
    targets: [
        .target(
            name: "FontDiffuserMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "swift/FontDiffuserMLX"),
        .executableTarget(name: "fdmlx-swift", dependencies: ["FontDiffuserMLX"], path: "swift/fdmlx-swift"),
        // development check against the Python version (needs tests/swift_fixtures.py output)
        .executableTarget(
            name: "fdmlx-swift-check",
            dependencies: ["FontDiffuserMLX", .product(name: "MLXNN", package: "mlx-swift")],
            path: "swift/checks"),
    ]
)
