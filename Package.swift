// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "EdgeRunner",
    platforms: [
        .iOS(.v26),
        .macOS(.v26),
    ],
    products: [
        .library(name: "EdgeRunner", targets: ["EdgeRunner"]),
    ],
    targets: [
        .target(
            name: "EdgeRunnerSharedTypes",
            path: "Sources/EdgeRunnerSharedTypes",
            publicHeadersPath: "include"
        ),
        .target(
            name: "EdgeRunnerMetal",
            dependencies: ["EdgeRunnerSharedTypes"],
            path: "Sources/EdgeRunnerMetal",
            resources: [.process("Shaders")]
        ),
        .target(
            name: "EdgeRunnerCore",
            dependencies: ["EdgeRunnerMetal", "EdgeRunnerSharedTypes"],
            path: "Sources/EdgeRunnerCore"
        ),
        .target(
            name: "EdgeRunner",
            dependencies: ["EdgeRunnerCore"],
            path: "Sources/EdgeRunner"
        ),
        .testTarget(
            name: "EdgeRunnerCoreTests",
            dependencies: ["EdgeRunnerCore", "EdgeRunnerMetal", "EdgeRunnerSharedTypes"]
        ),
        .testTarget(
            name: "EdgeRunnerMetalTests",
            dependencies: ["EdgeRunnerMetal", "EdgeRunnerSharedTypes"]
        ),
    ]
)
