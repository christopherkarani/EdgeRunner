// swift-tools-version: 6.2

import PackageDescription
import Foundation

// The ANE / Espresso experimental path links against the Apple-only IOSurface
// framework. Some build environments (cloud CI, sandboxed Linux containers,
// macOS images with an incomplete SDK) can't supply `IOSurface/IOSurface.h`
// and the build fails before any unrelated test — including the canonical
// `PublishableBenchmark/fullBenchmark` — can run. Setting
// `EDGERUNNER_SKIP_ANE=1` at package-resolution time drops these targets
// (and the `EspressoEdgeRunner` library product) from the manifest. The
// non-ANE module graph (EdgeRunner / Core / IO / Metal / SharedTypes) is
// untouched.
let skipANE = ProcessInfo.processInfo.environment["EDGERUNNER_SKIP_ANE"] == "1"

let aneTargets: [Target] = skipANE ? [] : [
    .target(
        name: "ANEInteropIO",
        path: "Sources/ANEInteropIO",
        publicHeadersPath: "include",
        linkerSettings: [.linkedFramework("IOSurface")]
    ),
    .target(
        name: "EspressoEdgeRunner",
        dependencies: ["EdgeRunnerIO", "EdgeRunnerMetal", "ANEInteropIO"],
        path: "Sources/EspressoEdgeRunner"
    ),
    .testTarget(
        name: "EspressoEdgeRunnerTests",
        dependencies: ["EspressoEdgeRunner", "EdgeRunnerIO", "EdgeRunnerMetal"]
    ),
]

let aneProducts: [Product] = skipANE ? [] : [
    .library(name: "EspressoEdgeRunner", targets: ["EspressoEdgeRunner"]),
]

let package = Package(
    name: "EdgeRunner",
    platforms: [
        .iOS(.v26),
        .macOS(.v26),
    ],
    products: [
        .library(name: "EdgeRunner", targets: ["EdgeRunner"]),
    ] + aneProducts,
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
            name: "EdgeRunnerIO",
            dependencies: ["EdgeRunnerMetal"],
            path: "Sources/EdgeRunnerIO"
        ),
        .target(
            name: "EdgeRunnerCore",
            dependencies: ["EdgeRunnerMetal", "EdgeRunnerSharedTypes", "EdgeRunnerIO"],
            path: "Sources/EdgeRunnerCore"
        ),
        .target(
            name: "EdgeRunner",
            dependencies: ["EdgeRunnerCore", "EdgeRunnerIO", "EdgeRunnerSharedTypes"],
            path: "Sources/EdgeRunner"
        ),
        .testTarget(
            name: "EdgeRunnerIOTests",
            dependencies: ["EdgeRunnerIO", "EdgeRunnerMetal"]
        ),
        .testTarget(
            name: "EdgeRunnerCoreTests",
            dependencies: ["EdgeRunnerCore", "EdgeRunner", "EdgeRunnerMetal", "EdgeRunnerSharedTypes", "EdgeRunnerIO"]
        ),
        .testTarget(
            name: "EdgeRunnerMetalTests",
            dependencies: ["EdgeRunnerMetal", "EdgeRunnerSharedTypes"]
        ),
        .testTarget(
            name: "EdgeRunnerTests",
            dependencies: ["EdgeRunner"]
        ),
    ] + aneTargets
)
