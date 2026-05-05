// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "EdgeRunnerChatApp",
    platforms: [
        .iOS(.v26),
        .macOS(.v26),
    ],
    products: [
        .library(name: "EdgeRunnerChatAppCore", targets: ["EdgeRunnerChatAppCore"]),
        .executable(name: "EdgeRunnerChatApp", targets: ["EdgeRunnerChatApp"]),
    ],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .target(
            name: "EdgeRunnerChatAppCore",
            dependencies: [
                .product(name: "EdgeRunner", package: "EdgeRunner"),
            ]
        ),
        .executableTarget(
            name: "EdgeRunnerChatApp",
            dependencies: ["EdgeRunnerChatAppCore"]
        ),
        .testTarget(
            name: "EdgeRunnerChatAppCoreTests",
            dependencies: ["EdgeRunnerChatAppCore"]
        ),
    ]
)
