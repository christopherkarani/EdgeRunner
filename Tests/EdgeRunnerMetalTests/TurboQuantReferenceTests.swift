import Foundation
import Testing
@testable import EdgeRunnerMetal

@Suite("TurboQuant Reference")
struct TurboQuantReferenceTests {
    @Test func hadamardRoundTrip() throws {
        let vector = (0..<128).map { index in
            Float(index % 11) - 5.0
        }

        let transformed = try TurboQuantTransform.randomizedHadamard(vector, seed: 42)
        let recovered = try TurboQuantTransform.inverseRandomizedHadamard(transformed, seed: 42)

        for (lhs, rhs) in zip(vector, recovered) {
            #expect(abs(lhs - rhs) < 1e-4)
        }
    }

    @Test func planarRoundTrip() throws {
        let vector = (0..<128).map { index in
            sin(Float(index) * 0.13) + cos(Float(index) * 0.09)
        }

        let transformed = try TurboQuantTransform.planarRotate(vector, seed: 42)
        let recovered = try TurboQuantTransform.inversePlanarRotate(transformed, seed: 42)

        for (lhs, rhs) in zip(vector, recovered) {
            #expect(abs(lhs - rhs) < 1e-4)
        }
    }

    @Test func codebooksAreMonotonic() {
        for codebook in [TurboQuantCodebooks.twoBit, TurboQuantCodebooks.threeBit, TurboQuantCodebooks.fiveBit, TurboQuantCodebooks.sixBit, TurboQuantCodebooks.sevenBit] {
            #expect(codebook.centroids == codebook.centroids.sorted())
            #expect(codebook.thresholds == codebook.thresholds.sorted())
            #expect(codebook.thresholds.count == codebook.centroids.count - 1)
        }
    }

    @Test func aggressiveEncodingIsDeterministic() throws {
        let vector = makeSignal()
        let encodedA = try TurboQuantReferenceEncoder.encode(
            vector,
            preset: .aggressive,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )
        let encodedB = try TurboQuantReferenceEncoder.encode(
            vector,
            preset: .aggressive,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )

        #expect(encodedA == encodedB)
        #expect(encodedA.primaryCodes.count > 0)
        #expect(encodedA.residualSigns.count == 4)
        #expect(encodedA.outlierMask.count == 4)
    }

    @Test func approximateDecodeProducesFiniteValues() throws {
        let vector = makeSignal()
        let encoded = try TurboQuantReferenceEncoder.encode(
            vector,
            preset: .aggressive,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )
        let decoded = try TurboQuantReferenceEncoder.approximateDecode(
            encoded,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )

        #expect(decoded.count == vector.count)
        #expect(decoded.allSatisfy { $0.isFinite })

        let mse = zip(vector, decoded).reduce(Float.zero) { partial, pair in
            let delta = pair.0 - pair.1
            return partial + (delta * delta)
        } / Float(vector.count)

        #expect(mse < 10)
    }

    @Test func balancedPresetUsesHigherBitBudget() {
        let aggressive = TurboQuantPreset.aggressive.descriptor
        let balanced = TurboQuantPreset.balanced.descriptor

        #expect(balanced.effectiveBits > aggressive.effectiveBits)
        #expect(balanced.highPrecisionBits > aggressive.highPrecisionBits)
        #expect(balanced.regularBits > aggressive.regularBits)
    }

    @Test func forkAlignedFixedTypesUseExpectedBitBudgets() {
        let turbo2 = TurboQuantPreset.turbo2.descriptor
        let turbo3 = TurboQuantPreset.turbo3.descriptor
        let planar3 = TurboQuantPreset.planar3.descriptor
        let turbo4 = TurboQuantPreset.turbo4.descriptor

        #expect(turbo2.regularBits == 2)
        #expect(turbo2.highPrecisionBits == 2)
        #expect(turbo3.regularBits == 3)
        #expect(turbo3.highPrecisionBits == 3)
        #expect(planar3.regularBits == 3)
        #expect(planar3.highPrecisionBits == 3)
        #expect(turbo4.regularBits == 4)
        #expect(turbo4.highPrecisionBits == 4)
    }

    @Test func layoutMatchesExpectedRowSize() throws {
        let planar3 = try TurboQuantLayout(preset: .planar3)
        let turbo3 = try TurboQuantLayout(preset: .turbo3)
        let turbo4 = try TurboQuantLayout(preset: .turbo4)
        let aggressive = try TurboQuantLayout(preset: .aggressive)
        let balanced = try TurboQuantLayout(preset: .balanced)
        let sixBit = try TurboQuantLayout(preset: .sixBit)
        let sevenBit = try TurboQuantLayout(preset: .sevenBit)

        #expect(planar3.codeWordsPerRow == 12)
        #expect(turbo3.codeWordsPerRow == 12)
        #expect(turbo4.codeWordsPerRow == 16)
        #expect(aggressive.codeWordsPerRow == 9)
        #expect(aggressive.runtimeCodeWordsPerRow == 9)
        #expect(aggressive.baseWordsPerRow == 8)
        #expect(aggressive.sidebandWordsPerRow == 1)
        #expect(balanced.codeWordsPerRow == 14)
        #expect(balanced.runtimeCodeWordsPerRow == 14)
        #expect(balanced.baseWordsPerRow == 12)
        #expect(balanced.sidebandWordsPerRow == 2)
        #expect(sixBit.codeWordsPerRow == 24)
        #expect(sixBit.runtimeCodeWordsPerRow == 24)
        #expect(sixBit.sidebandWordsPerRow == 0)
        #expect(sevenBit.codeWordsPerRow == 28)
        #expect(sevenBit.runtimeCodeWordsPerRow == 28)
        #expect(sevenBit.sidebandWordsPerRow == 0)
        #expect(aggressive.bytesPerRow < balanced.bytesPerRow)
        #expect(planar3.bytesPerRow == turbo3.bytesPerRow)
        #expect(turbo3.bytesPerRow < turbo4.bytesPerRow)
        #expect(balanced.bytesPerRow < sixBit.bytesPerRow)
        #expect(sixBit.bytesPerRow < sevenBit.bytesPerRow)
    }

    @Test func paddedHeadDimensionRoundTripsReferencePath() throws {
        let vector = Array(makeSignal().prefix(96))
        let layout = try TurboQuantLayout(preset: .turbo3, dimension: vector.count)
        let encoded = try TurboQuantReferenceEncoder.encode(
            vector,
            preset: .turbo3,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )
        let decoded = try TurboQuantReferenceEncoder.approximateDecode(
            encoded,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )

        #expect(layout.dimension == 96)
        #expect(layout.paddedDimension == 128)
        #expect(decoded.count == 96)
        #expect(decoded.allSatisfy { $0.isFinite })
    }

    @Test func aggressivePackLayoutUsesSplitPlane() throws {
        let descriptor = TurboQuantPreset.aggressive.descriptor
        let mask = (0..<128).map { $0 % 4 == 0 }
        let codes = (0..<128).map { index -> UInt8 in
            if mask[index] {
                return UInt8((index / 4) % 8)
            } else {
                return UInt8(index % 4)
            }
        }

        let packed = try BitPacker.packCodes(
            codes,
            outlierMask: mask,
            regularBits: descriptor.regularBits,
            highPrecisionBits: descriptor.highPrecisionBits
        )
        let unpacked = try BitPacker.unpackCodes(
            packed,
            count: 128,
            outlierMask: mask,
            regularBits: descriptor.regularBits,
            highPrecisionBits: descriptor.highPrecisionBits
        )

        #expect(unpacked == codes)
        let layout = try TurboQuantLayout(preset: .aggressive)
        #expect(packed.count == layout.codeWordsPerRow)
    }

    @Test func aggressiveRuntimeRowDecodesLikeLogicalRow() throws {
        let vector = makeSignal()
        let encoded = try TurboQuantReferenceEncoder.encode(
            vector,
            preset: .aggressive,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )
        let runtime = try TurboQuantReferenceEncoder.makeRuntimeRow(from: encoded)
        let logicalDecoded = try TurboQuantReferenceEncoder.approximateDecode(
            encoded,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )
        let runtimeDecoded = try TurboQuantReferenceEncoder.approximateDecode(
            runtimeRow: runtime,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )

        let layout = try TurboQuantLayout(preset: .aggressive)
        #expect(runtime.primaryCodes.count == layout.runtimeCodeWordsPerRow)
        for (lhs, rhs) in zip(logicalDecoded, runtimeDecoded) {
            #expect(abs(lhs - rhs) < 1e-5)
        }
    }

    @Test func balancedRuntimeRowDecodesLikeLogicalRow() throws {
        let vector = makeSignal()
        let encoded = try TurboQuantReferenceEncoder.encode(
            vector,
            preset: .balanced,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )
        let runtime = try TurboQuantReferenceEncoder.makeRuntimeRow(from: encoded)
        let logicalDecoded = try TurboQuantReferenceEncoder.approximateDecode(
            encoded,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )
        let runtimeDecoded = try TurboQuantReferenceEncoder.approximateDecode(
            runtimeRow: runtime,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )

        let layout = try TurboQuantLayout(preset: .balanced)
        #expect(runtime.primaryCodes.count == layout.runtimeCodeWordsPerRow)
        for (lhs, rhs) in zip(logicalDecoded, runtimeDecoded) {
            #expect(abs(lhs - rhs) < 1e-5)
        }
    }

    @Test func planar3EncodingIsDeterministic() throws {
        let vector = makeSignal()
        let encodedA = try TurboQuantReferenceEncoder.encode(
            vector,
            preset: .planar3,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )
        let encodedB = try TurboQuantReferenceEncoder.encode(
            vector,
            preset: .planar3,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )

        #expect(encodedA == encodedB)
        #expect(encodedA.primaryCodes.count == 12)
        #expect(encodedA.residualNorm == 0)
    }

    @Test func planar3RuntimeRowDecodesLikeLogicalRow() throws {
        let vector = makeSignal()
        let encoded = try TurboQuantReferenceEncoder.encode(
            vector,
            preset: .planar3,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )
        let runtime = try TurboQuantReferenceEncoder.makeRuntimeRow(from: encoded)
        let logicalDecoded = try TurboQuantReferenceEncoder.approximateDecode(
            encoded,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )
        let runtimeDecoded = try TurboQuantReferenceEncoder.approximateDecode(
            runtimeRow: runtime,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )

        let layout = try TurboQuantLayout(preset: .planar3)
        #expect(runtime.primaryCodes.count == layout.runtimeCodeWordsPerRow)
        for (lhs, rhs) in zip(logicalDecoded, runtimeDecoded) {
            #expect(abs(lhs - rhs) < 1e-5)
        }
    }

    @Test func planar3PreservesCentroidAlignedPairSignalBetterThanTurbo3() throws {
        let centroidAligned = try makePlanarCentroidAlignedSignal()
        let planarEncoded = try TurboQuantReferenceEncoder.encode(
            centroidAligned,
            preset: .planar3,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )
        let turboEncoded = try TurboQuantReferenceEncoder.encode(
            centroidAligned,
            preset: .turbo3,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )

        let planarDecoded = try TurboQuantReferenceEncoder.approximateDecode(
            planarEncoded,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )
        let turboDecoded = try TurboQuantReferenceEncoder.approximateDecode(
            turboEncoded,
            rotationSeed: TurboQuantSeeds.testRotation,
            residualSeed: TurboQuantSeeds.testResidual
        )

        let planarMSE = mse(lhs: centroidAligned, rhs: planarDecoded)
        let turboMSE = mse(lhs: centroidAligned, rhs: turboDecoded)

        #expect(planarMSE.isFinite)
        #expect(turboMSE.isFinite)
        #expect(planarMSE < turboMSE)
    }

    private func makeSignal() -> [Float] {
        (0..<128).map { index in
            let x = Float(index)
            return sin(x * 0.17) + cos(x * 0.07) * 0.5
        }
    }

    private func makePlanarCentroidAlignedSignal() throws -> [Float] {
        let centroids = TurboQuantCodebooks.threeBit.centroids
        var rotated = [Float](repeating: 0, count: 128)
        for pair in 0..<64 {
            let base = pair * 2
            rotated[base] = centroids[(pair + 1) % centroids.count]
            rotated[base + 1] = centroids[(pair * 3 + 2) % centroids.count]
        }
        return try TurboQuantTransform.inversePlanarRotate(rotated, seed: TurboQuantSeeds.testRotation)
    }

    private func mse(lhs: [Float], rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float.zero) { partial, pair in
            let delta = pair.0 - pair.1
            return partial + (delta * delta)
        } / Float(lhs.count)
    }
}
