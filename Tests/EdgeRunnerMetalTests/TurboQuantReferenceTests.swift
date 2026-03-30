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

    @Test func codebooksAreMonotonic() {
        for codebook in [TurboQuantCodebooks.twoBit, TurboQuantCodebooks.threeBit, TurboQuantCodebooks.fiveBit] {
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

    @Test func layoutMatchesExpectedRowSize() throws {
        let aggressive = try TurboQuantLayout(preset: .aggressive)
        let balanced = try TurboQuantLayout(preset: .balanced)

        #expect(aggressive.codeWordsPerRow == 9)
        #expect(aggressive.runtimeCodeWordsPerRow == 9)
        #expect(aggressive.baseWordsPerRow == 8)
        #expect(aggressive.sidebandWordsPerRow == 1)
        #expect(balanced.codeWordsPerRow == 14)
        #expect(balanced.runtimeCodeWordsPerRow == 14)
        #expect(balanced.baseWordsPerRow == 12)
        #expect(balanced.sidebandWordsPerRow == 2)
        #expect(aggressive.bytesPerRow < balanced.bytesPerRow)
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

    private func makeSignal() -> [Float] {
        (0..<128).map { index in
            let x = Float(index)
            return sin(x * 0.17) + cos(x * 0.07) * 0.5
        }
    }
}
