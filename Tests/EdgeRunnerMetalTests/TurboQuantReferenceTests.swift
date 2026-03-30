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
        #expect(balanced.codeWordsPerRow == 14)
        #expect(aggressive.bytesPerRow < balanced.bytesPerRow)
    }

    private func makeSignal() -> [Float] {
        (0..<128).map { index in
            let x = Float(index)
            return sin(x * 0.17) + cos(x * 0.07) * 0.5
        }
    }
}
