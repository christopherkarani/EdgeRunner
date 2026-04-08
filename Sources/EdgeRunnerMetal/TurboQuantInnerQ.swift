import Foundation
import Metal
import Synchronization

public final class TurboQuantInnerQState: @unchecked Sendable {
    private struct State: Sendable {
        var squaredAccumulator: [Float]
        var sampleCount: Int
        var scale: [Float]
        var scaleInv: [Float]
        var isActive: Bool
        var calibrationFinished: Bool
    }

    public let configuration: TurboQuantInnerQConfiguration
    public let channelCount: Int
    public let buffer: MTLBuffer

    private let state: Mutex<State>

    public init(
        device: MTLDevice,
        configuration: TurboQuantInnerQConfiguration,
        channelCount: Int = TurboQuantLayout.supportedDimension
    ) throws {
        self.configuration = configuration
        self.channelCount = channelCount

        let ones = [Float](repeating: 1, count: channelCount)
        guard let buffer = device.makeBuffer(
            bytes: ones,
            length: ones.count * MemoryLayout<Float>.stride,
            options: [.storageModeShared, .hazardTrackingModeUntracked]
        ) else {
            throw GQAError.encodingFailed
        }
        self.buffer = buffer
        self.state = Mutex(
            State(
                squaredAccumulator: [Float](repeating: 0, count: channelCount),
                sampleCount: 0,
                scale: [Float](repeating: 1, count: channelCount),
                scaleInv: ones,
                isActive: false,
                calibrationFinished: !configuration.enabled || configuration.calibrationSampleCount <= 0
            )
        )
    }

    public var isEnabled: Bool {
        configuration.enabled && configuration.calibrationSampleCount > 0
    }

    public var isActive: Bool {
        state.withLock { $0.isActive }
    }

    public var sampleCount: Int {
        state.withLock { $0.sampleCount }
    }

    public func currentScaleInv() -> [Float] {
        state.withLock { $0.scaleInv }
    }

    public func currentScale() -> [Float] {
        state.withLock { $0.scale }
    }

    public func observe(rows: some Sequence<[Float]>) {
        guard isEnabled else { return }

        var finalizedScaleInv: [Float]?
        state.withLock { state in
            guard !state.calibrationFinished else { return }

            for row in rows {
                guard row.count >= channelCount else { continue }
                for index in 0..<channelCount {
                    let value = row[index]
                    state.squaredAccumulator[index] += value * value
                }
                state.sampleCount += 1
            }

            guard state.sampleCount >= configuration.calibrationSampleCount else { return }

            state.calibrationFinished = true
            let finalized = Self.finalize(
                squaredAccumulator: state.squaredAccumulator,
                sampleCount: state.sampleCount,
                strength: configuration.strength
            )
            state.scale = finalized.scale
            state.scaleInv = finalized.scaleInv
            state.isActive = finalized.isActive
            finalizedScaleInv = finalized.scaleInv
        }

        if let finalizedScaleInv {
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: channelCount)
            for index in 0..<channelCount {
                pointer[index] = finalizedScaleInv[index]
            }
        }
    }

    private static func finalize(
        squaredAccumulator: [Float],
        sampleCount: Int,
        strength: Float
    ) -> (scale: [Float], scaleInv: [Float], isActive: Bool) {
        guard sampleCount > 0 else {
            let ones = [Float](repeating: 1, count: squaredAccumulator.count)
            return (ones, ones, false)
        }

        let rms = squaredAccumulator.map { sqrt($0 / Float(sampleCount)) }
        let meanRMS = rms.reduce(Float.zero, +) / Float(max(rms.count, 1))
        var maxRatio: Float = 0
        var minRatio: Float = .greatestFiniteMagnitude
        var scale = [Float](repeating: 1, count: rms.count)
        var scaleInv = [Float](repeating: 1, count: rms.count)

        for index in rms.indices {
            let ratio = rms[index] > 1e-10 ? (meanRMS / rms[index]) : 1
            let unclamped = pow(ratio, strength)
            let clamped = min(max(unclamped, 0.5), 2.0)
            scale[index] = clamped
            scaleInv[index] = 1 / clamped
            maxRatio = max(maxRatio, ratio)
            minRatio = min(minRatio, ratio)
        }

        if maxRatio < 1.2 && minRatio > (1 / 1.2) {
            let ones = [Float](repeating: 1, count: rms.count)
            return (ones, ones, false)
        }

        return (scale, scaleInv, true)
    }
}
