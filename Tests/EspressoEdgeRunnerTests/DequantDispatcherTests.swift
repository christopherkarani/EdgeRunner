import Testing
import Metal
@testable import EspressoEdgeRunner
import EdgeRunnerIO

@Suite("DequantDispatcher")
struct DequantDispatcherTests {

    private func makeDevice() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw EspressoError.metalDeviceUnavailable
        }
        return device
    }

    private func makeTensor(device: MTLDevice, floats: [Float], dataType: TensorDataType, shape: [Int]) -> TensorStorage {
        let byteCount = floats.count * MemoryLayout<Float>.size
        let buffer = device.makeBuffer(bytes: floats, length: byteCount, options: .storageModeShared)!
        return TensorStorage(
            buffer: buffer,
            byteOffset: 0,
            dataType: dataType,
            shape: shape,
            name: "test_tensor"
        )
    }

    @Test("Float32 passthrough returns same values")
    func float32Passthrough() async throws {
        let device = try makeDevice()
        let input: [Float] = [1.0, 2.0, 3.0, 4.0]
        let tensor = makeTensor(device: device, floats: input, dataType: .float32, shape: [4])

        let result = try await DequantDispatcher.dequantize(tensor: tensor, device: device)
        #expect(result == input)
    }

    @Test("Float16 conversion round-trips within tolerance")
    func float16Conversion() async throws {
        let device = try makeDevice()
        let values: [Float] = [1.0, 0.5, -1.0, 3.14]
        let fp16Values = values.map { Float16($0) }
        let fp16Bits = fp16Values.map { $0.bitPattern }

        let byteCount = fp16Bits.count * MemoryLayout<UInt16>.size
        let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        let ptr = buffer.contents().assumingMemoryBound(to: UInt16.self)
        for (i, bits) in fp16Bits.enumerated() {
            ptr[i] = bits
        }

        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .float16, shape: [4], name: "fp16_test"
        )

        let result = try await DequantDispatcher.dequantize(tensor: tensor, device: device)
        for (i, value) in result.enumerated() {
            #expect(abs(value - values[i]) < 1e-2, "Mismatch at \(i): \(value) vs \(values[i])")
        }
    }

    @Test("Q4_0 dispatch produces correct count via Metal kernel")
    func q4_0Dispatch() async throws {
        let device = try makeDevice()
        let blockCount = 2
        let byteCount = blockCount * 18
        let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        memset(buffer.contents(), 0, byteCount)

        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .q4_0, shape: [blockCount * 32], name: "q4_test"
        )

        let result = try await DequantDispatcher.dequantize(tensor: tensor, device: device)
        #expect(result.count == blockCount * 32)
    }

    @Test("Q5_0 dispatch produces correct count via Metal kernel")
    func q5_0Dispatch() async throws {
        let device = try makeDevice()
        let blockCount = 2
        let byteCount = blockCount * 22
        let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        memset(buffer.contents(), 0, byteCount)

        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .q5_0, shape: [blockCount * 32], name: "q5_0_test"
        )

        let result = try await DequantDispatcher.dequantize(tensor: tensor, device: device)
        #expect(result.count == blockCount * 32)
    }

    @Test("Q5_1 dispatch produces correct count via Metal kernel")
    func q5_1Dispatch() async throws {
        let device = try makeDevice()
        let blockCount = 3
        let byteCount = blockCount * 24
        let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        memset(buffer.contents(), 0, byteCount)

        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .q5_1, shape: [blockCount * 32], name: "q5_1_test"
        )

        let result = try await DequantDispatcher.dequantize(tensor: tensor, device: device)
        #expect(result.count == blockCount * 32)
    }

    @Test("Q2_K dispatch produces correct count via Metal kernel")
    func q2_kDispatch() async throws {
        let device = try makeDevice()
        let superBlockCount = 2
        let byteCount = superBlockCount * 84
        let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        memset(buffer.contents(), 0, byteCount)

        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .q2_K, shape: [superBlockCount * 256], name: "q2_k_test"
        )

        let result = try await DequantDispatcher.dequantize(tensor: tensor, device: device)
        #expect(result.count == superBlockCount * 256)
    }

    @Test("Q3_K dispatch produces correct count via Metal kernel")
    func q3_kDispatch() async throws {
        let device = try makeDevice()
        let superBlockCount = 2
        let byteCount = superBlockCount * 110
        let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        memset(buffer.contents(), 0, byteCount)

        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .q3_K, shape: [superBlockCount * 256], name: "q3_k_test"
        )

        let result = try await DequantDispatcher.dequantize(tensor: tensor, device: device)
        #expect(result.count == superBlockCount * 256)
    }

    @Test("Q5_K dispatch produces correct count via Metal kernel")
    func q5_kDispatch() async throws {
        let device = try makeDevice()
        let superBlockCount = 2
        let byteCount = superBlockCount * 176
        let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        memset(buffer.contents(), 0, byteCount)

        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .q5_K, shape: [superBlockCount * 256], name: "q5_k_test"
        )

        let result = try await DequantDispatcher.dequantize(tensor: tensor, device: device)
        #expect(result.count == superBlockCount * 256)
    }

    @Test("Q6_K dispatch produces correct count via Metal kernel")
    func q6_kDispatch() async throws {
        let device = try makeDevice()
        let superBlockCount = 2
        let byteCount = superBlockCount * 210
        let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
        memset(buffer.contents(), 0, byteCount)

        let tensor = TensorStorage(
            buffer: buffer, byteOffset: 0,
            dataType: .q6_K, shape: [superBlockCount * 256], name: "q6_k_test"
        )

        let result = try await DequantDispatcher.dequantize(tensor: tensor, device: device)
        #expect(result.count == superBlockCount * 256)
    }

    @Test("Unsupported data type throws")
    func unsupportedThrows() async {
        do {
            let device = try makeDevice()
            let buffer = device.makeBuffer(length: 4, options: .storageModeShared)!
            let tensor = TensorStorage(
                buffer: buffer, byteOffset: 0,
                dataType: .i8, shape: [4], name: "unsupported_test"
            )
            _ = try await DequantDispatcher.dequantize(tensor: tensor, device: device)
            Issue.record("Expected error to be thrown")
        } catch let error as EspressoError {
            if case .unsupportedDataType = error {
                // Expected
            } else {
                Issue.record("Wrong error type: \(error)")
            }
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("Non-aligned element count throws invalidTensorShape")
    func nonAlignedElementCountThrows() async {
        do {
            let device = try makeDevice()
            // 33 elements is not divisible by 32 (weightsPerBlock for q4_0)
            let buffer = device.makeBuffer(length: 100, options: .storageModeShared)!
            let tensor = TensorStorage(
                buffer: buffer, byteOffset: 0,
                dataType: .q4_0, shape: [33], name: "misaligned_test"
            )
            _ = try await DequantDispatcher.dequantize(tensor: tensor, device: device)
            Issue.record("Expected error to be thrown")
        } catch let error as EspressoError {
            if case .invalidTensorShape = error {
                // Expected
            } else {
                Issue.record("Wrong error type: \(error)")
            }
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("Buffer too small throws bufferOutOfBounds")
    func bufferTooSmallThrows() async {
        do {
            let device = try makeDevice()
            // Shape says 8 floats (32 bytes) but buffer only has 4 bytes
            let buffer = device.makeBuffer(length: 4, options: .storageModeShared)!
            let tensor = TensorStorage(
                buffer: buffer, byteOffset: 0,
                dataType: .float32, shape: [8], name: "oob_test"
            )
            _ = try await DequantDispatcher.dequantize(tensor: tensor, device: device)
            Issue.record("Expected error to be thrown")
        } catch let error as EspressoError {
            if case .bufferOutOfBounds = error {
                // Expected
            } else {
                Issue.record("Wrong error type: \(error)")
            }
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}
