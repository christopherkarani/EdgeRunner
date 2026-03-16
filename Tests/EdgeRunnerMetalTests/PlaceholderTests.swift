import Testing
@testable import EdgeRunnerMetal

@Test func metalBackendExists() async {
    let backend = MetalBackend.shared
    let device = await backend.device
    #expect(device.name.isEmpty == false)
}
