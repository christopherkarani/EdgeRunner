import Testing
import Metal
@testable import EdgeRunnerCore

@Suite("AutoTuner")
struct AutoTunerTests {
    @Test func defaultConfig() {
        let config = ThreadgroupConfig.default
        #expect(config.width > 0)
        #expect(config.height >= 1)
    }

    @Test func configForElementwise() {
        let config = AutoTuner.config(for: .elementwise, elementCount: 1024)
        #expect(config.width > 0)
        #expect(config.width <= 1024)
    }

    @Test func configForReduction() {
        let config = AutoTuner.config(for: .reduction, elementCount: 4096)
        #expect(config.width > 0)
    }

    @Test func configForTranspose() {
        let config = AutoTuner.config(for: .transpose, elementCount: 256)
        #expect(config.width > 0)
        #expect(config.height > 0)
    }

    @Test func threadgroupsCalculation() {
        let config = ThreadgroupConfig(width: 256, height: 1, depth: 1)
        let groups = config.threadgroups(for: 1000)
        #expect(groups.width == 4)
        #expect(groups.height == 1)
    }
}
