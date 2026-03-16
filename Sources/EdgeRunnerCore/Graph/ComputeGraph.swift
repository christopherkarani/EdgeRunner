import Foundation

public enum ComputeGraph {
    public static let maxFusionDepth = 11
    public static let maxBufferArgs = 31

    public static func topologicalSort(root: TensorOp) -> [TensorOp] {
        var visited = Set<UUID>()
        var result = [TensorOp]()

        func visit(_ node: TensorOp) {
            guard !visited.contains(node.id) else { return }
            visited.insert(node.id)
            for input in node.inputs { visit(input) }
            result.append(node)
        }

        visit(root)
        return result
    }

    public static func identifyFusionGroups(_ sorted: [TensorOp]) -> [[TensorOp]] {
        var groups = [[TensorOp]]()
        var currentGroup = [TensorOp]()
        var currentShape: Shape?

        for op in sorted {
            if op.isElementwise {
                if let shape = currentShape, shape == op.outputShape, currentGroup.count < maxFusionDepth {
                    currentGroup.append(op)
                } else {
                    if !currentGroup.isEmpty { groups.append(currentGroup) }
                    currentGroup = [op]
                    currentShape = op.outputShape
                }
            } else {
                if !currentGroup.isEmpty {
                    groups.append(currentGroup)
                    currentGroup = []
                    currentShape = nil
                }
                groups.append([op])
            }
        }
        if !currentGroup.isEmpty { groups.append(currentGroup) }
        return groups
    }
}
