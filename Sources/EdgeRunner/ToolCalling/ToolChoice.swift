import Foundation

public enum ToolChoice: Sendable, Equatable {
    case auto, required, none, specific(String)
}
