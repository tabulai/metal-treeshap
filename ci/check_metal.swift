import Darwin
import Foundation
import Metal

guard let device = MTLCreateSystemDefaultDevice() else {
    let message =
        "ERROR: this runner has no usable Metal device. " +
        "Set the repository variable METAL_RUNNER to an Apple-silicon runner " +
        "with GPU access (for example a macos-15-xlarge or self-hosted label).\n"
    FileHandle.standardError.write(Data(message.utf8))
    exit(EXIT_FAILURE)
}

print("Metal device: \(device.name)")
print("Registry ID: \(device.registryID)")
print("Unified memory: \(device.hasUnifiedMemory)")
