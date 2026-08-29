import AVFoundation
import Foundation

enum StripAudioError: Error, CustomStringConvertible {
  case invalidArguments
  case missingVideoTrack
  case cannotCreateTrack
  case cannotCreateExporter

  var description: String {
    switch self {
    case .invalidArguments:
      return "usage: strip_audio <input.mp4> <output.mp4>"
    case .missingVideoTrack:
      return "input has no video track"
    case .cannotCreateTrack:
      return "could not create a composition video track"
    case .cannotCreateExporter:
      return "could not create an AVAsset export session"
    }
  }
}

@main
struct StripAudio {
  static func main() async {
    do {
      try await run()
    } catch {
      FileHandle.standardError.write(Data("\(error)\n".utf8))
      exit(1)
    }
  }

  private static func run() async throws {
    guard CommandLine.arguments.count == 3 else {
      throw StripAudioError.invalidArguments
    }

    let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
    let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
    let asset = AVURLAsset(url: inputURL)
    let duration = try await asset.load(.duration)
    guard let sourceTrack = try await asset.loadTracks(withMediaType: .video).first else {
      throw StripAudioError.missingVideoTrack
    }

    let composition = AVMutableComposition()
    guard let destinationTrack = composition.addMutableTrack(
      withMediaType: .video,
      preferredTrackID: kCMPersistentTrackID_Invalid
    ) else {
      throw StripAudioError.cannotCreateTrack
    }
    try destinationTrack.insertTimeRange(
      CMTimeRange(start: .zero, duration: duration),
      of: sourceTrack,
      at: .zero
    )
    destinationTrack.preferredTransform = try await sourceTrack.load(.preferredTransform)

    guard let exporter = AVAssetExportSession(
      asset: composition,
      presetName: AVAssetExportPresetPassthrough
    ) else {
      throw StripAudioError.cannotCreateExporter
    }
    exporter.shouldOptimizeForNetworkUse = true
    try await exporter.export(to: outputURL, as: .mp4)
  }
}
