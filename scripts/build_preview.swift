import AppKit
import AVFoundation
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct PreviewSource {
  let label: String
  let path: String
}

enum PreviewError: Error, CustomStringConvertible {
  case cannotCreateBitmap
  case cannotCreateDestination
  case cannotCreateFrame(String)
  case cannotFinalize

  var description: String {
    switch self {
    case .cannotCreateBitmap:
      return "could not create the preview bitmap"
    case .cannotCreateDestination:
      return "could not create the GIF destination"
    case let .cannotCreateFrame(label):
      return "could not sample a frame for \(label)"
    case .cannotFinalize:
      return "could not finalize the GIF"
    }
  }
}

struct BuildPreview {
  private static let width = 720
  private static let height = 480
  private static let columns = 3
  private static let rows = 2
  private static let framesPerSecond = 10
  private static let frameCount = 24
  private static let startTime = 0.25

  private static let sources = [
    PreviewSource(label: "Wan 2.2 S2V", path: "results/videos/wan22-s2v-stock-avatar.mp4"),
    PreviewSource(label: "LongCat Avatar", path: "results/videos/longcat-stock-avatar.mp4"),
    PreviewSource(label: "LiveAvatar", path: "results/videos/liveavatar-stock-avatar.mp4"),
    PreviewSource(label: "SoulX FlashTalk", path: "results/videos/soulx-flashtalk-stock-avatar.mp4"),
    PreviewSource(label: "EchoMimic V3", path: "results/videos/echomimic-v3-flash-stock-avatar.mp4"),
    PreviewSource(label: "SoulX FlashHead", path: "results/videos/soulx-flashhead-stock-avatar.mp4"),
  ]

  static func main() {
    do {
      try run()
    } catch {
      FileHandle.standardError.write(Data("\(error)\n".utf8))
      exit(1)
    }
  }

  private static func run() throws {
    let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let outputPath = CommandLine.arguments.dropFirst().first
      ?? "assets/previews/talking-avatar-results-grid.gif"
    let outputURL = root.appendingPathComponent(outputPath)
    try FileManager.default.createDirectory(
      at: outputURL.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )

    let generators = sources.map { source -> AVAssetImageGenerator in
      let asset = AVURLAsset(url: root.appendingPathComponent(source.path))
      let generator = AVAssetImageGenerator(asset: asset)
      generator.appliesPreferredTrackTransform = true
      generator.requestedTimeToleranceBefore = .zero
      generator.requestedTimeToleranceAfter = .zero
      return generator
    }

    guard let destination = CGImageDestinationCreateWithURL(
      outputURL as CFURL,
      UTType.gif.identifier as CFString,
      frameCount,
      nil
    ) else {
      throw PreviewError.cannotCreateDestination
    }

    let loopProperties = [
      kCGImagePropertyGIFDictionary: [
        kCGImagePropertyGIFLoopCount: 0,
      ],
    ] as CFDictionary
    CGImageDestinationSetProperties(destination, loopProperties)

    let frameDelay = 1.0 / Double(framesPerSecond)
    let frameProperties = [
      kCGImagePropertyGIFDictionary: [
        kCGImagePropertyGIFDelayTime: frameDelay,
        kCGImagePropertyGIFUnclampedDelayTime: frameDelay,
      ],
    ] as CFDictionary

    for frameIndex in 0..<frameCount {
      let seconds = startTime + (Double(frameIndex) * frameDelay)
      let time = CMTime(seconds: seconds, preferredTimescale: 600)
      let images = try zip(sources, generators).map { source, generator in
        do {
          return try generator.copyCGImage(at: time, actualTime: nil)
        } catch {
          throw PreviewError.cannotCreateFrame(source.label)
        }
      }
      let frame = try makeFrame(images: images)
      CGImageDestinationAddImage(destination, frame, frameProperties)
    }

    guard CGImageDestinationFinalize(destination) else {
      throw PreviewError.cannotFinalize
    }
    print("Wrote \(outputPath) (\(frameCount) frames at \(framesPerSecond) fps)")
  }

  private static func makeFrame(images: [CGImage]) throws -> CGImage {
    guard let bitmap = NSBitmapImageRep(
      bitmapDataPlanes: nil,
      pixelsWide: width,
      pixelsHigh: height,
      bitsPerSample: 8,
      samplesPerPixel: 4,
      hasAlpha: true,
      isPlanar: false,
      colorSpaceName: .deviceRGB,
      bytesPerRow: 0,
      bitsPerPixel: 0
    ), let context = NSGraphicsContext(bitmapImageRep: bitmap) else {
      throw PreviewError.cannotCreateBitmap
    }

    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    NSColor.black.setFill()
    NSRect(x: 0, y: 0, width: width, height: height).fill()

    let cellWidth = width / columns
    let cellHeight = height / rows
    for (index, image) in images.enumerated() {
      let column = index % columns
      let row = index / columns
      let cell = NSRect(
        x: column * cellWidth,
        y: height - ((row + 1) * cellHeight),
        width: cellWidth,
        height: cellHeight
      )
      draw(image: image, label: sources[index].label, in: cell)
    }

    context.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()
    guard let image = bitmap.cgImage else {
      throw PreviewError.cannotCreateBitmap
    }
    return image
  }

  private static func draw(image: CGImage, label: String, in cell: NSRect) {
    let sourceWidth = CGFloat(image.width)
    let sourceHeight = CGFloat(image.height)
    let cropSize = min(sourceWidth, sourceHeight)
    let sourceRect = NSRect(
      x: (sourceWidth - cropSize) / 2,
      y: (sourceHeight - cropSize) / 2,
      width: cropSize,
      height: cropSize
    )
    let sourceImage = NSImage(cgImage: image, size: NSSize(width: sourceWidth, height: sourceHeight))
    sourceImage.draw(
      in: cell,
      from: sourceRect,
      operation: .copy,
      fraction: 1,
      respectFlipped: false,
      hints: [.interpolation: NSImageInterpolation.medium]
    )

    let labelHeight: CGFloat = 30
    NSColor(calibratedWhite: 0, alpha: 0.72).setFill()
    NSRect(x: cell.minX, y: cell.minY, width: cell.width, height: labelHeight).fill()

    let style = NSMutableParagraphStyle()
    style.alignment = .center
    let attributes: [NSAttributedString.Key: Any] = [
      .font: NSFont.systemFont(ofSize: 14, weight: .semibold),
      .foregroundColor: NSColor.white,
      .paragraphStyle: style,
    ]
    label.draw(
      in: NSRect(x: cell.minX + 6, y: cell.minY + 7, width: cell.width - 12, height: 18),
      withAttributes: attributes
    )

    NSColor(calibratedWhite: 1, alpha: 0.18).setStroke()
    NSBezierPath(rect: cell.insetBy(dx: 0.5, dy: 0.5)).stroke()
  }
}

BuildPreview.main()
