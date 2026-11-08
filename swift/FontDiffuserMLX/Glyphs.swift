// Glyph images in and out: render characters from a font file the way FontDiffuser was trained (black on white,
// the glyph's ink box centred in a 128 px square, resized to 96 px, scaled to [-1, 1]); turn a photo of one
// character into a style reference; write PNGs.
//
// Translated from fdmlx/glyphs.py and `ref_from_image` in fdmlx/cli.py. Rendering uses Core Text instead of
// FreeType, so edge anti-aliasing can differ by a grey level here and there; the glyph's size and position are
// the same. Resizing reimplements Pillow's bilinear filter (including its 8-bit fixed-point rounding), so the
// 128 -> 96 step matches the Python version exactly.

import CoreGraphics
import CoreText
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

public let glyphSize = 96

/// An 8-bit greyscale image, row-major, 0 = black.
public struct GrayImage {
    public var width: Int
    public var height: Int
    public var pixels: [UInt8]

    public init(width: Int, height: Int, fill: UInt8) {
        self.width = width
        self.height = height
        pixels = [UInt8](repeating: fill, count: width * height)
    }

    public init(width: Int, height: Int, pixels: [UInt8]) {
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    subscript(x: Int, y: Int) -> UInt8 {
        get { pixels[y * width + x] }
        set { pixels[y * width + x] = newValue }
    }

    /// Bounding box (x0, y0, x1, y1), exclusive end, of the pixels where `isInk` is true; nil if none.
    func bbox(_ isInk: (UInt8) -> Bool) -> (Int, Int, Int, Int)? {
        var x0 = width, y0 = height, x1 = 0, y1 = 0
        for y in 0 ..< height {
            for x in 0 ..< width where isInk(self[x, y]) {
                x0 = min(x0, x); y0 = min(y0, y); x1 = max(x1, x + 1); y1 = max(y1, y + 1)
            }
        }
        return x1 > x0 ? (x0, y0, x1, y1) : nil
    }

    func cropped(_ b: (Int, Int, Int, Int)) -> GrayImage {
        var out = GrayImage(width: b.2 - b.0, height: b.3 - b.1, fill: 0)
        for y in 0 ..< out.height {
            for x in 0 ..< out.width { out[x, y] = self[b.0 + x, b.1 + y] }
        }
        return out
    }

    mutating func paste(_ img: GrayImage, x px: Int, y py: Int) {
        for y in 0 ..< img.height where py + y >= 0 && py + y < height {
            for x in 0 ..< img.width where px + x >= 0 && px + x < width { self[px + x, py + y] = img[x, y] }
        }
    }

    /// Pillow's Image.resize(..., BILINEAR) for mode "L": a triangle filter widened when shrinking, applied
    /// horizontally then vertically, with 8-bit fixed-point coefficients and rounding after each pass.
    public func resized(width w: Int, height h: Int) -> GrayImage {
        var img = self
        if w != width { img = img.resamplePass(outSize: w, horizontal: true) }
        if h != height { img = img.resamplePass(outSize: h, horizontal: false) }
        return img
    }

    private func resamplePass(outSize: Int, horizontal: Bool) -> GrayImage {
        let inSize = horizontal ? width : height
        let precision = 22                                   // PRECISION_BITS = 32 - 8 - 2
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1.0)
        let support = 1.0 * filterScale                      // bilinear support is 1
        var bounds = [(Int, Int)](), kernels = [[Int32]]()
        for o in 0 ..< outSize {
            let center = (Double(o) + 0.5) * scale
            let xmin = max(Int(center - support + 0.5), 0)
            let xmax = min(Int(center + support + 0.5), inSize) - xmin
            var k = (0 ..< xmax).map { x -> Double in
                let d = abs((Double(x + xmin) - center + 0.5) / filterScale)
                return d < 1 ? 1 - d : 0
            }
            let ww = k.reduce(0, +)
            if ww != 0 { k = k.map { $0 / ww } }
            kernels.append(k.map { Int32($0 < 0 ? -0.5 + $0 * Double(1 << precision) : 0.5 + $0 * Double(1 << precision)) })
            bounds.append((xmin, xmax))
        }
        var out = horizontal ? GrayImage(width: outSize, height: height, fill: 0)
                             : GrayImage(width: width, height: outSize, fill: 0)
        let lines = horizontal ? height : width
        for line in 0 ..< lines {
            for o in 0 ..< outSize {
                let (xmin, xmax) = bounds[o]
                var ss = Int64(1 << (precision - 1))
                for x in 0 ..< xmax {
                    let v = horizontal ? self[xmin + x, line] : self[line, xmin + x]
                    ss += Int64(v) * Int64(kernels[o][x])
                }
                let v: UInt8 = ss <= 0 ? 0 : (ss >= Int64(1 << (precision + 8)) ? 255 : UInt8(ss >> Int64(precision)))
                if horizontal { out[o, line] = v } else { out[line, o] = v }
            }
        }
        return out
    }

    /// (96, 96, 3) float32 in [-1, 1], like Resize(bilinear) + ToTensor + Normalize(0.5, 0.5).
    public func modelInput(size: Int = glyphSize) -> [Float] {
        let small = resized(width: size, height: size)
        var out = [Float]()
        out.reserveCapacity(size * size * 3)
        for p in small.pixels {
            let v = Float(p) / 255.0 * 2 - 1
            out.append(v); out.append(v); out.append(v)
        }
        return out
    }
}

public enum GlyphError: Error, CustomStringConvertible {
    case cannotOpenFont(URL)
    case missingGlyphs(URL, String)
    case cannotReadImage(URL)
    case cannotWriteImage(URL)

    public var description: String {
        switch self {
        case .cannotOpenFont(let url): return "cannot open font \(url.path)"
        case .missingGlyphs(let url, let chars): return "\(url.lastPathComponent) has no glyph for: \(chars)"
        case .cannotReadImage(let url): return "cannot read image \(url.path)"
        case .cannotWriteImage(let url): return "cannot write \(url.path)"
        }
    }
}

/// A font file, rendered at 128 px per em (the first face, for collections).
public final class GlyphFont {
    public let url: URL
    let font: CTFont
    let size: Int

    public init(url: URL, size: Int = 128) throws {
        guard let descs = CTFontManagerCreateFontDescriptorsFromURL(url as CFURL) as? [CTFontDescriptor],
              let first = descs.first else { throw GlyphError.cannotOpenFont(url) }
        self.url = url
        self.size = size
        font = CTFontCreateWithFontDescriptor(first, CGFloat(size), nil)
    }

    /// The font's own glyph for a character (no fallback to other fonts), or nil.
    func glyph(_ ch: Character) -> CGGlyph? {
        let units = Array(String(ch).utf16)
        var glyphs = [CGGlyph](repeating: 0, count: units.count)
        guard CTFontGetGlyphsForCharacters(font, units, &glyphs, units.count), glyphs[0] != 0 else { return nil }
        return glyphs[0]
    }

    public func hasGlyph(_ ch: Character) -> Bool { glyph(ch) != nil }

    /// One glyph as a 128 x 128 image, white background: ink box centred, shrunk to fit if larger.
    public func render(_ ch: Character) -> GrayImage {
        let s = size
        let fsize = s
        var out = GrayImage(width: fsize, height: fsize, fill: 255)
        guard var g = glyph(ch) else { return out }
        // white ink on black, on a canvas with room around the em box; the pen sits on whole pixels
        let cw = s * 3
        var buf = [UInt8](repeating: 0, count: cw * cw)
        buf.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(data: raw.baseAddress, width: cw, height: cw, bitsPerComponent: 8,
                                      bytesPerRow: cw, space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return }
            ctx.setShouldAntialias(true)
            ctx.setShouldSmoothFonts(false)
            ctx.setShouldSubpixelPositionFonts(false)
            ctx.setShouldSubpixelQuantizeFonts(false)
            ctx.setFillColor(gray: 1, alpha: 1)
            var pos = CGPoint(x: s, y: s)
            CTFontDrawGlyphs(font, &g, &pos, 1, ctx)
        }
        let canvas = GrayImage(width: cw, height: cw, pixels: buf)
        guard let box = canvas.bbox({ $0 > 0 }) else { return out }
        var ink = canvas.cropped(box)
        if ink.height > fsize {
            ink = ink.resized(width: pyRound(Double(ink.width) * Double(fsize) / Double(ink.height)), height: fsize)
        }
        if ink.width > fsize {
            ink = ink.resized(width: fsize, height: pyRound(Double(ink.height) * Double(fsize) / Double(ink.width)))
        }
        ink.pixels = ink.pixels.map { 255 - $0 }
        out.paste(ink, x: pyRound(Double(fsize - ink.width) / 2), y: pyRound(Double(fsize - ink.height) / 2))
        return out
    }

    /// (count, 96, 96, 3) model input for a string of characters.
    public func glyphs(_ chars: String) throws -> MLXArray {
        let missing = chars.filter { !hasGlyph($0) }
        if !missing.isEmpty { throw GlyphError.missingGlyphs(url, missing) }
        var data = [Float]()
        for ch in chars { data += render(ch).modelInput() }
        return MLXArray(data, [chars.count, glyphSize, glyphSize, 3])
    }
}

/// Python's round(): halves go to the even neighbour.
func pyRound(_ x: Double) -> Int { Int(x.rounded(.toNearestOrEven)) }

/// Read an image file as greyscale with Pillow's luma weights (L = 0.299 R + 0.587 G + 0.114 B).
/// Transparent areas count as white paper.
public func readGrayImage(_ url: URL) throws -> GrayImage {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw GlyphError.cannotReadImage(url) }
    let (w, h) = (img.width, img.height)
    var rgba = [UInt8](repeating: 255, count: w * h * 4)
    rgba.withUnsafeMutableBytes { raw in
        guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    }
    var out = GrayImage(width: w, height: h, fill: 0)
    for i in 0 ..< w * h {
        let (r, g, b) = (UInt32(rgba[i * 4]), UInt32(rgba[i * 4 + 1]), UInt32(rgba[i * 4 + 2]))
        out.pixels[i] = UInt8((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16)
    }
    return out
}

/// A style reference from an image (for example a photo or scan of one handwritten character): dark ink on a
/// light background, contrast stretched, cropped to the ink and centred like a rendered glyph. (1, 96, 96, 3).
public func styleReference(fromImage url: URL) throws -> MLXArray {
    var im = try readGrayImage(url)
    // ImageOps.autocontrast: stretch the darkest..lightest grey to 0..255
    if let lo = im.pixels.min(), let hi = im.pixels.max(), hi > lo {
        let scale = 255.0 / Double(hi - lo), offset = -Double(lo) * scale
        let lut = (0 ..< 256).map { UInt8(min(max(Int(Double($0) * scale + offset), 0), 255)) }
        im.pixels = im.pixels.map { lut[Int($0)] }
    }
    if let box = im.bbox({ $0 < 128 }) { im = im.cropped(box) }
    // rendered glyphs leave a margin: the longer ink side is ~116 of 128 px in the content font the model was
    // trained with (108-124 in other fonts), so scale to that instead of filling the box
    let k = 116.0 / Double(max(im.width, im.height))
    im = im.resized(width: max(1, pyRound(Double(im.width) * k)), height: max(1, pyRound(Double(im.height) * k)))
    var sq = GrayImage(width: 128, height: 128, fill: 255)
    sq.paste(im, x: (128 - im.width) / 2, y: (128 - im.height) / 2)
    return MLXArray(sq.modelInput(), [1, glyphSize, glyphSize, 3])
}

/// Write RGB bytes (width * height * 3) as a PNG.
public func writePNG(rgb: [UInt8], width: Int, height: Int, to url: URL) throws {
    let data = Data(rgb) as CFData
    guard let provider = CGDataProvider(data: data),
          let img = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                            bytesPerRow: width * 3, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw GlyphError.cannotWriteImage(url) }
    CGImageDestinationAddImage(dest, img, nil)
    guard CGImageDestinationFinalize(dest) else { throw GlyphError.cannotWriteImage(url) }
}

extension GlyphImages {
    /// Save glyph i as a PNG.
    public func writePNG(_ i: Int, to url: URL) throws {
        try FontDiffuserMLX.writePNG(rgb: Array(image(i)), width: size, height: size, to: url)
    }

    /// A contact sheet: `cols` glyphs per row in light grey cells, like the Python version's sheet.png.
    public func writeSheet(to url: URL, cols: Int? = nil, cell: Int = 320, pad: Int = 24) throws {
        let cols = cols ?? min(8, count)
        let rows = (count + cols - 1) / cols
        let (W, H) = (cols * cell + 2 * pad, rows * cell + 2 * pad)
        var rgb = [UInt8](repeating: 255, count: W * H * 3)
        // draw in RGBX, then drop the padding byte
        var rgbx = [UInt8](repeating: 255, count: W * H * 4)
        rgbx.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(data: raw.baseAddress, width: W, height: H, bitsPerComponent: 8,
                                      bytesPerRow: W * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return }
            ctx.interpolationQuality = .high
            ctx.setStrokeColor(red: 200 / 255, green: 200 / 255, blue: 200 / 255, alpha: 1)
            ctx.setLineWidth(2)
            for i in 0 ..< count {
                let (r, k) = (i / cols, i % cols)
                let (x0, y0) = (pad + k * cell, pad + r * cell)
                let flippedY = H - y0 - cell                 // Core Graphics counts y from the bottom
                let data = Data(image(i)) as CFData
                if let provider = CGDataProvider(data: data),
                   let img = CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 24,
                                     bytesPerRow: size * 3, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                     bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                     provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) {
                    ctx.draw(img, in: CGRect(x: x0 + 32, y: flippedY + 32, width: cell - 64, height: cell - 64))
                }
                ctx.stroke(CGRect(x: x0, y: flippedY, width: cell, height: cell))
            }
        }
        for i in 0 ..< W * H {
            rgb[i * 3] = rgbx[i * 4]; rgb[i * 3 + 1] = rgbx[i * 4 + 1]; rgb[i * 3 + 2] = rgbx[i * 4 + 2]
        }
        try FontDiffuserMLX.writePNG(rgb: rgb, width: W, height: H, to: url)
    }
}
