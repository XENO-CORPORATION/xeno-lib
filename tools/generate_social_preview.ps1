Add-Type -AssemblyName System.Drawing
$width = 1280
$height = 640
$bmp = New-Object System.Drawing.Bitmap $width, $height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

$rect = New-Object System.Drawing.Rectangle 0, 0, $width, $height
$bgBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush $rect, ([System.Drawing.Color]::FromArgb(255,8,15,28)), ([System.Drawing.Color]::FromArgb(255,17,37,58)), 20
$g.FillRectangle($bgBrush, $rect)

$orange = [System.Drawing.Color]::FromArgb(255,255,132,63)
$orangeSoft = [System.Drawing.Color]::FromArgb(160,255,132,63)
$teal = [System.Drawing.Color]::FromArgb(255,73,214,197)
$tealSoft = [System.Drawing.Color]::FromArgb(120,73,214,197)
$white = [System.Drawing.Color]::FromArgb(255,244,247,250)
$muted = [System.Drawing.Color]::FromArgb(255,168,182,201)
$line = [System.Drawing.Color]::FromArgb(90,255,255,255)

$accentBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush ([System.Drawing.Rectangle]::new(820, -60, 520, 520)), $orange, $teal, 50
$g.FillEllipse($accentBrush, 860, -80, 520, 520)
$g.FillEllipse((New-Object System.Drawing.SolidBrush $tealSoft), 940, 340, 230, 230)
$g.FillEllipse((New-Object System.Drawing.SolidBrush $orangeSoft), 1030, 110, 120, 120)

$panelBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(46,255,255,255))
$panelPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(68,255,255,255), 2)
$panelRect = New-Object System.Drawing.Rectangle 68, 72, 760, 500
$g.FillRectangle($panelBrush, $panelRect)
$g.DrawRectangle($panelPen, $panelRect)

$fontFamily = 'Segoe UI'
$titleFont = New-Object System.Drawing.Font($fontFamily, 86, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
$subFont = New-Object System.Drawing.Font($fontFamily, 30, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
$bodyFont = New-Object System.Drawing.Font($fontFamily, 22, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
$chipFont = New-Object System.Drawing.Font($fontFamily, 20, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
$smallFont = New-Object System.Drawing.Font($fontFamily, 18, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)

$whiteBrush = New-Object System.Drawing.SolidBrush $white
$mutedBrush = New-Object System.Drawing.SolidBrush $muted
$orangeBrush = New-Object System.Drawing.SolidBrush $orange
$tealBrush = New-Object System.Drawing.SolidBrush $teal

$g.DrawString('xeno-lib', $titleFont, $whiteBrush, 108, 122)
$g.DrawString('Pure Rust multimedia processing library and CLI', $subFont, $mutedBrush, 112, 238)
$g.DrawString('Built to ship image, video, and audio workflows with', $bodyFont, $whiteBrush, 112, 316)
$g.DrawString('benchmark gating, FFmpeg parity tracking, and release-ready CI.', $bodyFont, $whiteBrush, 112, 352)

function Draw-Chip([System.Drawing.Graphics]$graphics, [string]$text, [int]$x, [int]$y, [System.Drawing.Color]$fill, [System.Drawing.Color]$fore, [System.Drawing.Font]$font) {
    $size = $graphics.MeasureString($text, $font)
    $w = [Math]::Ceiling($size.Width) + 30
    $h = 42
    $brush = New-Object System.Drawing.SolidBrush $fill
    $graphics.FillRectangle($brush, $x, $y, $w, $h)
    $graphics.DrawString($text, $font, (New-Object System.Drawing.SolidBrush $fore), $x + 15, $y + 8)
}

Draw-Chip $g 'Images' 112 430 ([System.Drawing.Color]::FromArgb(255,37,72,89)) $white $chipFont
Draw-Chip $g 'Video' 238 430 ([System.Drawing.Color]::FromArgb(255,87,48,38)) $white $chipFont
Draw-Chip $g 'Audio' 350 430 ([System.Drawing.Color]::FromArgb(255,28,87,79)) $white $chipFont
Draw-Chip $g 'CLI + Library' 462 430 ([System.Drawing.Color]::FromArgb(255,66,53,94)) $white $chipFont

$g.DrawString('github.com/XENO-CORPORATION/xeno-lib', $smallFont, $orangeBrush, 112, 510)
$g.DrawLine((New-Object System.Drawing.Pen $line, 3), 112, 286, 330, 286)
$g.DrawLine((New-Object System.Drawing.Pen $line, 3), 112, 498, 530, 498)

$bmp.Save((Resolve-Path 'docs\assets').Path + '\social-preview.png', [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose()
$bmp.Dispose()
