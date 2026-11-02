import { useEffect, useRef } from 'react'
import { Icon } from './AnalyzeIcon'

const ROWS = [
    ['Left_Front', 'Mid_Front', 'Right_Front'],
    ['Left_Mid', 'Mid_Court', 'Right_Mid'],
    ['Left_Back', 'Mid_Back', 'Right_Back'],
]

const ZONE_INDEX = Object.fromEntries(ROWS.flat().map((zone, index) => [zone, index]))

function zoneKey(label) {
    const key = String(label || '')
        .trim()
        .toLowerCase()
        .replace(/[\s-]+/g, '_')
    const match = Object.keys(ZONE_INDEX).find((zone) => zone.toLowerCase() === key)
    return match || null
}

function positionOf(event) {
    const position = event?.metrics?.position
    if (!position) return null
    if (typeof position === 'string') return { label: position, confidence: 1 }
    return {
        label: position.label,
        confidence: typeof position.confidence === 'number' ? position.confidence : 1,
    }
}

export function courtHeatFromTimeline(timeline) {
    const weights = Object.fromEntries(ROWS.flat().map((zone) => [zone, 0]))
    let total = 0
    for (const event of timeline || []) {
        const position = positionOf(event)
        const zone = zoneKey(position?.label)
        if (!zone) continue
        const weight = Math.max(position.confidence, 0.05)
        weights[zone] += weight
        total += weight
    }
    return { weights, total }
}

function pathSamples(points) {
    if (points.length === 0) return []
    const step = 0.04
    const samples = [{ x: points[0].x, y: points[0].y, peak: 48 }]
    let carry = 0
    for (let i = 1; i < points.length; i++) {
        const dx = points[i].x - points[i - 1].x
        const dy = points[i].y - points[i - 1].y
        const dist = Math.hypot(dx, dy)
        if (dist < 0.02) continue
        let walked = step - carry
        while (walked <= dist) {
            const t = walked / dist
            samples.push({
                x: points[i - 1].x + dx * t,
                y: points[i - 1].y + dy * t,
                peak: 48,
            })
            walked += step
        }
        carry = dist - (walked - step)
    }
    const last = points[points.length - 1]
    const end = samples[samples.length - 1]
    if (Math.hypot(last.x - end.x, last.y - end.y) > 0.02) {
        samples.push({ x: last.x, y: last.y, peak: 48 })
    }
    return samples
}

export function movementSpots(timeline) {
    const trail = []
    for (const event of timeline || []) {
        for (const foot of event.feet || []) {
            if (Number.isFinite(foot?.x) && Number.isFinite(foot?.y)) {
                trail.push({ x: foot.x, y: foot.y })
            }
        }
    }
    if (trail.length > 0) return { spots: pathSamples(trail), radiusM: 0.55, tracked: true }

    const { weights, total } = courtHeatFromTimeline(timeline)
    if (total <= 0) return { spots: [], radiusM: 1.15, tracked: false }
    const fallback = []
    for (const [zone, weight] of Object.entries(weights)) {
        if (weight <= 0) continue
        const index = ZONE_INDEX[zone]
        fallback.push({
            x: (index % 3 + 0.5) / 3,
            y: (Math.floor(index / 3) + 0.5) / 3,
            peak: Math.max(8, Math.round((weight / total) * 150)),
        })
    }
    return { spots: fallback, radiusM: 1.15, tracked: false }
}

function prettyZone(zone) {
    return zone.replace(/_/g, ' ').replace('Mid Court', 'Mid court')
}

function jet(t) {
    const u = Math.min(1, Math.max(0, t))
    if (u < 0.25) {
        const k = u / 0.25
        return [0, Math.round(80 + 175 * k), 255]
    }
    if (u < 0.5) {
        const k = (u - 0.25) / 0.25
        return [0, 255, Math.round(255 * (1 - k))]
    }
    if (u < 0.75) {
        const k = (u - 0.5) / 0.25
        return [Math.round(255 * k), 255, 0]
    }
    const k = (u - 0.75) / 0.25
    return [255, Math.round(255 * (1 - k)), 0]
}

function drawCourtHeat(canvas, spots, radiusM) {
    const width = canvas.width
    const height = canvas.height
    const ctx = canvas.getContext('2d')
    const padX = width * 0.02
    const padY = height * 0.02
    const court = { x: padX, y: padY, w: width - padX * 2, h: height - padY * 2 }

    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = '#2f9e4e'
    ctx.fillRect(court.x, court.y, court.w, court.h)

    const heat = document.createElement('canvas')
    heat.width = width
    heat.height = height
    const heatCtx = heat.getContext('2d')
    heatCtx.fillStyle = '#000'
    heatCtx.fillRect(0, 0, width, height)
    heatCtx.globalCompositeOperation = 'lighter'
    const radius = court.w * (radiusM / 6.1)
    for (const spot of spots) {
        const x = court.x + spot.x * court.w
        const y = court.y + spot.y * court.h
        const peak = spot.peak
        const mid = Math.round(peak * 0.22)
        const gradient = heatCtx.createRadialGradient(x, y, 0, x, y, radius)
        gradient.addColorStop(0, `rgb(${peak},${peak},${peak})`)
        gradient.addColorStop(0.32, `rgb(${mid},${mid},${mid})`)
        gradient.addColorStop(1, 'rgb(0,0,0)')
        heatCtx.fillStyle = gradient
        heatCtx.beginPath()
        heatCtx.arc(x, y, radius, 0, Math.PI * 2)
        heatCtx.fill()
    }

    const blurred = document.createElement('canvas')
    blurred.width = width
    blurred.height = height
    const blurredCtx = blurred.getContext('2d', { willReadFrequently: true })
    blurredCtx.filter = `blur(${Math.max(1, Math.round(court.w * (0.12 / 6.1)))}px)`
    blurredCtx.drawImage(heat, 0, 0)
    const density = blurredCtx.getImageData(0, 0, width, height)
    const image = ctx.getImageData(0, 0, width, height)
    let max = 1
    for (let i = 0; i < density.data.length; i += 4) {
        const pixel = i / 4
        const px = pixel % width
        const py = Math.floor(pixel / width)
        const inside = px >= court.x && px <= court.x + court.w && py >= court.y && py <= court.y + court.h
        const value = inside ? density.data[i] : 0
        density.data[i] = value
        if (value > max) max = value
    }
    for (let i = 0; i < density.data.length; i += 4) {
        const t = density.data[i] / max
        if (t < 0.14) continue
        const [r, g, b] = jet(t)
        const alpha = Math.min(0.92, 0.28 + Math.pow(t, 0.8) * 0.7)
        image.data[i] = Math.round(image.data[i] * (1 - alpha) + r * alpha)
        image.data[i + 1] = Math.round(image.data[i + 1] * (1 - alpha) + g * alpha)
        image.data[i + 2] = Math.round(image.data[i + 2] * (1 - alpha) + b * alpha)
        image.data[i + 3] = 255
    }
    ctx.putImageData(image, 0, 0)

    const line = Math.max(1.5, width * 0.0035)
    ctx.strokeStyle = 'rgba(255,255,255,0.96)'
    ctx.lineWidth = line
    ctx.strokeRect(court.x + line / 2, court.y + line / 2, court.w - line, court.h - line)
    const singles = court.w * (0.46 / 6.1)
    const shortY = court.y + court.h * (1.98 / 6.7)
    const longY = court.y + court.h * ((6.7 - 0.76) / 6.7)
    ctx.beginPath()
    ctx.moveTo(court.x + singles, court.y)
    ctx.lineTo(court.x + singles, court.y + court.h)
    ctx.moveTo(court.x + court.w - singles, court.y)
    ctx.lineTo(court.x + court.w - singles, court.y + court.h)
    ctx.moveTo(court.x + singles, shortY)
    ctx.lineTo(court.x + court.w - singles, shortY)
    ctx.moveTo(court.x, longY)
    ctx.lineTo(court.x + court.w, longY)
    ctx.moveTo(court.x + court.w / 2, shortY)
    ctx.lineTo(court.x + court.w / 2, court.y + court.h)
    ctx.stroke()
    ctx.lineWidth = line * 2.6
    ctx.beginPath()
    ctx.moveTo(court.x, court.y)
    ctx.lineTo(court.x + court.w, court.y)
    ctx.stroke()
}

export default function AnalyzeCourtHeatmap({ timeline }) {
    const canvasRef = useRef(null)
    const { spots, radiusM, tracked } = movementSpots(timeline)
    const { weights, total } = courtHeatFromTimeline(timeline)
    const signature = spots.map((spot) => `${spot.x.toFixed(3)},${spot.y.toFixed(3)},${spot.peak}`).join('|')

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas || spots.length === 0) return undefined

        const paint = () => {
            const cssWidth = canvas.clientWidth
            if (cssWidth < 40) return
            const cssHeight = cssWidth * (6.7 / 6.1)
            const dpr = Math.min(window.devicePixelRatio || 1, 2)
            const nextWidth = Math.round(cssWidth * dpr)
            const nextHeight = Math.round(cssHeight * dpr)
            if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
                canvas.width = nextWidth
                canvas.height = nextHeight
            }
            drawCourtHeat(canvas, spots, radiusM)
        }

        paint()
        const observer = new ResizeObserver(paint)
        observer.observe(canvas)
        return () => observer.disconnect()
    }, [signature, radiusM, spots])

    if (spots.length === 0) return null

    const ranked = Object.entries(weights).sort((a, b) => b[1] - a[1])
    const hottest = ranked[0]?.[0]
    const hottestShare = total > 0 && hottest ? ranked[0][1] / total : 0
    const summary = tracked
        ? 'Court heatmap of where the player moved during the rally.'
        : `Court heatmap. Most shots from ${prettyZone(hottest).toLowerCase()}, ${Math.round(hottestShare * 100)} percent.`

    return (
        <div className="app-result-panel">
            <div className="flex items-start justify-between gap-3 mb-3">
                <div>
                    <span className="text-xs text-[var(--text-subtle)] flex items-center gap-1.5 mb-1">
                        <Icon name="location_on" size={13} />
                        On-court heatmap
                    </span>
                    <div className="text-sm font-semibold text-[var(--text)]">
                        {tracked ? 'Where they moved' : `Most shots from ${prettyZone(hottest).toLowerCase()}`}
                    </div>
                </div>
                {!tracked && (
                    <div className="text-right">
                        <span className="text-xs font-mono text-[var(--text-subtle)] block mb-1">Share</span>
                        <div className="text-sm font-semibold tabular-nums">{Math.round(hottestShare * 100)}%</div>
                    </div>
                )}
            </div>
            <canvas
                ref={canvasRef}
                className="w-full max-w-[440px] mx-auto block rounded-md"
                style={{ aspectRatio: '6.1 / 6.7' }}
                role="img"
                aria-label={summary}
            />
            <p className="text-[11px] text-[var(--text-subtle)] text-center mt-2">
                Top is the front, at the net. Bottom is the back.
            </p>
        </div>
    )
}
