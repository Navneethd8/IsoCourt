import { useState, useCallback, useRef, useEffect, useReducer } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import { track } from './analytics.js'

export const loadingSteps = [
    { icon: 'movie_filter', label: 'Splitting clip into frames' },
    { icon: 'directions_run', label: 'Tracing poses' },
    { icon: 'query_stats', label: 'Analyzing strokes' },
    { icon: 'rate_review', label: 'Generating feedback' },
]

const stepWeights = [0.10, 0.35, 0.45, 0.10]

export const formatSeconds = (seconds) => {
    const minutes = Math.floor(seconds / 60).toString().padStart(2, '0')
    const remainder = (seconds % 60).toString().padStart(2, '0')
    return `${minutes}:${remainder}`
}

export const timestampToSeconds = (timestamp) => {
    const parts = String(timestamp || '').split(':')
    if (parts.length === 2) return parseInt(parts[0], 10) * 60 + parseInt(parts[1], 10)
    if (parts.length === 1) return parseInt(parts[0], 10)
    return 0
}

export const getQualityColor = (quality) => {
    const value = String(quality).toLowerCase()
    if (value.includes('elite') || value.includes('expert') || value.includes('advanced')) return 'text-brand'
    if (value.includes('proficient')) return 'text-cyan-400'
    if (value.includes('competent')) return 'text-amber-400'
    if (value.includes('developing') || value.includes('emerging')) return 'text-orange-400'
    return 'text-rose-400'
}

export const getQualityBarColor = (quality) => {
    const value = String(quality).toLowerCase()
    if (value.includes('elite') || value.includes('expert')) return 'bg-brand'
    if (value.includes('advanced')) return 'bg-brand-dark'
    if (value.includes('proficient')) return 'bg-cyan-500'
    if (value.includes('competent')) return 'bg-amber-500'
    if (value.includes('developing') || value.includes('emerging')) return 'bg-orange-500'
    return 'bg-rose-500'
}

export const resolveTimeline = (summary, streamed) => {
    const streamedRows = Array.isArray(streamed) ? streamed : []
    const base = Array.isArray(summary?.timeline) && summary.timeline.length > 0
        ? summary.timeline
        : streamedRows
    const poseByWindow = new Map()
    const poseByTimestamp = new Map()

    for (const row of streamedRows) {
        if (!row?.pose_image) continue
        if (typeof row.window === 'number') poseByWindow.set(row.window, row.pose_image)
        if (row.timestamp) poseByTimestamp.set(row.timestamp, row.pose_image)
    }

    return base.map((event, index) => {
        const { event: _event, window, ...rest } = event
        if (!rest.pose_image) {
            rest.pose_image =
                (typeof window === 'number' && poseByWindow.get(window)) ||
                (rest.timestamp && poseByTimestamp.get(rest.timestamp)) ||
                streamedRows[index]?.pose_image
        }
        return rest
    })
}

export const timelineHasPoseFrames = (timeline) =>
    Array.isArray(timeline) && timeline.some((event) => event?.pose_image)

const initialAnalysisState = { loading: false, loadingStep: -1, capacityError: null, queueAhead: null }
const analysisStateReducer = (state, action) => ({ ...state, [action.type]: action.value })

export function useAnalyzeController() {
    const [file, setFile] = useState(null)
    const [result, setResult] = useState(null)
    const [preview, setPreview] = useState(null)
    const [analysisState, dispatchAnalysisState] = useReducer(analysisStateReducer, initialAnalysisState)
    const { loading, loadingStep, capacityError, queueAhead } = analysisState
    const setLoading = (value) => dispatchAnalysisState({ type: 'loading', value })
    const setLoadingStep = (value) => dispatchAnalysisState({ type: 'loadingStep', value })
    const setCapacityError = (value) => dispatchAnalysisState({ type: 'capacityError', value })
    const setQueueAhead = (value) => dispatchAnalysisState({ type: 'queueAhead', value })
    const videoRef = useRef(null)
    const loadingTimers = useRef([])
    const [lightboxEvent, setLightboxEvent] = useState(null)
    const [frameTip, setFrameTip] = useState(null)
    const [frameTipLoading, setFrameTipLoading] = useState(false)
    const [streamingTimeline, setStreamingTimeline] = useState([])
    const frameTipCacheRef = useRef({})
    const retryTimerRef = useRef(null)
    const frameTipAbortRef = useRef(null)
    const attemptRef = useRef(null)

    // Mobile detection
    const [isMobile, setIsMobile] = useState(window.innerWidth < 768)
    useEffect(() => {
        const handleResize = () => setIsMobile(window.innerWidth < 768)
        window.addEventListener('resize', handleResize)
        return () => window.removeEventListener('resize', handleResize)
    }, [])

    // Camera recording state (Laptop)
    const [inputMode, setInputMode] = useState('upload') // 'upload' | 'record'

    const beginAnalyze = (stage) => {
        attemptRef.current = { stage, settled: false }
        track('analyze_started', { source: inputMode === 'record' ? 'record' : 'upload' })
    }

    const settleAnalyze = (outcome, params = {}) => {
        const attempt = attemptRef.current
        if (!attempt || attempt.settled) return
        attempt.settled = true
        track(outcome, { stage: attempt.stage, ...params }, { beacon: outcome === 'analyze_failed' })
    }

    useEffect(() => {
        const onHide = () => {
            const attempt = attemptRef.current
            if (!attempt || attempt.settled) return
            attempt.settled = true
            track('analyze_failed', { stage: attempt.stage, reason: 'no_result' }, { beacon: true })
        }
        window.addEventListener('pagehide', onHide)
        return () => window.removeEventListener('pagehide', onHide)
    }, [])
    const [isRecording, setIsRecording] = useState(false)
    const [cameraError, setCameraError] = useState(null)
    const [recordingSeconds, setRecordingSeconds] = useState(0)
    const cameraPreviewRef = useRef(null)
    const mediaRecorderRef = useRef(null)
    const cameraStreamRef = useRef(null)
    const recordedChunksRef = useRef([])
    const recordingTimerRef = useRef(null)
    const [isFullScreen, setIsFullScreen] = useState(false)

    // Native Camera Reference
    const nativeVideoInputRef = useRef(null)

    const startLoadingSteps = () => {
        // Get video duration from the element
        const duration = videoRef.current?.duration || 3
        // Rough estimate: ~2s processing per second of video, minimum 4s total
        const estimatedTime = Math.max(4, duration * 2) * 1000

        setLoadingStep(0)

        let elapsed = 0
        loadingTimers.current = []
        for (let i = 1; i < loadingSteps.length; i++) {
            elapsed += estimatedTime * stepWeights[i - 1]
            const timer = setTimeout(() => setLoadingStep(i), elapsed)
            loadingTimers.current.push(timer)
        }
    }

    const stopLoadingSteps = () => {
        loadingTimers.current.forEach(clearTimeout)
        loadingTimers.current = []
        setLoadingStep(-1)
    }

    const onDrop = useCallback((acceptedFiles) => {
        setFile(acceptedFiles[0] || null)
    }, [])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'video/*': [] }
    })

    const handleNativeVideoSelect = (e) => {
        const next = e.target.files?.[0]
        if (next) setFile(next)
    }

    useEffect(() => {
        if (!file) {
            setPreview(null)
            return undefined
        }
        const url = URL.createObjectURL(file)
        setPreview(url)
        return () => {
            URL.revokeObjectURL(url)
        }
    }, [file])

    // ── Laptop Camera helpers ──────────────────────────────────────────
    const openCamera = async (isCancelled = () => false) => {
        if (!isCancelled()) setCameraError(null)
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
            if (isCancelled()) {
                stream.getTracks().forEach((track) => track.stop())
                return
            }
            cameraStreamRef.current = stream
            if (cameraPreviewRef.current) {
                cameraPreviewRef.current.srcObject = stream
                cameraPreviewRef.current.play().catch(e => console.error("Error playing camera stream:", e))
            }
        } catch (err) {
            if (isCancelled()) return
            if (err.name === 'NotAllowedError') {
                setCameraError('Camera permission denied. Please allow camera access and try again.')
            } else if (err.name === 'NotFoundError') {
                setCameraError('No camera found on this device.')
            } else {
                setCameraError('Could not access camera: ' + err.message)
            }
        }
    }

    const closeCamera = () => {
        if (cameraStreamRef.current) {
            cameraStreamRef.current.getTracks().forEach(t => t.stop())
            cameraStreamRef.current = null
        }
        if (cameraPreviewRef.current) {
            cameraPreviewRef.current.srcObject = null
        }
        if (recordingTimerRef.current) clearInterval(recordingTimerRef.current)
        setIsRecording(false)
        setRecordingSeconds(0)
    }

    const startRecording = () => {
        if (!cameraStreamRef.current) return
        recordedChunksRef.current = []
        const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
            ? 'video/webm;codecs=vp9'
            : MediaRecorder.isTypeSupported('video/webm')
                ? 'video/webm'
                : 'video/mp4'
        const recorder = new MediaRecorder(cameraStreamRef.current, { mimeType })
        recorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) recordedChunksRef.current.push(e.data)
        }
        recorder.onstop = () => {
            const ext = mimeType.includes('mp4') ? 'mp4' : 'webm'
            const blob = new Blob(recordedChunksRef.current, { type: mimeType })
            const recorded = new File([blob], `recording.${ext}`, { type: mimeType })
            setFile(recorded)
            closeCamera()
        }
        mediaRecorderRef.current = recorder
        recorder.start(250) // collect chunks every 250 ms
        setIsRecording(true)
        setRecordingSeconds(0)
        recordingTimerRef.current = setInterval(() => setRecordingSeconds(s => s + 1), 1000)
    }

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop()
        }
        if (recordingTimerRef.current) clearInterval(recordingTimerRef.current)
        setIsRecording(false)
    }

    const switchMode = (mode) => {
        if (mode === inputMode) return
        if (inputMode === 'record') {
            stopRecording()
            closeCamera()
        }
        setFile(null)
        setResult(null)
        setCameraError(null)
        setInputMode(mode)
    }

    useEffect(() => {
        let cancelled = false
        if (isMobile || inputMode !== 'record') return undefined

        ;(async () => {
            setCameraError(null)
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
                if (cancelled) {
                    stream.getTracks().forEach((track) => track.stop())
                    return
                }
                cameraStreamRef.current = stream
                if (cameraPreviewRef.current) {
                    cameraPreviewRef.current.srcObject = stream
                    cameraPreviewRef.current.play().catch((e) => console.error('Error playing camera stream:', e))
                }
            } catch (err) {
                if (cancelled) return
                if (err.name === 'NotAllowedError') {
                    setCameraError('Camera permission denied. Please allow camera access and try again.')
                } else if (err.name === 'NotFoundError') {
                    setCameraError('No camera found on this device.')
                } else {
                    setCameraError('Could not access camera: ' + err.message)
                }
            }
        })()

        return () => {
            cancelled = true
            stopRecording()
            closeCamera()
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [inputMode, isMobile])

    useEffect(() => {
        if (isMobile) {
            closeCamera()
        }
        return () => {
            closeCamera()
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isMobile])

    const toggleFullScreen = () => {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch((err) => {
                console.error(`Error attempting to enable fullscreen: ${err.message}`);
            });
            setIsFullScreen(true)
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
                setIsFullScreen(false)
            }
        }
    }

    useEffect(() => {
        const handleFullscreenChange = () => {
            setIsFullScreen(!!document.fullscreenElement)
        }
        document.addEventListener('fullscreenchange', handleFullscreenChange)
        return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
    }, [])

    const closeLightbox = () => {
        frameTipAbortRef.current?.abort()
        frameTipAbortRef.current = null
        setLightboxEvent(null)
        setFrameTip(null)
        setFrameTipLoading(false)
    }

    useEffect(() => {
        const onKey = (e) => { if (e.key === 'Escape') closeLightbox() }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    }, [])

    useEffect(() => {
        return () => {
            frameTipAbortRef.current?.abort()
            if (retryTimerRef.current) clearTimeout(retryTimerRef.current)
            loadingTimers.current.forEach(clearTimeout)
        }
    }, [])

    const handleSubmit = async () => {
        if (loading || !file) return

        setLoading(true)
        setCapacityError(null)
        startLoadingSteps()
        beginAnalyze('submit')
        const formData = new FormData()
        formData.append('file', file)

        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
            const response = await axios.post(`${apiUrl}/analyze`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            })

            if (response.data.validation_failed) {
                const isOverDuration = response.data.over_duration_limit || false
                setResult({
                    validation_error: true,
                    error_message: response.data.error,
                    over_duration_limit: isOverDuration,
                    validation_details: response.data.validation_details
                })
                settleAnalyze('analyze_failed', { reason: isOverDuration ? 'too_long' : 'validation' })
            } else {
                setResult(response.data)
                settleAnalyze('analyze_completed')
            }
        } catch (error) {
            console.error("Error uploading file:", error)

            // 503: server at capacity
            if (error.response?.status === 503) {
                const retryAfter = error.response?.data?.detail?.retry_after || 30
                setCapacityError(retryAfter)
                if (retryTimerRef.current) clearTimeout(retryTimerRef.current)
                retryTimerRef.current = setTimeout(() => setCapacityError(null), retryAfter * 1000)
                settleAnalyze('analyze_failed', { reason: 'capacity', status_code: 503 })
            } else if (error.response?.data?.validation_failed) {
                const isOverDuration = error.response.data.over_duration_limit || false
                setResult({
                    validation_error: true,
                    error_message: error.response.data.error,
                    over_duration_limit: isOverDuration,
                    validation_details: error.response.data.validation_details
                })
                settleAnalyze('analyze_failed', { reason: isOverDuration ? 'too_long' : 'validation' })
            } else {
                const errorMessage = error.response?.data?.detail || error.message || "Error analyzing video"
                const statusCode = error.response?.status
                settleAnalyze('analyze_failed', {
                    reason: 'http_error',
                    ...(Number.isInteger(statusCode) ? { status_code: statusCode } : {}),
                })
                alert(`Analysis failed: ${errorMessage}`)
            }
        } finally {
            if (attemptRef.current && !attemptRef.current.settled) {
                settleAnalyze('analyze_failed', { reason: 'no_result' })
            }
            setLoading(false)
            stopLoadingSteps()
        }
    }

    // Stream-based analysis: POST /clips/jobs then GET /clips/jobs/{id}/stream (queued SSE + progress).
    const handleStreamAnalysis = async () => {
        if (loading || !file) return

        setLoading(true)
        setCapacityError(null)
        setQueueAhead(null)
        startLoadingSteps()

        const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
        const formData = new FormData()
        formData.append('file', file)
        beginAnalyze('stream')

        const readSseStream = async (response) => {
            const reader = response.body.getReader()
            const decoder = new TextDecoder()
            let buffer = ''
            const streamedTimeline = []

            const pushStreamedEvent = (parsed) => {
                streamedTimeline.push(parsed)
                setStreamingTimeline((prev) => [...prev, parsed])
            }

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split('\n\n')
                buffer = lines.pop() || ''

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue
                    let parsed
                    try { parsed = JSON.parse(line.slice(6)) } catch { continue }

                    if (parsed.event === 'queue') {
                        const ahead = typeof parsed.ahead === 'number' ? parsed.ahead : null
                        setQueueAhead(ahead)
                    } else if (parsed.event === 'progress') {
                        pushStreamedEvent(parsed)
                    } else if (parsed.event === 'done') {
                        setQueueAhead(null)
                        const summary = parsed.summary || {}
                        const timeline = resolveTimeline(summary, streamedTimeline)
                        setResult({ ...summary, timeline })
                        setStreamingTimeline([])
                        settleAnalyze('analyze_completed')
                    } else if (parsed.event === 'error') {
                        setQueueAhead(null)
                        const isOverDuration = parsed.over_duration_limit || false
                        setResult({
                            validation_error: true,
                            error_message: parsed.error,
                            over_duration_limit: isOverDuration,
                        })
                        settleAnalyze('analyze_failed', { reason: isOverDuration ? 'too_long' : 'stream_dropped' })
                    }
                }
            }
        }

        try {
            const jobRes = await fetch(`${apiUrl}/clips/jobs`, {
                method: 'POST',
                body: formData,
            })

            if (jobRes.status === 503) {
                const body = await jobRes.json().catch(() => ({}))
                const d = body?.detail
                const retryAfter = typeof d === 'object' && d != null && d.retry_after != null ? d.retry_after : 60
                setCapacityError(retryAfter)
                if (retryTimerRef.current) clearTimeout(retryTimerRef.current)
                retryTimerRef.current = setTimeout(() => setCapacityError(null), retryAfter * 1000)
                settleAnalyze('analyze_failed', { reason: 'capacity', status_code: 503 })
                return
            }

            if (!jobRes.ok) {
                const errText = await jobRes.text().catch(() => '')
                settleAnalyze('analyze_failed', { reason: 'http_error', status_code: jobRes.status })
                alert(`Could not start analysis (${jobRes.status}): ${errText.slice(0, 120)}`)
                return
            }

            const jobJson = await jobRes.json()
            const jobId = jobJson.job_id
            if (!jobId) {
                settleAnalyze('analyze_failed', { reason: 'no_result' })
                alert('Invalid response from server (missing job_id).')
                return
            }

            const streamRes = await fetch(`${apiUrl}/clips/jobs/${jobId}/stream`)
            if (streamRes.status === 404) {
                setQueueAhead(null)
                settleAnalyze('analyze_failed', { reason: 'stream_dropped', status_code: 404 })
                alert('Analysis job not found. Please try uploading again.')
                return
            }
            if (!streamRes.ok) {
                setQueueAhead(null)
                const t = await streamRes.text().catch(() => '')
                settleAnalyze('analyze_failed', { reason: 'http_error', status_code: streamRes.status })
                alert(`Stream failed (${streamRes.status}): ${t.slice(0, 120)}`)
                return
            }

            await readSseStream(streamRes)
        } catch (err) {
            console.error('Stream error:', err)
            setQueueAhead(null)
            settleAnalyze('analyze_failed', { reason: 'stream_dropped' })
        } finally {
            if (attemptRef.current && !attemptRef.current.settled) {
                settleAnalyze('analyze_failed', { reason: 'no_result' })
            }
            setLoading(false)
            stopLoadingSteps()
            setQueueAhead(null)
        }
    }

    const handleTimelineClick = (timestamp) => {
        if (!videoRef.current) return
        const seconds = timestampToSeconds(timestamp)
        videoRef.current.currentTime = seconds
        videoRef.current.play()
        videoRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }

    /** Seek clip, open pose lightbox, and request its coach tip. */
    const openFrameAnalysis = async (event) => {
        if (!event?.pose_image) return
        handleTimelineClick(event.timestamp)
        frameTipAbortRef.current?.abort()
        const controller = new AbortController()
        frameTipAbortRef.current = controller
        setLightboxEvent(event)
        const metrics = event.metrics || {}
        const cacheKey = `${event.label}|${metrics.subtype?.label || metrics.subtype || ''}|${metrics.technique?.label || metrics.technique || ''}|${metrics.placement?.label || metrics.placement || ''}|${metrics.position?.label || metrics.position || ''}|${metrics.intent?.label || metrics.intent || ''}|${metrics.quality || ''}`
        const cachedTip = frameTipCacheRef.current[cacheKey]
        if (cachedTip) {
            setFrameTip(cachedTip)
            return
        }

        setFrameTip(null)
        setFrameTipLoading(true)
        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
            const response = await fetch(`${apiUrl}/frame-tip`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: controller.signal,
                body: JSON.stringify({
                    label: event.label,
                    subtype: metrics.subtype?.label || metrics.subtype || 'None',
                    technique: metrics.technique?.label || metrics.technique || 'Unknown',
                    placement: metrics.placement?.label || metrics.placement || 'Unknown',
                    position: metrics.position?.label || metrics.position || 'Unknown',
                    intent: metrics.intent?.label || metrics.intent || 'None',
                    quality: metrics.quality || 'Developing',
                    confidence: event.confidence || 0,
                }),
            })
            if (!response.ok) throw new Error(`Frame tip request failed (${response.status})`)
            const data = await response.json()
            if (!controller.signal.aborted) {
                frameTipCacheRef.current[cacheKey] = data.tip
                setFrameTip(data.tip)
            }
        } catch (error) {
            if (!controller.signal.aborted) setFrameTip(null)
        } finally {
            setFrameTipLoading(false)
        }
    }

    const displayTimeline = result?.timeline?.length
        ? result.timeline
        : streamingTimeline.length > 0
            ? resolveTimeline(null, streamingTimeline)
            : []

    return { file, setFile, result, setResult, loading, setLoading, preview, setPreview, loadingStep, setLoadingStep, capacityError, setCapacityError, queueAhead, setQueueAhead, videoRef, lightboxEvent, setLightboxEvent, closeLightbox, frameTip, frameTipLoading, streamingTimeline, isMobile, inputMode, isRecording, cameraError, recordingSeconds, cameraPreviewRef, cameraStreamRef, nativeVideoInputRef, isFullScreen, getRootProps, getInputProps, isDragActive, switchMode, handleNativeVideoSelect, openCamera, toggleFullScreen, startRecording, stopRecording, handleStreamAnalysis, handleSubmit, formatSeconds, loadingSteps, displayTimeline, getQualityColor, getQualityBarColor, timelineHasPoseFrames, handleTimelineClick, openFrameAnalysis }
}
