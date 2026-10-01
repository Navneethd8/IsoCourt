import { useState, useRef, useEffect, useCallback, useReducer } from 'react'
import { track } from './analytics.js'

const FRAME_INTERVAL_MS = 200

const initialSessionUi = {
    status: 'idle',
    statusMsg: null,
    errorMsg: null,
    lastResult: null,
    onBreak: null,
    cameraError: null,
}

function sessionUiReducer(state, action) {
    switch (action.type) {
        case 'START_CONNECTING':
            return { ...initialSessionUi, status: 'connecting' }
        case 'SET_CAPACITY':
            return { ...state, status: 'capacity', errorMsg: action.errorMsg }
        case 'SET_ERROR':
            return { ...state, status: 'error', errorMsg: action.errorMsg }
        case 'SET_CAMERA_ERROR':
            return { ...state, status: 'error', cameraError: action.cameraError }
        case 'SET_LIVE':
            return { ...state, status: 'live' }
        case 'SET_PAUSED':
            return { ...state, status: 'paused' }
        case 'SET_IDLE':
            return { ...initialSessionUi, status: 'idle' }
        case 'SET_STATUS_MSG':
            return { ...state, statusMsg: action.message }
        case 'SET_BREAK':
            return { ...state, onBreak: action.reason, lastResult: null }
        case 'CLEAR_BREAK':
            return { ...state, onBreak: null }
        case 'SET_LAST_RESULT':
            return { ...state, lastResult: action.result, statusMsg: null, onBreak: null }
        case 'SET_ERROR_MSG':
            return { ...state, errorMsg: action.errorMsg }
        case 'END_SESSION':
            return { ...initialSessionUi, status: 'idle' }
        case 'DISMISS_CAPACITY':
            return { ...state, status: 'idle', errorMsg: null }
        case 'DISMISS_ERROR':
            return { ...initialSessionUi, status: 'idle' }
        default:
            return state
    }
}

function getActiveFullscreenElement() {
    return document.fullscreenElement || document.webkitFullscreenElement || null
}

function beginFrameLoop(video, canvas, ws, intervalRef) {
    if (intervalRef.current) clearInterval(intervalRef.current)
    intervalRef.current = setInterval(() => {
        if (!video || video.readyState < 2) return
        if (!ws || ws.readyState !== WebSocket.OPEN) return
        canvas.width = video.videoWidth || 640
        canvas.height = video.videoHeight || 480
        const ctx = canvas.getContext('2d')
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        canvas.toBlob(blob => {
            if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob)
        }, 'image/jpeg', 0.6)
    }, FRAME_INTERVAL_MS)
}

export function useLiveSession() {
    const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
    const wsBase = apiUrl.replace(/^http/, 'ws')

    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const videoContainerRef = useRef(null)
    const wsRef = useRef(null)
    const streamRef = useRef(null)
    const intervalRef = useRef(null)
    const chatScrollRef = useRef(null)
    const overlayScrollRef = useRef(null)
    const sessionIdRef = useRef(null)
    const busyRef = useRef(false)
    const mountedRef = useRef(true)
    const onBreakRef = useRef(null)
    const liveTrackRef = useRef({
        generation: 0,
        attempted: false,
        started: false,
        coached: false,
        failed: false,
    })

    const failLive = useCallback((reason, stage, statusCode) => {
        const current = liveTrackRef.current
        if (current.failed) return
        current.failed = true
        const params = { reason, stage }
        if (Number.isInteger(statusCode)) params.status_code = statusCode
        track('live_session_failed', params, { beacon: true })
    }, [])

    const noteLiveEnded = useCallback(() => {
        const current = liveTrackRef.current
        if (!current.attempted || current.failed || current.coached) return
        failLive('no_result', current.started ? 'session' : 'create')
    }, [failLive])

    const noteCoaching = useCallback(() => {
        const current = liveTrackRef.current
        if (current.coached) return
        current.coached = true
        track('live_coaching_delivered')
    }, [])

    const noteStarted = useCallback(() => {
        const current = liveTrackRef.current
        if (current.started) return
        current.started = true
        track('live_session_started')
    }, [])

    const [sessionUi, dispatch] = useReducer(sessionUiReducer, initialSessionUi)
    const { status, statusMsg, errorMsg, lastResult, onBreak, cameraError } = sessionUi

    const [sessionId, setSessionId] = useState(null)
    const [chatLog, setChatLog] = useState([])
    const [voiceEnabled, setVoiceEnabled] = useState(true)
    const voiceEnabledRef = useRef(true)
    const [isNativeFullscreen, setIsNativeFullscreen] = useState(false)
    const [isImmersive, setIsImmersive] = useState(false)
    const isExpanded = isNativeFullscreen || isImmersive

    const bestVoiceRef = useRef(null)
    const pickBestVoice = useCallback(() => {
        const voices = window.speechSynthesis?.getVoices() || []
        const en = voices.filter(v => v.lang.startsWith('en'))
        if (!en.length) return
        const ranked = [
            v => /Google UK English Male/i.test(v.name),
            v => /Google UK English Female/i.test(v.name),
            v => /Google US English/i.test(v.name),
            v => /Daniel.*Premium/i.test(v.name),
            v => /Samantha.*Enhanced/i.test(v.name),
            v => /\(Enhanced\)/i.test(v.name) || /\(Premium\)/i.test(v.name),
            v => /Microsoft.*Online/i.test(v.name) && v.lang.startsWith('en'),
            v => !v.localService,
        ]
        for (const test of ranked) {
            const match = en.find(test)
            if (match) { bestVoiceRef.current = match; return }
        }
        bestVoiceRef.current = en[0]
    }, [])

    useEffect(() => {
        pickBestVoice()
        window.speechSynthesis?.addEventListener('voiceschanged', pickBestVoice)
        return () => window.speechSynthesis?.removeEventListener('voiceschanged', pickBestVoice)
    }, [pickBestVoice])

    const speak = useCallback((text) => {
        if (!voiceEnabledRef.current || !window.speechSynthesis) return
        window.speechSynthesis.cancel()
        const utterance = new SpeechSynthesisUtterance(text)
        utterance.rate = 1.05
        utterance.pitch = 1.0
        utterance.volume = 1.0
        if (bestVoiceRef.current) utterance.voice = bestVoiceRef.current
        window.speechSynthesis.speak(utterance)
    }, [])

    const toggleVoice = useCallback(() => {
        setVoiceEnabled(prev => !prev)
    }, [])

    useEffect(() => {
        voiceEnabledRef.current = voiceEnabled
        if (!voiceEnabled) window.speechSynthesis?.cancel()
    }, [voiceEnabled])

    useEffect(() => {
        const el = chatScrollRef.current
        if (!el) return
        el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
    }, [chatLog])

    useEffect(() => {
        const el = overlayScrollRef.current
        if (!el) return
        el.scrollTop = el.scrollHeight
    }, [chatLog])

    const exitExpanded = useCallback(async () => {
        if (isImmersive) {
            setIsImmersive(false)
            document.body.classList.remove('app-live-immersive-active')
        }
        const active = getActiveFullscreenElement()
        if (active === videoContainerRef.current) {
            try {
                if (document.exitFullscreen) await document.exitFullscreen()
                else if (document.webkitExitFullscreen) await document.webkitExitFullscreen()
            } catch {}
        }
    }, [isImmersive])

    useEffect(() => {
        const onFullscreenChange = () => {
            setIsNativeFullscreen(getActiveFullscreenElement() === videoContainerRef.current)
        }
        document.addEventListener('fullscreenchange', onFullscreenChange)
        document.addEventListener('webkitfullscreenchange', onFullscreenChange)
        return () => {
            document.removeEventListener('fullscreenchange', onFullscreenChange)
            document.removeEventListener('webkitfullscreenchange', onFullscreenChange)
            document.body.classList.remove('app-live-immersive-active')
        }
    }, [])

    const toggleFullscreen = useCallback(async () => {
        const el = videoContainerRef.current
        if (!el) return

        if (isImmersive) {
            setIsImmersive(false)
            document.body.classList.remove('app-live-immersive-active')
            return
        }

        if (getActiveFullscreenElement() === el) {
            try {
                if (document.exitFullscreen) await document.exitFullscreen()
                else if (document.webkitExitFullscreen) await document.webkitExitFullscreen()
            } catch {}
            return
        }

        const requestFs = el.requestFullscreen?.bind(el) || el.webkitRequestFullscreen?.bind(el)
        if (requestFs) {
            try {
                await requestFs()
                return
            } catch {}
        }

        setIsImmersive(true)
        document.body.classList.add('app-live-immersive-active')
    }, [isImmersive])

    const stopCamera = useCallback(() => {
        window.speechSynthesis?.cancel()
        if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
        if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null }
        if (videoRef.current) { videoRef.current.srcObject = null }
    }, [])

    const pushChat = useCallback((type, text) => {
        setChatLog(prev => [...prev, { id: Date.now() + Math.random(), type, text, ts: new Date() }])
    }, [])

    const endSession = useCallback(async () => {
        if (busyRef.current) return
        busyRef.current = true
        noteLiveEnded()
        try {
            await exitExpanded()
            stopCamera()
            const sid = sessionIdRef.current
            if (sid) {
                try { await fetch(`${apiUrl}/live/sessions/${sid}`, { method: 'DELETE' }) } catch {}
            }
            sessionIdRef.current = null
            setSessionId(null)
            dispatch({ type: 'END_SESSION' })
        } finally {
            busyRef.current = false
        }
    }, [apiUrl, stopCamera, exitExpanded, noteLiveEnded])

    const pauseSession = useCallback(() => {
        if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
        dispatch({ type: 'SET_PAUSED' })
        pushChat('system', 'Session paused. Take a break.')
    }, [pushChat])

    const resumeSession = useCallback(() => {
        const v = videoRef.current, c = canvasRef.current, w = wsRef.current
        if (v && c && w) beginFrameLoop(v, c, w, intervalRef)
        dispatch({ type: 'SET_LIVE' })
        pushChat('system', 'Resumed. Back on court!')
    }, [pushChat])

    useEffect(() => {
        mountedRef.current = true
        const onHide = () => noteLiveEnded()
        window.addEventListener('pagehide', onHide)
        return () => {
            mountedRef.current = false
            window.removeEventListener('pagehide', onHide)
            noteLiveEnded()
            stopCamera()
        }
    }, [stopCamera, noteLiveEnded])

    useEffect(() => {
        if (!sessionId) return undefined

        const generation = liveTrackRef.current.generation
        const isCurrent = () => liveTrackRef.current.generation === generation
        const ws = new WebSocket(`${wsBase}/live/sessions/${sessionId}/ws`)
        wsRef.current = ws

        ws.onopen = () => {
            if (!isCurrent() || !mountedRef.current || wsRef.current !== ws) {
                ws.close()
                return
            }
            dispatch({ type: 'SET_LIVE' })
            pushChat('system', 'Session started. Point your camera at the court.')
            noteStarted()
            beginFrameLoop(videoRef.current, canvasRef.current, ws, intervalRef)
            busyRef.current = false
        }

        ws.onmessage = (msg) => {
            if (!isCurrent() || !mountedRef.current || wsRef.current !== ws) return
            try {
                const data = JSON.parse(msg.data)
                if (data.event === 'status') {
                    dispatch({ type: 'SET_STATUS_MSG', message: data.message })
                    pushChat('system', data.message)
                } else if (data.event === 'break') {
                    const reason = data.reason || 'game_break'
                    onBreakRef.current = reason
                    dispatch({ type: 'SET_BREAK', reason })
                    pushChat('system', data.message || 'Break detected.')
                } else if (data.event === 'analysis') {
                    if (onBreakRef.current) pushChat('system', 'Play resumed. Analyzing…')
                    onBreakRef.current = null
                    dispatch({ type: 'SET_LAST_RESULT', result: data })
                    const conf = data.confidence != null ? `${(data.confidence * 100).toFixed(0)}%` : ''
                    pushChat('analysis', `${data.label} ${conf}`)
                    noteCoaching()
                } else if (data.event === 'commentary') {
                    if (data.text) {
                        pushChat('coach', data.text)
                        speak(data.text)
                        noteCoaching()
                    }
                } else if (data.event === 'error') {
                    dispatch({ type: 'SET_ERROR_MSG', errorMsg: data.error })
                    pushChat('system', `Error: ${data.error}`)
                    failLive('ws_error', 'session')
                }
            } catch {}
        }

        ws.onerror = () => {
            if (!isCurrent()) return
            if (mountedRef.current && wsRef.current === ws) {
                dispatch({ type: 'SET_ERROR', errorMsg: 'WebSocket connection failed.' })
            }
            failLive('ws_error', 'socket')
            busyRef.current = false
        }

        ws.onclose = () => {
            if (intervalRef.current) clearInterval(intervalRef.current)
            if (isCurrent()) noteLiveEnded()
            busyRef.current = false
        }

        return () => {
            if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
            ws.close()
            if (wsRef.current === ws) wsRef.current = null
        }
    }, [sessionId, wsBase, pushChat, speak, noteStarted, noteCoaching, failLive, noteLiveEnded])

    const startSession = useCallback(async () => {
        if (busyRef.current) return
        busyRef.current = true
        dispatch({ type: 'START_CONNECTING' })
        setChatLog([])

        noteLiveEnded()
        const generation = liveTrackRef.current.generation + 1
        liveTrackRef.current = {
            generation,
            attempted: true,
            started: false,
            coached: false,
            failed: false,
        }
        track('live_start_attempted')

        let sid
        try {
            const res = await fetch(`${apiUrl}/live/sessions`, { method: 'POST' })
            if (res.status === 503) {
                const body = await res.json().catch(() => ({}))
                dispatch({ type: 'SET_CAPACITY', errorMsg: body?.detail?.error || 'Live sessions are at capacity. Try again later.' })
                failLive('capacity', 'create', 503)
                busyRef.current = false
                return
            }
            if (!res.ok) {
                failLive('http_error', 'create', res.status)
                throw new Error(`Server error ${res.status}`)
            }
            const data = await res.json()
            sid = data.session_id
            if (!mountedRef.current) {
                try { await fetch(`${apiUrl}/live/sessions/${sid}`, { method: 'DELETE' }) } catch {}
                failLive('no_result', 'create')
                busyRef.current = false
                return
            }
            sessionIdRef.current = sid
        } catch (e) {
            dispatch({ type: 'SET_ERROR', errorMsg: e.message || 'Failed to start session' })
            failLive('http_error', 'create')
            busyRef.current = false
            return
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
                audio: false,
            })
            streamRef.current = stream
            if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play() }
            if (!mountedRef.current) {
                stream.getTracks().forEach(track => track.stop())
                try { await fetch(`${apiUrl}/live/sessions/${sid}`, { method: 'DELETE' }) } catch {}
                sessionIdRef.current = null
                failLive('no_result', 'camera')
                busyRef.current = false
                return
            }
        } catch {
            dispatch({ type: 'SET_CAMERA_ERROR', cameraError: 'Camera access denied or unavailable.' })
            try { await fetch(`${apiUrl}/live/sessions/${sid}`, { method: 'DELETE' }) } catch {}
            sessionIdRef.current = null
            failLive('camera_denied', 'camera')
            busyRef.current = false
            return
        }

        setSessionId(sid)
    }, [apiUrl, failLive, noteLiveEnded])

    const dismissCapacity = useCallback(() => {
        dispatch({ type: 'DISMISS_CAPACITY' })
    }, [])

    const dismissError = useCallback(() => {
        dispatch({ type: 'DISMISS_ERROR' })
    }, [])

    const overlayMessages = chatLog.slice(-10)

    return {
        cameraError,
        canvasRef,
        chatLog,
        chatScrollRef,
        dismissCapacity,
        dismissError,
        endSession,
        errorMsg,
        isExpanded,
        isImmersive,
        isNativeFullscreen,
        lastResult,
        onBreak,
        overlayMessages,
        overlayScrollRef,
        pauseSession,
        resumeSession,
        startSession,
        status,
        statusMsg,
        toggleFullscreen,
        toggleVoice,
        videoContainerRef,
        videoRef,
        voiceEnabled,
    }
}
