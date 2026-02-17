import { useState, useCallback, useRef } from 'react'
import Logo from './components/Logo';
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import { motion } from 'framer-motion'

function Icon({ name, size = 20, className = '' }) {
    return (
        <span
            className={`material-symbols-outlined ${className}`}
            style={{ fontSize: size }}
        >
            {name}
        </span>
    )
}

export default function App() {
    const [file, setFile] = useState(null)
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [preview, setPreview] = useState(null)
    const [loadingStep, setLoadingStep] = useState(-1)
    const videoRef = useRef(null)
    const loadingTimers = useRef([])

    const loadingSteps = [
        { icon: 'movie_filter', label: 'Splitting clip into frames' },
        { icon: 'directions_run', label: 'Tracing poses' },
        { icon: 'query_stats', label: 'Analyzing strokes' },
        { icon: 'rate_review', label: 'Generating feedback' },
    ]

    // Weighted distribution: frame splitting is fast, pose tracing and stroke analysis are the bulk
    const stepWeights = [0.10, 0.35, 0.45, 0.10]

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
        const file = acceptedFiles[0]
        setFile(file)
        setPreview(URL.createObjectURL(file))
    }, [])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'video/*': [] }
    })

    const handleSubmit = async () => {
        if (!file) return

        setLoading(true)
        startLoadingSteps()
        const formData = new FormData()
        formData.append('file', file)

        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
            const response = await axios.post(`${apiUrl}/analyze`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            })

            // Check if validation failed
            if (response.data.validation_failed) {
                setResult({
                    validation_error: true,
                    error_message: response.data.error,
                    validation_details: response.data.validation_details
                })
            } else {
                setResult(response.data)
            }
        } catch (error) {
            console.error("Error uploading file:", error)

            // Check if it's a validation error from the backend
            if (error.response?.data?.validation_failed) {
                setResult({
                    validation_error: true,
                    error_message: error.response.data.error,
                    validation_details: error.response.data.validation_details
                })
            } else {
                const errorMessage = error.response?.data?.detail || error.message || "Error analyzing video"
                alert(`Analysis failed: ${errorMessage}`)
            }
        } finally {
            setLoading(false)
            stopLoadingSteps()
        }
    }

    const handleTimelineClick = (timestamp) => {
        if (!videoRef.current) return

        const parts = timestamp.split(':')
        let seconds = 0
        if (parts.length === 2) {
            seconds = parseInt(parts[0]) * 60 + parseInt(parts[1])
        } else if (parts.length === 1) {
            seconds = parseInt(parts[0])
        }

        videoRef.current.currentTime = seconds
        videoRef.current.play()
        videoRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }

    const getQualityColor = (quality) => {
        // Updated for 10-point scale
        const q = String(quality).toLowerCase()
        if (q.includes('elite') || q.includes('expert')) return 'text-emerald-400'
        if (q.includes('advanced')) return 'text-emerald-500' // New cyan/emerald
        if (q.includes('proficient')) return 'text-cyan-400'
        if (q.includes('competent')) return 'text-amber-400'
        if (q.includes('developing') || q.includes('emerging')) return 'text-orange-400'
        return 'text-rose-400'
    }

    const getQualityBarColor = (quality) => {
        // Updated for 10-point scale
        const q = String(quality).toLowerCase()
        if (q.includes('elite') || q.includes('expert')) return 'bg-emerald-500'
        if (q.includes('advanced')) return 'bg-emerald-600'
        if (q.includes('proficient')) return 'bg-cyan-500'
        if (q.includes('competent')) return 'bg-amber-500'
        if (q.includes('developing') || q.includes('emerging')) return 'bg-orange-500'
        return 'bg-rose-500'
    }

    return (
        <div className="min-h-screen w-screen bg-neutral-950 text-neutral-100 p-6 md:p-10">
            <div className="max-w-2xl mx-auto">

                <header className="flex items-center gap-2 mb-8">
                    <Logo size={24} className="text-emerald-500" />
                    <h1 className="text-xl font-semibold tracking-tight">
                        Bad<span className="text-emerald-500">Coach</span>
                    </h1>
                </header>

                {/* Upload */}
                <section className="bg-neutral-900 border border-neutral-800 rounded-lg p-6 mb-6">
                    <h2 className="text-sm font-medium text-neutral-400 mb-4 flex items-center gap-2">
                        <Icon name="upload" size={18} />
                        Upload Clip
                    </h2>

                    <div
                        {...getRootProps()}
                        className={`
                            border border-dashed rounded-md min-h-[220px] flex flex-col items-center justify-center cursor-pointer transition-colors
                            ${isDragActive
                                ? 'border-emerald-500 bg-emerald-500/5'
                                : 'border-neutral-700 hover:border-neutral-500'
                            }
                        `}
                    >
                        <input {...getInputProps()} />
                        {preview ? (
                            <div className="w-full rounded-md overflow-hidden">
                                <video
                                    ref={videoRef}
                                    src={preview}
                                    className="w-full max-h-[300px] object-contain bg-black"
                                    controls
                                />
                            </div>
                        ) : (
                            <div className="text-center text-neutral-500 p-6">
                                <Icon name="video_file" size={36} className="block mx-auto mb-3 text-neutral-600" />
                                <p className="text-sm font-medium text-neutral-300">Drag & drop video here</p>
                                <p className="text-xs mt-1 text-neutral-500">or click to select file</p>
                            </div>
                        )}
                    </div>

                    <button
                        onClick={handleSubmit}
                        disabled={!file || loading}
                        className="w-full mt-4 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium py-2.5 rounded-md transition-colors"
                    >
                        {loadingStep >= 0 ? (
                            <span className="flex items-center justify-center gap-2">
                                <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                Analyzing...
                            </span>
                        ) : 'Analyze Stroke'}
                    </button>
                </section>

                {/* Results */}
                <section className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
                    <h2 className="text-sm font-medium text-neutral-400 mb-4 flex items-center gap-2">
                        <Icon name="analytics" size={18} />
                        Analysis Results
                    </h2>

                    {loadingStep >= 0 ? (
                        <div className="py-12 px-4">
                            <div className="space-y-4">
                                {loadingSteps.map((step, idx) => {
                                    const isActive = idx === loadingStep
                                    const isDone = idx < loadingStep
                                    return (
                                        <div
                                            key={idx}
                                            className={`flex items-center gap-3 py-2 px-3 rounded-md transition-all duration-300 ${isActive ? 'bg-neutral-800' : ''
                                                }`}
                                        >
                                            <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-300 ${isDone ? 'bg-emerald-600' : isActive ? 'bg-neutral-700' : 'bg-neutral-800'
                                                }`}>
                                                {isDone ? (
                                                    <Icon name="check" size={14} className="text-white" />
                                                ) : isActive ? (
                                                    <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
                                                ) : (
                                                    <span className="w-1.5 h-1.5 bg-neutral-600 rounded-full" />
                                                )}
                                            </div>
                                            <Icon
                                                name={step.icon}
                                                size={18}
                                                className={`transition-colors duration-300 ${isDone ? 'text-emerald-500' : isActive ? 'text-neutral-200' : 'text-neutral-600'
                                                    }`}
                                            />
                                            <span className={`text-sm transition-colors duration-300 ${isDone ? 'text-neutral-400' : isActive ? 'text-neutral-200 font-medium' : 'text-neutral-600'
                                                }`}>
                                                {step.label}{isActive ? '...' : ''}
                                            </span>
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    ) : result?.validation_error ? (
                        <div className="py-8 px-4">
                            <div className="bg-red-950/30 border border-red-900/50 rounded-lg p-6">
                                <div className="flex items-start gap-4">
                                    <div className="flex-shrink-0">
                                        <Icon name="error" size={32} className="text-red-500" />
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="text-lg font-semibold text-red-400 mb-2">Not a Badminton Video</h3>
                                        <p className="text-sm text-neutral-300 mb-4">
                                            {result.error_message}
                                        </p>

                                        {result.validation_details && (
                                            <div className="mt-4 p-3 bg-neutral-950/50 rounded border border-neutral-800">
                                                <span className="text-xs text-neutral-500 block mb-2">Detection Details</span>
                                                <div className="space-y-1.5 text-xs">
                                                    {result.validation_details.pose_confidence !== undefined && (
                                                        <div className="flex justify-between">
                                                            <span className="text-neutral-400">Pose Detection Score:</span>
                                                            <span className={result.validation_details.pose_confidence > 0.3 ? 'text-green-400' : 'text-red-400'}>
                                                                {(result.validation_details.pose_confidence * 100).toFixed(1)}%
                                                            </span>
                                                        </div>
                                                    )}
                                                    {result.validation_details.model_confidence !== undefined && (
                                                        <div className="flex justify-between">
                                                            <span className="text-neutral-400">Model Confidence:</span>
                                                            <span className={result.validation_details.model_confidence > 0.5 ? 'text-green-400' : 'text-red-400'}>
                                                                {(result.validation_details.model_confidence * 100).toFixed(1)}%
                                                            </span>
                                                        </div>
                                                    )}
                                                    {result.validation_details.overhead_score !== undefined && (
                                                        <div className="flex justify-between">
                                                            <span className="text-neutral-400">Overhead Motion:</span>
                                                            <span className={result.validation_details.overhead_score > 0.3 ? 'text-green-400' : 'text-red-400'}>
                                                                {(result.validation_details.overhead_score * 100).toFixed(1)}%
                                                            </span>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        )}

                                        <button
                                            onClick={() => { setResult(null); setFile(null); setPreview(null); }}
                                            className="mt-4 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-white text-sm rounded transition-colors"
                                        >
                                            Try Another Video
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ) : result ? (
                        <div className="space-y-4">
                            <div className="space-y-3">
                                {/* Performance & Tactical Analysis */}
                                <div className="p-4 bg-neutral-950 rounded-md border border-neutral-800">
                                    <div className="flex justify-between items-start mb-4">
                                        <div>
                                            <span className="text-xs text-neutral-500 block mb-1">Execution Quality</span>
                                            <div className={`text-xl font-bold ${getQualityColor(result.quality)}`}>
                                                {result.quality}
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <span className="text-xs font-mono text-neutral-400 block mb-1">Score</span>
                                            <div className="text-lg font-semibold text-white">{result.quality_numeric || 0} / 10</div>
                                        </div>
                                    </div>

                                    <div className="w-full bg-neutral-800 h-1.5 rounded-full mb-6 overflow-hidden">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${((result.quality_numeric || 0) / 10) * 100}%` }}
                                            transition={{ duration: 0.8, ease: "easeOut" }}
                                            className={`h-full rounded-full ${getQualityBarColor(result.quality)}`}
                                        />
                                    </div>

                                    {result.tactical_analysis && (
                                        <div className="pt-4 border-t border-neutral-800/50">
                                            <span className="text-xs text-neutral-500 block mb-3">Tactical Metrics</span>
                                            <div className="flex flex-wrap gap-2">
                                                <div className="px-2 py-1 bg-blue-500/10 border border-blue-500/30 rounded text-[10px] font-medium text-blue-400 flex items-center gap-1.5">
                                                    <Icon name="pan_tool_alt" size={12} />
                                                    {result.tactical_analysis.technique?.label || 'Unknown'}
                                                    {result.tactical_analysis.technique?.confidence > 0 && (
                                                        <span className="text-[9px] opacity-60">{(result.tactical_analysis.technique.confidence * 100).toFixed(0)}%</span>
                                                    )}
                                                </div>
                                                <div className="px-2 py-1 bg-purple-500/10 border border-purple-500/30 rounded text-[10px] font-medium text-purple-400 flex items-center gap-1.5">
                                                    <Icon name="explore" size={12} />
                                                    {result.tactical_analysis.placement?.label || 'Unknown'}
                                                    {result.tactical_analysis.placement?.confidence > 0 && (
                                                        <span className="text-[9px] opacity-60">{(result.tactical_analysis.placement.confidence * 100).toFixed(0)}%</span>
                                                    )}
                                                </div>
                                                <div className="px-2 py-1 bg-rose-500/10 border border-purple-500/30 rounded text-[10px] font-medium text-rose-400 flex items-center gap-1.5">
                                                    <Icon name="location_on" size={12} />
                                                    {result.tactical_analysis.position?.label || 'Unknown'}
                                                    {result.tactical_analysis.position?.confidence > 0 && (
                                                        <span className="text-[9px] opacity-60">{(result.tactical_analysis.position.confidence * 100).toFixed(0)}%</span>
                                                    )}
                                                </div>
                                                <div className="px-2 py-1 bg-amber-500/10 border border-amber-500/30 rounded text-[10px] font-medium text-amber-400 flex items-center gap-1.5">
                                                    <Icon name="psychology" size={12} />
                                                    {result.tactical_analysis.intent?.label || 'None'}
                                                    {result.tactical_analysis.intent?.confidence > 0 && (
                                                        <span className="text-[9px] opacity-60">{(result.tactical_analysis.intent.confidence * 100).toFixed(0)}%</span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Coach's Recommendations */}
                            {result.recommendations && result.recommendations.length > 0 && (
                                <div className="p-4 bg-emerald-950/20 rounded-md border border-emerald-900/40">
                                    <span className="text-xs text-emerald-500/80 flex items-center gap-1.5 mb-3 font-medium">
                                        <Icon name="tips_and_updates" size={14} />
                                        Coach's Recommendations
                                    </span>
                                    <ul className="space-y-2">
                                        {result.recommendations.map((tip, idx) => (
                                            <li key={idx} className="text-sm text-neutral-300 flex items-start gap-2">
                                                <span className="text-emerald-500 mt-1">•</span>
                                                {tip}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {/* Timeline Breakdown */}
                            {result.timeline && (
                                <div className="p-4 bg-neutral-950 rounded-md border border-neutral-800">
                                    <span className="text-xs text-neutral-500 flex items-center gap-1.5 mb-4">
                                        <Icon name="timeline" size={14} />
                                        Play-by-Play Breakdown
                                    </span>

                                    {/* Mobile: vertical text timeline */}
                                    <div className="block md:hidden relative border-l border-neutral-800 ml-2 space-y-6 py-1 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                                        {result.timeline.map((event, idx) => (
                                            <div
                                                key={idx}
                                                onClick={() => handleTimelineClick(event.timestamp)}
                                                className="relative pl-6 py-2 rounded hover:bg-neutral-900 transition-colors cursor-pointer group"
                                            >
                                                <div className={`absolute -left-[5.5px] top-4 w-2.5 h-2.5 rounded-full border-2 ${event.label === 'Other'
                                                    ? 'bg-neutral-950 border-neutral-600'
                                                    : 'bg-neutral-950 border-emerald-500'
                                                    }`} />
                                                <div className="flex flex-col gap-3">
                                                    <div className="flex items-start justify-between">
                                                        <div>
                                                            <span className="text-xs font-mono text-neutral-500 block">{event.timestamp}</span>
                                                            <span className={`text-base font-semibold ${event.label === 'Other' ? 'text-neutral-500' : 'text-neutral-100'}`}>
                                                                {event.label.replace(/_/g, ' ')}
                                                            </span>
                                                            <span className="text-[10px] text-neutral-600 ml-2">{(event.confidence * 100).toFixed(0)}%</span>
                                                        </div>
                                                        {event.pose_image && (
                                                            <div className="w-24 h-16 rounded overflow-hidden bg-black/50 border border-neutral-800 flex-shrink-0">
                                                                <img
                                                                    src={`data:image/jpeg;base64,${event.pose_image}`}
                                                                    alt={event.label}
                                                                    className="w-full h-full object-contain"
                                                                />
                                                            </div>
                                                        )}
                                                    </div>

                                                    {event.metrics && (
                                                        <div className="flex flex-wrap gap-1.5">
                                                            <span className="px-1.5 py-0.5 bg-blue-500/5 text-blue-400/70 border border-blue-500/20 rounded text-[9px] uppercase tracking-wider font-semibold">
                                                                {event.metrics.technique?.label || event.metrics.technique || 'Unknown'}
                                                            </span>
                                                            <span className="px-1.5 py-0.5 bg-purple-500/5 text-purple-400/70 border border-purple-500/20 rounded text-[9px] uppercase tracking-wider font-semibold">
                                                                {event.metrics.placement?.label || event.metrics.placement || 'Unknown'}
                                                            </span>
                                                            <span className="px-1.5 py-0.5 bg-rose-500/5 text-rose-400/70 border border-rose-500/20 rounded text-[9px] uppercase tracking-wider font-semibold">
                                                                {event.metrics.position?.label || event.metrics.position || 'Unknown'}
                                                            </span>
                                                            <span className="px-1.5 py-0.5 bg-amber-500/5 text-amber-400/70 border border-amber-500/20 rounded text-[9px] uppercase tracking-wider font-semibold">
                                                                {event.metrics.intent?.label || event.metrics.intent || 'None'}
                                                            </span>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                    </div>

                                    {/* Desktop: horizontal skeleton strip */}
                                    <div className="hidden md:flex gap-4 overflow-x-auto pb-6 pt-2 custom-scrollbar">
                                        {result.timeline.map((event, idx) => (
                                            <div
                                                key={idx}
                                                onClick={() => handleTimelineClick(event.timestamp)}
                                                className="flex-shrink-0 cursor-pointer group w-44"
                                            >
                                                <div className={`w-44 h-32 rounded overflow-hidden bg-black border transition-all duration-300 ${event.pose_image ? 'border-neutral-800 group-hover:border-emerald-600 group-hover:scale-[1.02]' : 'border-neutral-900 flex items-center justify-center'}`}>
                                                    {event.pose_image ? (
                                                        <img
                                                            src={`data:image/jpeg;base64,${event.pose_image}`}
                                                            alt={event.label}
                                                            className="w-full h-full object-contain"
                                                        />
                                                    ) : (
                                                        <Icon name="hide_image" size={24} className="text-neutral-800" />
                                                    )}
                                                </div>
                                                <div className="mt-3 space-y-2 px-1">
                                                    <div className="flex justify-between items-center">
                                                        <span className="text-[10px] font-mono text-neutral-500">{event.timestamp}</span>
                                                        <span className="text-[10px] text-neutral-600">{(event.confidence * 100).toFixed(0)}%</span>
                                                    </div>
                                                    <span className={`text-sm block truncate group-hover:text-emerald-400 transition-colors ${event.label === 'Other' ? 'text-neutral-500' : 'text-neutral-200 font-semibold'}`}>
                                                        {event.label.replace(/_/g, ' ')}
                                                    </span>

                                                    {event.metrics && (
                                                        <div className="grid grid-cols-2 gap-1 mt-2 border-t border-neutral-800/30 pt-2">
                                                            <div className="flex items-center gap-1 text-[8px] text-neutral-500 uppercase font-bold tracking-tighter truncate">
                                                                <Icon name="pan_tool_alt" size={10} className="text-blue-500/50" />
                                                                {event.metrics.technique?.label || event.metrics.technique || '???'}
                                                            </div>
                                                            <div className="flex items-center gap-1 text-[8px] text-neutral-500 uppercase font-bold tracking-tighter truncate">
                                                                <Icon name="explore" size={10} className="text-purple-500/50" />
                                                                {event.metrics.placement?.label || event.metrics.placement || '???'}
                                                            </div>
                                                            <div className="flex items-center gap-1 text-[8px] text-neutral-500 uppercase font-bold tracking-tighter truncate">
                                                                <Icon name="location_on" size={10} className="text-rose-500/50" />
                                                                {event.metrics.position?.label || event.metrics.position || '???'}
                                                            </div>
                                                            <div className="flex items-center gap-1 text-[8px] text-neutral-500 uppercase font-bold tracking-tighter truncate">
                                                                <Icon name="psychology" size={10} className="text-amber-500/50" />
                                                                {event.metrics.intent?.label || event.metrics.intent || '???'}
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center py-16 text-neutral-600">
                            <Icon name="pending" size={32} className="mb-3 text-neutral-700" />
                            <p className="text-sm">Upload a clip to get started</p>
                        </div>
                    )}
                </section>
            </div>
        </div>
    )
}
