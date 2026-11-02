import { Icon } from './AnalyzeIcon'

function TimelineEventMetrics({ metrics, variant }) {
    if (!metrics) return null

    if (variant === 'mobile') {
        return (
            <div className="flex flex-wrap gap-1.5">
                <span className="px-1.5 py-0.5 bg-blue-500/5 text-blue-400/70 border border-blue-500/20 rounded text-[9px] uppercase tracking-wider font-semibold">
                    {metrics.technique?.label || metrics.technique || 'Unknown'}
                </span>
                <span className="px-1.5 py-0.5 bg-purple-500/5 text-purple-400/70 border border-purple-500/20 rounded text-[9px] uppercase tracking-wider font-semibold">
                    {metrics.placement?.label || metrics.placement || 'Unknown'}
                </span>
                <span className="px-1.5 py-0.5 bg-rose-500/5 text-rose-400/70 border border-rose-500/20 rounded text-[9px] uppercase tracking-wider font-semibold">
                    {metrics.position?.label || metrics.position || 'Unknown'}
                </span>
                <span className="px-1.5 py-0.5 bg-amber-500/5 text-amber-400/70 border border-amber-500/20 rounded text-[9px] uppercase tracking-wider font-semibold">
                    {metrics.intent?.label || metrics.intent || 'None'}
                </span>
            </div>
        )
    }

    return (
        <div className="mt-2 grid grid-cols-2 gap-1 border-t border-neutral-200 pt-2">
            <div className="flex items-center gap-1 truncate text-[8px] font-bold uppercase tracking-tighter text-neutral-500">
                <Icon name="pan_tool_alt" size={10} className="text-blue-500/50" />
                {metrics.technique?.label || metrics.technique || '???'}
            </div>
            <div className="flex items-center gap-1 truncate text-[8px] font-bold uppercase tracking-tighter text-neutral-500">
                <Icon name="explore" size={10} className="text-purple-500/50" />
                {metrics.placement?.label || metrics.placement || '???'}
            </div>
            <div className="flex items-center gap-1 truncate text-[8px] font-bold uppercase tracking-tighter text-neutral-500">
                <Icon name="location_on" size={10} className="text-rose-500/50" />
                {metrics.position?.label || metrics.position || '???'}
            </div>
            <div className="flex items-center gap-1 truncate text-[8px] font-bold uppercase tracking-tighter text-neutral-500">
                <Icon name="psychology" size={10} className="text-amber-500/50" />
                {metrics.intent?.label || metrics.intent || '???'}
            </div>
        </div>
    )
}

function MobileTimelineEvent({ event, handleTimelineClick, openFrameAnalysis }) {
    return (
        <div
            key={`${event.timestamp}-${event.label}`}
            onClick={() => handleTimelineClick(event.timestamp)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    handleTimelineClick(event.timestamp)
                }
            }}
            className="relative pl-6 py-2 rounded app-timeline-row transition-colors cursor-pointer group"
        >
            <div className={`absolute -left-[5.5px] top-4 w-2.5 h-2.5 rounded-full border-2 app-timeline-dot ${event.label === 'Other'
                ? 'app-timeline-dot--muted'
                : 'app-timeline-dot--brand'
                }`} />
            <div className="flex flex-col gap-3">
                <div className="flex items-start justify-between gap-3">
                    <div>
                        <span className="text-xs font-mono text-neutral-500 block">{event.timestamp}</span>
                        <span className={`text-base font-semibold ${event.label === 'Other' ? 'app-timeline-event-title--muted' : 'app-timeline-event-title'}`}>
                            {event.label.replace(/_/g, ' ')}
                        </span>
                        <span className="text-[10px] text-neutral-500 ml-2">{(event.confidence * 100).toFixed(0)}%</span>
                    </div>
                    {event.pose_image && (
                        <div
                            onClick={(e) => { e.stopPropagation(); openFrameAnalysis(event) }}
                            role="button"
                            tabIndex={0}
                            aria-label={`Inspect ${event.label} at ${event.timestamp}`}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' || e.key === ' ') {
                                    e.preventDefault()
                                    e.stopPropagation()
                                    openFrameAnalysis(event)
                                }
                            }}
                            className="app-timeline-pose relative h-16 w-24 flex-shrink-0 cursor-zoom-in overflow-hidden rounded border bg-black/50 transition-colors hover:border-brand group/pose"
                        >
                            <img
                                src={`data:image/jpeg;base64,${event.pose_image}`}
                                alt={event.label}
                                className="h-full w-full object-contain"
                            />
                            <div className="absolute inset-0 flex items-center justify-center bg-black/0 transition-colors group-hover/pose:bg-black/30">
                                <Icon name="zoom_in" size={16} className="text-white opacity-0 transition-opacity group-hover/pose:opacity-80" />
                            </div>
                        </div>
                    )}
                </div>

                <TimelineEventMetrics metrics={event.metrics} variant="mobile" />
            </div>
        </div>
    )
}

function DesktopTimelineEvent({ event, handleTimelineClick, openFrameAnalysis }) {
    return (
        <div
            key={`${event.timestamp}-${event.label}`}
            onClick={() => handleTimelineClick(event.timestamp)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    handleTimelineClick(event.timestamp)
                }
            }}
            className="flex-shrink-0 w-44 cursor-pointer group"
        >
            <div
                onClick={(e) => { if (event.pose_image) { e.stopPropagation(); openFrameAnalysis(event) } }}
                role={event.pose_image ? 'button' : undefined}
                tabIndex={event.pose_image ? 0 : undefined}
                aria-label={event.pose_image ? `Inspect ${event.label} at ${event.timestamp}` : undefined}
                onKeyDown={(e) => {
                    if (event.pose_image && (e.key === 'Enter' || e.key === ' ')) {
                        e.preventDefault()
                        e.stopPropagation()
                        openFrameAnalysis(event)
                    }
                }}
                className={`app-timeline-pose relative h-32 w-44 overflow-hidden rounded border bg-black transition-all duration-300 ${event.pose_image
                    ? 'cursor-zoom-in group-hover:scale-[1.02] group-hover:border-brand'
                    : 'flex items-center justify-center border-[var(--border)]'
                    }`}
            >
                {event.pose_image ? (
                    <>
                        <img
                            src={`data:image/jpeg;base64,${event.pose_image}`}
                            alt={event.label}
                            className="h-full w-full object-contain"
                        />
                        <div className="absolute inset-0 flex items-center justify-center bg-black/0 transition-colors group-hover:bg-black/30">
                            <Icon name="zoom_in" size={20} className="text-white opacity-0 transition-opacity group-hover:opacity-80" />
                        </div>
                    </>
                ) : (
                    <Icon name="hide_image" size={24} className="text-neutral-700" />
                )}
            </div>
            <div className="mt-3 space-y-2 px-1">
                <div className="flex items-center justify-between">
                    <span className="font-mono text-[10px] text-neutral-500">{event.timestamp}</span>
                    <span className="text-[10px] text-neutral-600">{(event.confidence * 100).toFixed(0)}%</span>
                </div>
                <span className={`block truncate text-sm transition-colors group-hover:text-brand ${event.label === 'Other' ? 'app-timeline-event-title--muted' : 'font-semibold app-timeline-event-title'}`}>
                    {event.label.replace(/_/g, ' ')}
                </span>

                <TimelineEventMetrics metrics={event.metrics} variant="desktop" />
            </div>
        </div>
    )
}

export default function AnalyzeTimeline({
    displayTimeline,
    loadingStep,
    result,
    handleTimelineClick,
    openFrameAnalysis,
}) {
    return (
        <div className="app-timeline-panel">
            <span className="mb-4 flex items-center gap-1.5 text-xs text-[var(--text-subtle)]">
                <Icon name="timeline" size={14} />
                Play-by-Play Breakdown
                {loadingStep >= 0 && !result?.timeline?.length && (
                    <span className="text-[10px] font-normal text-[var(--text-muted)]">(updating…)</span>
                )}
            </span>

            <p className="mb-3 text-[11px] text-[var(--text-muted)]">
                Tap a skeleton for that frame and a coach tip.
            </p>

            <div className="block md:hidden relative border-l app-timeline-rail ml-2 space-y-6 py-1 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                {displayTimeline.map((event) => (
                    <MobileTimelineEvent
                        key={`${event.timestamp}-${event.label}`}
                        event={event}
                        handleTimelineClick={handleTimelineClick}
                        openFrameAnalysis={openFrameAnalysis}
                    />
                ))}
            </div>

            <div className="hidden md:flex gap-4 overflow-x-auto pb-6 pt-2 custom-scrollbar">
                {displayTimeline.map((event) => (
                    <DesktopTimelineEvent
                        key={`${event.timestamp}-${event.label}`}
                        event={event}
                        handleTimelineClick={handleTimelineClick}
                        openFrameAnalysis={openFrameAnalysis}
                    />
                ))}
            </div>
        </div>
    )
}
