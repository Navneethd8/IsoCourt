import { useRef, useEffect } from 'react'
import { Icon } from './AnalyzeIcon'
import { getQualityColor } from '../useAnalyzeController'

export default function AnalyzeFrameLightbox({
    lightboxEvent,
    closeLightbox,
    frameTip,
    frameTipLoading,
}) {
    const dialogRef = useRef(null)

    useEffect(() => {
        if (lightboxEvent) {
            dialogRef.current?.showModal()
        } else {
            dialogRef.current?.close()
        }
    }, [lightboxEvent])

    const handleCancel = (e) => {
        e.preventDefault()
        closeLightbox()
    }

    return (
        <dialog
            ref={dialogRef}
            className="fixed inset-0 z-[100] m-0 h-full max-h-none w-full max-w-none items-center justify-center border-none bg-transparent p-4 md:p-8 [&[open]]:flex"
            onCancel={handleCancel}
            aria-label="Frame analysis"
        >
            <button
                type="button"
                className="absolute inset-0 z-0 border-0 bg-black/80 backdrop-blur-sm p-0"
                onClick={closeLightbox}
                aria-label="Close frame analysis"
            />
            {lightboxEvent && (
                <div className="relative z-10 max-h-[90vh] w-full max-w-lg overflow-hidden rounded-lg app-modal">
                    <button
                        type="button"
                        onClick={closeLightbox}
                        className="absolute top-3 right-3 z-20 app-lightbox-close"
                        aria-label="Close"
                    >
                        <Icon name="close" size={18} />
                    </button>

                    {lightboxEvent.pose_image && (
                        <div className="flex w-full items-center justify-center bg-black">
                            <img
                                src={`data:image/jpeg;base64,${lightboxEvent.pose_image}`}
                                alt={lightboxEvent.label}
                                className="max-h-[50vh] w-full object-contain"
                            />
                        </div>
                    )}

                    <div className="space-y-4 p-5">
                        <div className="flex items-center justify-between">
                            <div>
                                <span className="block font-mono text-xs app-lightbox-label">{lightboxEvent.timestamp}</span>
                                <span className={`text-lg font-bold ${lightboxEvent.label === 'Other' ? 'app-lightbox-label' : 'app-lightbox-title'}`}>
                                    {lightboxEvent.label?.replace(/_/g, ' ')}
                                </span>
                            </div>
                            <div className="text-right">
                                <span className="block text-xs app-lightbox-label">Confidence</span>
                                <span className="text-lg font-semibold text-brand">
                                    {(lightboxEvent.confidence * 100).toFixed(0)}%
                                </span>
                            </div>
                        </div>

                        {lightboxEvent.metrics && (
                            <div className="grid grid-cols-2 gap-2">
                                <div className="rounded border border-blue-500/20 bg-blue-500/10 px-2.5 py-2">
                                    <span className="mb-0.5 block text-[9px] font-semibold uppercase tracking-wider text-blue-400/60">Technique</span>
                                    <span className="flex items-center gap-1.5 text-xs font-medium text-blue-300">
                                        <Icon name="pan_tool_alt" size={12} />
                                        {lightboxEvent.metrics.technique?.label || lightboxEvent.metrics.technique || 'Unknown'}
                                    </span>
                                </div>
                                <div className="rounded border border-purple-500/20 bg-purple-500/10 px-2.5 py-2">
                                    <span className="mb-0.5 block text-[9px] font-semibold uppercase tracking-wider text-purple-400/60">Placement</span>
                                    <span className="flex items-center gap-1.5 text-xs font-medium text-purple-300">
                                        <Icon name="explore" size={12} />
                                        {lightboxEvent.metrics.placement?.label || lightboxEvent.metrics.placement || 'Unknown'}
                                    </span>
                                </div>
                                <div className="rounded border border-rose-500/20 bg-rose-500/10 px-2.5 py-2">
                                    <span className="mb-0.5 block text-[9px] font-semibold uppercase tracking-wider text-rose-400/60">Position</span>
                                    <span className="flex items-center gap-1.5 text-xs font-medium text-rose-300">
                                        <Icon name="location_on" size={12} />
                                        {lightboxEvent.metrics.position?.label || lightboxEvent.metrics.position || 'Unknown'}
                                    </span>
                                </div>
                                <div className="rounded border border-amber-500/20 bg-amber-500/10 px-2.5 py-2">
                                    <span className="mb-0.5 block text-[9px] font-semibold uppercase tracking-wider text-amber-400/60">Intent</span>
                                    <span className="flex items-center gap-1.5 text-xs font-medium text-amber-300">
                                        <Icon name="psychology" size={12} />
                                        {lightboxEvent.metrics.intent?.label || lightboxEvent.metrics.intent || 'None'}
                                    </span>
                                </div>
                            </div>
                        )}

                        {lightboxEvent.metrics?.quality && (
                            <div className="flex items-center gap-2 border-t app-lightbox-divider pt-2">
                                <Icon name="star" size={14} className="text-amber-500" />
                                <span className="text-xs app-lightbox-label">Quality:</span>
                                <span className={`text-xs font-semibold ${getQualityColor(lightboxEvent.metrics.quality)}`}>
                                    {lightboxEvent.metrics.quality}
                                </span>
                            </div>
                        )}

                        <div className="border-t app-lightbox-divider pt-3">
                            <span className="mb-2 flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-wider text-brand/80">
                                <Icon name="tips_and_updates" size={12} />
                                AI Coach Tip
                            </span>
                            {frameTipLoading ? (
                                <div className="flex items-center gap-2">
                                    <span className="h-3 w-3 animate-spin rounded-full border-2 border-brand/30 border-t-emerald-400" />
                                    <span className="text-xs italic app-lightbox-label">Generating tip...</span>
                                </div>
                            ) : frameTip ? (
                                <p className="text-sm leading-relaxed app-lightbox-body">{frameTip}</p>
                            ) : (
                                <p className="text-xs italic app-lightbox-label">Tip unavailable</p>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </dialog>
    )
}
