import { track } from '../analytics.js'
import AppShell from './AppShell'
import { Icon } from './AnalyzeIcon'
import AnalyzeFrameLightbox from './AnalyzeFrameLightbox'
import AnalyzeInputSection from './AnalyzeInputSection'
import AnalyzeResultsSection from './AnalyzeResultsSection'

export default function AnalyzePageContent({ controller }) {
    const {
        file,
        setFile,
        result,
        setResult,
        loading,
        preview,
        setPreview,
        loadingStep,
        capacityError,
        setCapacityError,
        queueAhead,
        videoRef,
        lightboxEvent,
        closeLightbox,
        frameTip,
        frameTipLoading,
        isMobile,
        inputMode,
        isRecording,
        cameraError,
        recordingSeconds,
        cameraPreviewRef,
        cameraStreamRef,
        nativeVideoInputRef,
        isFullScreen,
        getRootProps,
        getInputProps,
        isDragActive,
        switchMode,
        handleNativeVideoSelect,
        openCamera,
        toggleFullScreen,
        startRecording,
        stopRecording,
        handleStreamAnalysis,
        formatSeconds,
        loadingSteps,
        displayTimeline,
        getQualityColor,
        getQualityBarColor,
        timelineHasPoseFrames,
        handleTimelineClick,
        openFrameAnalysis,
    } = controller

    return (
        <AppShell active="analyze" mainClassName="mx-auto max-w-4xl px-5 py-8 sm:px-8">
            <AnalyzeFrameLightbox
                lightboxEvent={lightboxEvent}
                closeLightbox={closeLightbox}
                frameTip={frameTip}
                frameTipLoading={frameTipLoading}
            />

            {capacityError !== null && (
                <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 flex items-start gap-3 bg-amber-950/90 border border-amber-700/60 text-amber-200 text-sm px-5 py-3.5 rounded-xl shadow-2xl backdrop-blur-sm max-w-sm w-full">
                    <Icon name="hourglass_top" size={18} className="text-amber-400 flex-shrink-0 mt-0.5" />
                    <div>
                        <p className="font-semibold text-amber-300 mb-0.5">IsoCourt is fully loaded right now 🏸</p>
                        <p className="text-xs text-amber-200/80">We're popular! Please try again in ~{capacityError}s. Your clip is worth the wait.</p>
                    </div>
                    <button
                        type="button"
                        onClick={() => setCapacityError(null)}
                        className="ml-auto text-amber-400 hover:text-amber-200"
                        aria-label="Dismiss capacity notice"
                    >
                        <Icon name="close" size={16} />
                    </button>
                </div>
            )}

            {queueAhead !== null && loading && (
                <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 flex items-start gap-3 bg-sky-950/90 border border-sky-700/50 text-sky-100 text-sm px-5 py-3 rounded-xl shadow-2xl backdrop-blur-sm max-w-sm w-full mt-[4.5rem]">
                    <Icon name="pending_actions" size={18} className="text-sky-400 flex-shrink-0 mt-0.5" />
                    <div>
                        <p className="font-semibold text-sky-200 mb-0.5">You&apos;re in the queue</p>
                        <p className="text-xs text-sky-100/85">
                            {queueAhead === 0
                                ? 'Starting your analysis shortly…'
                                : `${queueAhead} video${queueAhead === 1 ? '' : 's'} ahead of yours.`}
                        </p>
                    </div>
                </div>
            )}

            <h1 className="app-page-title mb-1">analyze a clip</h1>
            <p className="app-page-lead mb-6">Upload or record a rally — Birdzo reads strokes, footwork, and form.</p>

            <AnalyzeInputSection
                file={file}
                loading={loading}
                loadingStep={loadingStep}
                inputMode={inputMode}
                switchMode={switchMode}
                handleStreamAnalysis={handleStreamAnalysis}
                uploadPanelProps={{
                    preview,
                    result,
                    videoRef,
                    setFile,
                    setPreview,
                    getRootProps,
                    getInputProps,
                    isDragActive,
                }}
                recordPanelProps={{
                    deviceType: isMobile ? 'mobile' : 'desktop',
                    mobileProps: {
                        preview,
                        result,
                        videoRef,
                        setFile,
                        setPreview,
                        nativeVideoInputRef,
                        handleNativeVideoSelect,
                    },
                    desktopProps: {
                        preview,
                        result,
                        videoRef,
                        setFile,
                        setPreview,
                        cameraError,
                        cameraPreviewRef,
                        cameraStreamRef,
                        isRecording,
                        isFullScreen,
                        recordingSeconds,
                        openCamera,
                        toggleFullScreen,
                        startRecording,
                        stopRecording,
                        formatSeconds,
                    },
                }}
            />

            <AnalyzeResultsSection
                result={result}
                setResult={setResult}
                setFile={setFile}
                setPreview={setPreview}
                loadingStep={loadingStep}
                loadingSteps={loadingSteps}
                displayTimeline={displayTimeline}
                preview={preview}
                videoRef={videoRef}
                getQualityColor={getQualityColor}
                getQualityBarColor={getQualityBarColor}
                timelineHasPoseFrames={timelineHasPoseFrames}
                handleTimelineClick={handleTimelineClick}
                openFrameAnalysis={openFrameAnalysis}
            />

            {result && !result.validation_error && (
                <div className="mt-4 text-center">
                    <a
                        href="/#feedback"
                        onClick={() => track('feedback_link_clicked', { placement: 'analyze_page' })}
                        className="inline-flex items-center gap-1.5 text-xs text-neutral-500 hover:text-brand transition-colors"
                    >
                        <Icon name="chat" size={14} />
                        Have feedback? Let us know
                    </a>
                </div>
            )}
        </AppShell>
    )
}
