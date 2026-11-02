import { Icon } from './AnalyzeIcon'
import AnalyzeLoadingSteps from './AnalyzeLoadingSteps'
import AnalyzeValidationError from './AnalyzeValidationError'
import AnalyzeResultsContent from './AnalyzeResultsContent'

export default function AnalyzeResultsSection({
    result,
    setResult,
    setFile,
    setPreview,
    loadingStep,
    loadingSteps,
    displayTimeline,
    getQualityColor,
    getQualityBarColor,
    timelineHasPoseFrames,
    handleTimelineClick,
    openFrameAnalysis,
}) {
    const isLoadingOnly = loadingStep >= 0 && displayTimeline.length === 0
    const hasValidationError = Boolean(result?.validation_error)
    const hasContent = Boolean(result || displayTimeline.length > 0)

    return (
        <section className="app-card">
            <h2 className="app-card-title">
                <Icon name="analytics" size={18} />
                Analysis Results
            </h2>

            {isLoadingOnly ? (
                <AnalyzeLoadingSteps loadingSteps={loadingSteps} loadingStep={loadingStep} />
            ) : hasValidationError ? (
                <AnalyzeValidationError
                    result={result}
                    setResult={setResult}
                    setFile={setFile}
                    setPreview={setPreview}
                />
            ) : hasContent ? (
                <AnalyzeResultsContent
                    result={result}
                    loadingStep={loadingStep}
                    loadingSteps={loadingSteps}
                    displayTimeline={displayTimeline}
                    getQualityColor={getQualityColor}
                    getQualityBarColor={getQualityBarColor}
                    timelineHasPoseFrames={timelineHasPoseFrames}
                    handleTimelineClick={handleTimelineClick}
                    openFrameAnalysis={openFrameAnalysis}
                />
            ) : (
                <div className="flex flex-col items-center justify-center py-16 text-neutral-600">
                    <Icon name="pending" size={32} className="mb-3 text-neutral-700" />
                    <p className="text-sm">Upload a clip to get started</p>
                </div>
            )}
        </section>
    )
}
