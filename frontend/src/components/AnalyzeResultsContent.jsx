import { Icon } from './AnalyzeIcon'
import AnalyzeLoadingStepChips from './AnalyzeLoadingStepChips'
import AnalyzeResultSummary from './AnalyzeResultSummary'
import AnalyzeCourtHeatmap from './AnalyzeCourtHeatmap'
import AnalyzeRecommendations from './AnalyzeRecommendations'
import AnalyzeTimeline from './AnalyzeTimeline'

export default function AnalyzeResultsContent({
    result,
    loadingStep,
    loadingSteps,
    displayTimeline,
    getQualityColor,
    getQualityBarColor,
    timelineHasPoseFrames,
    handleTimelineClick,
    openFrameAnalysis,
}) {
    return (
        <div className="space-y-4">
            {loadingStep >= 0 && (
                <AnalyzeLoadingStepChips loadingSteps={loadingSteps} loadingStep={loadingStep} />
            )}

            {result?.cache_hit && (
                <div className="flex items-center gap-2 text-[11px] text-brand/70 font-medium px-1">
                    <Icon name="bolt" size={13} className="text-brand" />
                    Instant result: same clip analyzed before
                </div>
            )}
            {result?.cache_hit && result.timeline?.length > 0 && !timelineHasPoseFrames(result.timeline) && (
                <p className="text-xs text-amber-700 dark:text-amber-400/90 px-1">
                    Cached result has no pose frames. Trim a second off the clip or wait ~1 hour, then analyze again for skeleton views.
                </p>
            )}

            {result && (
                <AnalyzeResultSummary
                    result={result}
                    getQualityColor={getQualityColor}
                    getQualityBarColor={getQualityBarColor}
                />
            )}

            {displayTimeline.length > 0 && (
                <AnalyzeCourtHeatmap timeline={displayTimeline} />
            )}

            {displayTimeline.length > 0 && (
                <AnalyzeTimeline
                    displayTimeline={displayTimeline}
                    loadingStep={loadingStep}
                    result={result}
                    handleTimelineClick={handleTimelineClick}
                    openFrameAnalysis={openFrameAnalysis}
                />
            )}

            {result?.recommendations && result.recommendations.length > 0 && (
                <AnalyzeRecommendations recommendations={result.recommendations} />
            )}
        </div>
    )
}
