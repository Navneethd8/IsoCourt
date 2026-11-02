import { Icon } from './AnalyzeIcon'
import { plainCoachText } from '../plainCoachText'

export default function AnalyzeRecommendations({ recommendations }) {
    return (
        <div className="app-coach-panel">
            <span className="app-coach-panel__title">
                <Icon name="tips_and_updates" size={14} />
                Coach's Recommendations
            </span>
            <ul className="space-y-2">
                {recommendations.map((tip, index) => {
                    const text = plainCoachText(tip)
                    return (
                        <li key={`${index}-${text}`} className="app-coach-panel__item">
                            <span className="text-brand mt-1">•</span>
                            {text}
                        </li>
                    )
                })}
            </ul>
        </div>
    )
}
