import { track } from '../analytics.js'
import FigmaButton from './FigmaButton'

export default function LandingFinalCta() {
    return (
        <section className="figma-section figma-final-cta px-5 sm:px-8 scroll-mt-20">
            <div className="figma-section-inner figma-section-inner--narrow text-center">
                <h2 className="figma-section-title">what do i do now?</h2>
                <p className="figma-final-sub mt-8">
                    give <span className="figma-brand-accent">IsoCourt</span> a go and see what{' '}
                    <span className="figma-brand-accent">birdzo</span> has in mind for you
                </p>
                <div className="figma-hero-ctas mt-10">
                    <FigmaButton
                        variant="primary"
                        href="/analyze"
                        onClick={() => track('cta_click', { cta: 'analyze', placement: 'footer' })}
                    >
                        Drop a clip
                    </FigmaButton>
                    <FigmaButton
                        variant="secondary"
                        href="/live"
                        onClick={() => track('cta_click', { cta: 'live', placement: 'footer' })}
                    >
                        Go live
                    </FigmaButton>
                </div>
            </div>
        </section>
    )
}
