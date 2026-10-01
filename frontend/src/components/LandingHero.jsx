import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { track } from '../analytics.js'
import FigmaButton from './FigmaButton'
import HeroStaticBackdrop from './HeroStaticBackdrop'

const HeroFigmaBackdrop = lazy(() => import('./HeroFigmaBackdrop'))

export default function LandingHero() {
    const heroRef = useRef(null)
    const [heroFx, setHeroFx] = useState(false)

    useEffect(() => {
        // Static racket paints immediately. Upgrade to scroll-driven motion after idle
        // (or a short timeout) so Lighthouse / first paint stay light without a blank hero.
        let cancelled = false
        const enable = () => {
            if (!cancelled) setHeroFx(true)
        }
        const idleId =
            typeof window.requestIdleCallback === 'function'
                ? window.requestIdleCallback(enable, { timeout: 1800 })
                : null
        const t = window.setTimeout(enable, idleId == null ? 400 : 2200)
        return () => {
            cancelled = true
            if (idleId != null && typeof window.cancelIdleCallback === 'function') {
                window.cancelIdleCallback(idleId)
            }
            window.clearTimeout(t)
        }
    }, [])

    return (
        <section ref={heroRef} className="figma-hero">
            <div className="figma-hero-artboard" aria-hidden>
                {heroFx ? (
                    <Suspense fallback={<HeroStaticBackdrop />}>
                        <HeroFigmaBackdrop scrollTarget={heroRef} />
                    </Suspense>
                ) : (
                    <HeroStaticBackdrop />
                )}
            </div>

            <div className="figma-hero-content">
                <h1 className="figma-display-title">
                    <span>Meet </span>
                    <span className="figma-brand-accent">Birdzo</span>
                    <span>, your </span>
                    <span className="figma-brand-accent">second pair of eyes</span>
                    <br />
                    on court.
                </h1>
                <p className="figma-hero-sub">
                    Upload a rally or single stroke. IsoCourt&apos;s Birdzo coach reads footwork, contact, and shot type. Fast, specific, zero fluff.
                </p>

                <div className="figma-hero-ctas">
                    <FigmaButton
                        variant="primary"
                        href="/analyze"
                        onClick={() => track('cta_click', { cta: 'analyze', placement: 'hero' })}
                    >
                        Drop a clip
                    </FigmaButton>
                    <FigmaButton
                        variant="secondary"
                        href="/live"
                        onClick={() => track('cta_click', { cta: 'live', placement: 'hero' })}
                    >
                        Go live
                    </FigmaButton>
                </div>
            </div>
        </section>
    )
}
