import { Link } from 'react-router-dom'
import { track } from '../analytics.js'
import Logo from './Logo'
import ThemeToggle from './ThemeToggle'

export default function LandingHeader() {
    return (
        <header className="figma-top-bar">
            <div className="mx-auto flex h-14 sm:h-16 max-w-6xl items-center justify-between gap-4 px-5 sm:px-8">
                <Link
                    to="/"
                    className="flex min-w-0 items-center gap-2.5 text-white"
                    aria-label="IsoCourt home"
                >
                    <Logo size={24} className="shrink-0 text-white" />
                    <span className="font-display text-lg font-bold tracking-tight hidden sm:inline">
                        IsoCourt
                    </span>
                </Link>
                <nav className="flex items-center gap-3 sm:gap-5" aria-label="Primary">
                    <Link
                        to="/analyze"
                        onClick={() => track('cta_click', { cta: 'analyze', placement: 'nav' })}
                        className="font-mono text-[11px] uppercase tracking-[0.18em] text-white hover:text-white/90 transition-colors"
                    >
                        Analyze
                    </Link>
                    <Link
                        to="/live"
                        onClick={() => track('cta_click', { cta: 'live', placement: 'nav' })}
                        className="font-mono text-[11px] uppercase tracking-[0.18em] text-white hover:text-white/90 transition-colors"
                    >
                        Live
                    </Link>
                    <ThemeToggle />
                </nav>
            </div>
        </header>
    )
}
