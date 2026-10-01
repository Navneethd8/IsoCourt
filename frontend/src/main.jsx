import React, { Suspense, lazy, useEffect } from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { installDeferredBoot, trackPageView } from './analytics.js'
import './fonts.css'
import './index.css'
import { ThemeProvider } from './context/ThemeContext.jsx'
import JsonLd from './seo/JsonLd.jsx'
// Homepage is the critical path — keep it in the main bundle so / never waits on a lazy chunk.
import LandingPage from './components/LandingPage.jsx'

const App = lazy(() => import('./App.jsx'))
const LiveSession = lazy(() => import('./components/LiveSession.jsx'))
const PrivacyPage = lazy(() => import('./components/PrivacyPage.jsx'))
const TermsPage = lazy(() => import('./components/TermsPage.jsx'))
const FaqPage = lazy(() => import('./components/FaqPage.jsx'))
const GlossaryPage = lazy(() => import('./components/GlossaryPage.jsx'))
const ComparePage = lazy(() => import('./components/ComparePage.jsx'))
const WhatIsStrokeAnalysisPage = lazy(() => import('./components/WhatIsStrokeAnalysisPage.jsx'))
const NotFoundPage = lazy(() => import('./components/NotFoundPage.jsx'))

installDeferredBoot()

function RouteAnalytics() {
    const { pathname } = useLocation()
    useEffect(() => {
        trackPageView(pathname)
    }, [pathname])
    return null
}

function RouteFallback() {
    return (
        <div className="theme-page flex min-h-screen items-center justify-center text-sm text-[var(--text-muted)]">
            Loading…
        </div>
    )
}

/** Drop crawler-only HTML after the SPA has painted — never before. */
function ClearSeoStatic() {
    useEffect(() => {
        const drop = () => document.getElementById('seo-static')?.remove()
        const id = window.requestAnimationFrame(drop)
        return () => window.cancelAnimationFrame(id)
    }, [])
    return null
}

ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <ThemeProvider>
            <BrowserRouter>
                <RouteAnalytics />
                <JsonLd />
                <ClearSeoStatic />
                <Suspense fallback={<RouteFallback />}>
                    <Routes>
                        <Route path="/" element={<LandingPage />} />
                        <Route path="/analyze" element={<App />} />
                        <Route path="/live" element={<LiveSession />} />
                        <Route path="/faq" element={<FaqPage />} />
                        <Route path="/glossary" element={<GlossaryPage />} />
                        <Route path="/compare" element={<ComparePage />} />
                        <Route
                            path="/what-is-ai-badminton-stroke-analysis"
                            element={<WhatIsStrokeAnalysisPage />}
                        />
                        <Route path="/privacy" element={<PrivacyPage />} />
                        <Route path="/terms" element={<TermsPage />} />
                        <Route path="*" element={<NotFoundPage />} />
                    </Routes>
                </Suspense>
            </BrowserRouter>
        </ThemeProvider>
    </React.StrictMode>,
)
