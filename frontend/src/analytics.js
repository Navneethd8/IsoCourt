import ReactGA from 'react-ga4'
import {
    BOT_COOKIE,
    PRODUCTION_HOSTS,
    VISITOR_COOKIE,
    isBotUserAgent,
    parseVisitorId,
    readCookie,
} from '../visitor-policy.js'

/** GA4 property for isocourt.fit. Key events are marked in Admin, not here. See frontend/GA4_KEY_EVENTS.md. */
export const GA_MEASUREMENT_ID = 'G-TET6JN36Q4'

const IMMEDIATE_PATHS = new Set(['/analyze', '/live'])
const INTERACTION_EVENTS = ['pointerdown', 'keydown', 'scroll', 'touchstart']

const EVENT_NAMES = new Set([
    'page_view',
    'cta_click',
    'analyze_started',
    'analyze_completed',
    'analyze_failed',
    'live_start_attempted',
    'live_session_started',
    'live_coaching_delivered',
    'live_session_failed',
    'feedback_sent',
    'feedback_link_clicked',
])

const ENUMS = {
    cta: new Set(['analyze', 'live']),
    placement: new Set(['hero', 'nav', 'footer', 'landing_page', 'analyze_page']),
    source: new Set(['upload', 'record']),
    reason: new Set([
        'http_error',
        'validation',
        'too_long',
        'capacity',
        'stream_dropped',
        'camera_denied',
        'ws_error',
        'no_result',
    ]),
    stage: new Set(['submit', 'stream', 'create', 'camera', 'socket', 'session']),
}

const REQUIRED = {
    page_view: ['page_path'],
    cta_click: ['cta', 'placement'],
    analyze_started: ['source'],
    analyze_failed: ['reason'],
    live_session_failed: ['reason'],
    feedback_sent: ['placement'],
    feedback_link_clicked: ['placement'],
}

let booted = false
const queue = []
let lastPageView = { path: '', at: 0 }

function normalizePath(path) {
    if (typeof path !== 'string') return null
    const pathname = path.split('?')[0].split('#')[0]
    if (!/^\/[A-Za-z0-9._~/-]*$/.test(pathname) || pathname.length > 128) return null
    return pathname
}

function sanitize(name, params, beacon) {
    if (!EVENT_NAMES.has(name)) return null
    const clean = {}
    if (params && typeof params === 'object') {
        for (const [key, value] of Object.entries(params)) {
            if (key === 'page_path') {
                const pagePath = normalizePath(value)
                if (pagePath) clean.page_path = pagePath
                continue
            }
            if (key === 'status_code') {
                if (Number.isInteger(value) && value >= 100 && value <= 599) clean.status_code = value
                continue
            }
            const allowed = ENUMS[key]
            if (allowed && allowed.has(value)) clean[key] = value
        }
    }
    const required = REQUIRED[name] || []
    if (required.some((key) => clean[key] == null)) return null
    if (beacon) clean.transport_type = 'beacon'
    return { name, params: clean }
}

function dispatch(item) {
    ReactGA.event(item.name, item.params)
}

function visitorId() {
    if (typeof document === 'undefined') return null
    return parseVisitorId(readCookie(document.cookie, VISITOR_COOKIE))
}

/** Production site only, with the server-set visitor cookie, and not a bot. */
export function analyticsAllowed() {
    if (typeof window === 'undefined') return false
    if (!PRODUCTION_HOSTS.has(window.location.hostname)) return false
    if (navigator.webdriver) return false
    if (isBotUserAgent(navigator.userAgent)) return false
    if (readCookie(document.cookie, BOT_COOKIE) === '1') return false
    return Boolean(visitorId())
}

export function bootAnalytics() {
    if (booted || typeof window === 'undefined') return
    if (!analyticsAllowed()) return
    booted = true
    ReactGA.initialize(GA_MEASUREMENT_ID, {
        gtagOptions: {
            send_page_view: false,
            client_id: visitorId(),
        },
    })
    INTERACTION_EVENTS.forEach((type) => {
        window.removeEventListener(type, bootAnalytics)
    })
    const pending = queue.splice(0, queue.length)
    pending.forEach(dispatch)
}

/** Homepage waits for a real interaction so the GA script stays off the LCP path. */
export function installDeferredBoot() {
    if (typeof window === 'undefined' || booted || !analyticsAllowed()) return
    INTERACTION_EVENTS.forEach((type) => {
        window.addEventListener(type, bootAnalytics, { once: true, passive: true })
    })
}

export function track(name, params = {}, options = {}) {
    if (!analyticsAllowed()) return
    const item = sanitize(name, params, Boolean(options.beacon))
    if (!item) return
    if (!booted) {
        queue.push(item)
        bootAnalytics()
        return
    }
    dispatch(item)
}

export function trackPageView(path) {
    if (!analyticsAllowed()) return
    const pagePath = normalizePath(path)
    if (!pagePath) return
    const now = Date.now()
    if (lastPageView.path === pagePath && now - lastPageView.at < 800) return
    lastPageView = { path: pagePath, at: now }
    const item = sanitize('page_view', { page_path: pagePath }, false)
    if (!item) return
    if (IMMEDIATE_PATHS.has(pagePath)) bootAnalytics()
    if (booted) dispatch(item)
    else queue.push(item)
}
