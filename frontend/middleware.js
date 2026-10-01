import {
    BOT_COOKIE,
    PRODUCTION_HOSTS,
    VISITOR_COOKIE,
    botCookie,
    createVisitorId,
    isBotUserAgent,
    isCloudflareBot,
    parseVisitorId,
    readCookie,
    visitorCookie,
} from './visitor-policy.js'

function next(cookies) {
    const headers = new Headers({ 'x-middleware-next': '1' })
    for (const cookie of cookies) headers.append('set-cookie', cookie)
    return new Response(null, { headers })
}

export const config = {
    matcher: ['/((?!assets/|marketing/|fonts/).*)'],
}

function requestHost(request) {
    try {
        const hostname = new URL(request.url).hostname.toLowerCase()
        if (hostname) return hostname
    } catch {
        /* fall through to the host header */
    }
    const fromHeader = request.headers.get('x-forwarded-host') || request.headers.get('host') || ''
    return fromHeader.split(',')[0].trim().split(':')[0].toLowerCase()
}

export default function middleware(request) {
    const host = requestHost(request)
    if (!PRODUCTION_HOSTS.has(host)) return next([])

    if (isBotUserAgent(request.headers.get('user-agent')) || isCloudflareBot(request.headers)) {
        const already = readCookie(request.headers.get('cookie'), BOT_COOKIE)
        return next(already === '1' ? [] : [botCookie()])
    }

    const existing = parseVisitorId(readCookie(request.headers.get('cookie'), VISITOR_COOKIE))
    if (existing) return next([])
    return next([visitorCookie(createVisitorId())])
}
