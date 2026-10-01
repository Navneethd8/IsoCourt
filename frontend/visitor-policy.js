/** Shared by the Vercel middleware and the browser tag. */

export const PRODUCTION_HOSTS = new Set(['www.isocourt.fit', 'isocourt.fit'])
export const VISITOR_COOKIE = 'ic_vid'
export const BOT_COOKIE = 'ic_bot'
export const VISITOR_MAX_AGE = 60 * 60 * 24 * 730

const BOT_UA =
    /bot\b|crawler|spider|slurp|facebookexternalhit|embedly|quora link preview|whatsapp|telegrambot|wget|curl\/|headless|phantomjs|selenium|puppeteer|playwright|lighthouse|pagespeed|gtmetrix|pingdom|uptimerobot|huggingface|bytespider|petalbot|ahrefs|semrush|dotbot|mj12bot|yandex|baiduspider|duckduckbot|applebot|bingpreview|googlebot|adsbot|mediapartners-google/i

export function isBotUserAgent(ua) {
    if (!ua || typeof ua !== 'string') return true
    return BOT_UA.test(ua)
}

export function isCloudflareBot(headers) {
    if (!headers || typeof headers.get !== 'function') return false
    if (headers.get('cf-verified-bot') === 'true') return true
    const score = headers.get('cf-bot-score')
    if (score == null || score === '') return false
    const value = Number(score)
    return Number.isFinite(value) && value < 30
}

export function readCookie(header, name) {
    if (!header || typeof header !== 'string') return null
    for (const part of header.split(';')) {
        const [key, ...rest] = part.trim().split('=')
        if (key === name) return rest.join('=')
    }
    return null
}

/** GA4 client ids are two dot-separated integers. The second is a unix timestamp. */
export function parseVisitorId(value) {
    if (typeof value !== 'string') return null
    return /^\d{10}\.\d{10}$/.test(value) ? value : null
}

export function createVisitorId(nowSeconds = Math.floor(Date.now() / 1000)) {
    const buf = new Uint32Array(1)
    crypto.getRandomValues(buf)
    const n = (buf[0] % 9000000000) + 1000000000
    return `${n}.${nowSeconds}`
}

export function visitorCookie(id) {
    return `${VISITOR_COOKIE}=${id}; Path=/; Max-Age=${VISITOR_MAX_AGE}; Secure; SameSite=Lax`
}

export function botCookie() {
    return `${BOT_COOKIE}=1; Path=/; Max-Age=86400; Secure; SameSite=Lax`
}
