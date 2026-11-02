/** Model tips are rendered as plain text, so markdown asterisks must not survive. */
export function plainCoachText(text) {
    if (!text) return ''
    return String(text)
        .replace(/\*\*([\s\S]+?)\*\*/g, '$1')
        .replace(/__([\s\S]+?)__/g, '$1')
        .replace(/[*_`]/g, '')
        .replace(/[ \t]{2,}/g, ' ')
        .trim()
}
