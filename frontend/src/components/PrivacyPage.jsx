import LegalLayout from './LegalLayout'
import { usePageSeo } from '../seo/usePageSeo'

export default function PrivacyPage() {
    usePageSeo('/privacy')
    return (
        <LegalLayout title="Privacy">
            <p>
                IsoCourt processes the <strong className="text-[var(--text)] font-semibold">video clips you upload</strong> so we can run pose and stroke analysis. Clips are
                handled in line with how our backend is configured (retention and storage may change; we aim to keep this page accurate).
            </p>
            <p>
                If you use the <strong className="text-[var(--text)] font-semibold">feedback form</strong>, we receive the name, email, and message you submit so we can reply.
            </p>
            <p>
                On www.isocourt.fit the server sets a first-party cookie named ic_vid so repeat visits from the same browser can be counted without an account. Known bots do not get that cookie. This site also uses <strong className="text-[var(--text)] font-semibold">Google Analytics</strong> on www.isocourt.fit only (see Google&apos;s privacy policy for how they process data). You can use browser controls or extensions to limit tracking.
            </p>
            <p className="text-[var(--text-subtle)] text-xs pt-4">
                This is a summary, not legal advice. For deletion requests or questions, contact us through the feedback form on the home page.
            </p>
        </LegalLayout>
    )
}
