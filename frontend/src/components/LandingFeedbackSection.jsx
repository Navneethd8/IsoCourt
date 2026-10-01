import { useRef, useState } from 'react'
import { track } from '../analytics.js'
import FigmaButton from './FigmaButton'
import SvgIcon from './SvgIcon'

export default function LandingFeedbackSection() {
    const [fbName, setFbName] = useState('')
    const [fbEmail, setFbEmail] = useState('')
    const [fbMessage, setFbMessage] = useState('')
    const [fbStatus, setFbStatus] = useState('idle')
    const [fbError, setFbError] = useState('')
    const busyRef = useRef(false)

    const API = import.meta.env.VITE_API_URL || ''

    const handleFeedbackSubmit = async (e) => {
        e.preventDefault()
        if (busyRef.current) return
        if (!fbName.trim() || !fbEmail.trim() || !fbMessage.trim()) return
        busyRef.current = true
        setFbStatus('sending')
        setFbError('')
        try {
            const { default: axios } = await import('axios')
            await axios.post(`${API}/feedback`, {
                name: fbName.trim(),
                email: fbEmail.trim(),
                message: fbMessage.trim(),
            })
            setFbStatus('sent')
            setFbName('')
            setFbEmail('')
            setFbMessage('')
            track('feedback_sent', { placement: 'landing_page' })
        } catch (err) {
            setFbStatus('error')
            setFbError(err?.response?.data?.detail || 'Something went wrong. Please try again.')
        } finally {
            busyRef.current = false
        }
    }

    return (
        <section id="feedback" className="figma-section figma-feedback px-5 sm:px-8 scroll-mt-20">
            <div className="figma-section-inner figma-section-inner--tight">
                <h2 className="figma-section-title text-center">court notes welcome</h2>
                <p className="figma-section-lead mt-4 text-center">
                    Wrong call, wild idea, or “this saved my smash.” We read every message between training blocks.
                </p>

                {fbStatus === 'sent' ? (
                    <div className="mt-8 rounded-xl border border-brand/30 bg-brand/10 p-8 text-center">
                        <SvgIcon name="check_circle" size={40} className="mx-auto mb-3 figma-brand-accent" />
                        <h3 className="text-lg font-semibold mb-2 figma-brand-accent">
                            Thanks for your feedback!
                        </h3>
                        <button
                            type="button"
                            onClick={() => setFbStatus('idle')}
                            className="text-xs hover:underline figma-brand-accent"
                        >
                            Send another message
                        </button>
                    </div>
                ) : (
                    <form onSubmit={handleFeedbackSubmit} className="figma-feedback-form mt-8 space-y-4">
                        <div className="grid gap-4 sm:grid-cols-2">
                            <div>
                                <label htmlFor="fb-name" className="text-xs font-medium text-[var(--text-subtle)] block mb-1.5">
                                    Name
                                </label>
                                <input
                                    id="fb-name"
                                    type="text"
                                    value={fbName}
                                    onChange={(e) => setFbName(e.target.value)}
                                    required
                                    className="figma-input w-full"
                                />
                            </div>
                            <div>
                                <label htmlFor="fb-email" className="text-xs font-medium text-[var(--text-subtle)] block mb-1.5">
                                    Email
                                </label>
                                <input
                                    id="fb-email"
                                    type="email"
                                    value={fbEmail}
                                    onChange={(e) => setFbEmail(e.target.value)}
                                    required
                                    className="figma-input w-full"
                                />
                            </div>
                        </div>
                        <div>
                            <label htmlFor="fb-message" className="text-xs font-medium text-[var(--text-subtle)] block mb-1.5">
                                Message
                            </label>
                            <textarea
                                id="fb-message"
                                value={fbMessage}
                                onChange={(e) => setFbMessage(e.target.value)}
                                required
                                rows={4}
                                className="figma-input w-full resize-none"
                            />
                        </div>
                        {fbStatus === 'error' && (
                            <p className="text-xs text-red-600">{fbError}</p>
                        )}
                        <FigmaButton
                            type="submit"
                            variant="primary"
                            className="figma-cta--block"
                            disabled={fbStatus === 'sending'}
                            loading={fbStatus === 'sending'}
                        >
                            Send feedback
                        </FigmaButton>
                    </form>
                )}
            </div>
        </section>
    )
}
