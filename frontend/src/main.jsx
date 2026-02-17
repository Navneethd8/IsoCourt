import React from 'react'
import ReactDOM from 'react-dom/client'
import ReactGA from "react-ga4";
import App from './App.jsx'
import './index.css'

ReactGA.initialize("G-TET6JN36Q4"); // TODO: Replace with your Measurement ID

ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>,
)
