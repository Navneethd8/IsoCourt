import React from 'react';

const Logo = ({ size = 24, className = "" }) => {
    return (
        <svg
            width={size}
            height={size}
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className={className}
        >
            <path
                d="M19.52 2.49C17.18.15 12.9.62 9.97 3.55c-1.6 1.6-2.52 3.87-2.54 5.46-.02 1.58.26 3.89-1.35 5.5l-3.54 3.53c-.39.39-.39 1.02 0 1.42.39.39 1.02.39 1.42 0l3.53-3.54c1.61-1.61 3.92-1.33 5.5-1.35s3.86-.94 5.46-2.54c2.93-2.92 3.41-7.2 1.07-9.54m-9.2 9.19c-1.53-1.53-1.05-4.61 1.06-6.72s5.18-2.59 6.72-1.06c1.53 1.53 1.05 4.61-1.06 6.72s-5.18 2.59-6.72 1.06"
                fill="currentColor"
            />

            <g transform="translate(14, 15) scale(0.35) rotate(150 12 12)">
                <path d="M8 18.5c0 2.21 1.79 4 4 4s4-1.79 4-4H8z" fill="currentColor" />

                <rect x="8" y="16.5" width="8" height="2" rx="0.5" fill="currentColor" />

                <path d="M4 4.5c0-1.5 2.66-1.5 2.66 0c0-1.5 5.34-1.5 5.34 0c0-1.5 5.34-1.5 5.34 0c0-1.5 2.66-1.5 2.66 0L16 16.5H8L4 4.5z" fill="currentColor" />

                <path d="M6.5 12.5h11 M9.33 4.5L10.5 16.5 M14.66 4.5L13.5 16.5" stroke="black" stroke-width="0.5" stroke-linecap="round" />
            </g>
        </svg>
    );
};

export default Logo;
