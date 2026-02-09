"use client";

import { useRef, useEffect } from "react";

interface TranscriptSegment {
    text: string;
    start?: number;
    end?: number;
    confidence?: number;
    timestamp?: string;
    isMedicalTerm?: boolean;
}

interface LiveTranscriptProps {
    segments: TranscriptSegment[];
    isRecording: boolean;
}

// Medical terms to highlight
const MEDICAL_TERMS = new Set([
    "blood glucose",
    "blood pressure",
    "hypertension",
    "diabetes",
    "dyspnea",
    "tachycardia",
    "pyrexia",
    "myalgia",
    "arthralgia",
    "cephalalgia",
    "dizziness",
    "nausea",
    "emesis",
    "dysuria",
    "edema",
    "pruritus",
    "fever",
    "headache",
    "chest pain",
    "abdominal pain",
    "back pain",
    "fatigue",
    "weakness",
    "shortness of breath",
    "palpitations",
    "constipation",
    "diarrhea",
    "vomiting",
]);

function highlightMedicalTerms(text: string): React.ReactNode[] {
    const lowerText = text.toLowerCase();
    const nodes: React.ReactNode[] = [];
    let lastIndex = 0;

    // Find all medical terms and their positions
    const matches: { term: string; start: number }[] = [];

    MEDICAL_TERMS.forEach((term) => {
        let index = lowerText.indexOf(term);
        while (index !== -1) {
            matches.push({ term, start: index });
            index = lowerText.indexOf(term, index + 1);
        }
    });

    // Sort by position
    matches.sort((a, b) => a.start - b.start);

    // Build nodes with highlights
    matches.forEach((match, i) => {
        const { term, start } = match;
        const end = start + term.length;

        // Skip overlapping matches
        if (start < lastIndex) return;

        // Add text before this match
        if (start > lastIndex) {
            nodes.push(text.slice(lastIndex, start));
        }

        // Add highlighted term
        nodes.push(
            <span
                key={`term-${i}`}
                className="bg-emerald-500/20 text-emerald-400 px-1 rounded font-medium"
            >
                {text.slice(start, end)}
            </span>
        );

        lastIndex = end;
    });

    // Add remaining text
    if (lastIndex < text.length) {
        nodes.push(text.slice(lastIndex));
    }

    return nodes.length > 0 ? nodes : [text];
}

export function LiveTranscript({
    segments,
    isRecording,
}: LiveTranscriptProps) {
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [segments]);

    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700/50">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
                    Live Transcript
                </h2>
                {isRecording && (
                    <span className="flex items-center gap-1.5 text-xs text-red-400">
                        <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                        Recording
                    </span>
                )}
            </div>

            {/* Transcript area */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto p-4 space-y-3 scroll-smooth"
                style={{ maxHeight: "calc(100vh - 200px)" }}
            >
                {segments.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-gray-500">
                        <div className="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center mb-3">
                            <svg
                                className="w-8 h-8"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={1.5}
                                    d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                                />
                            </svg>
                        </div>
                        <p className="text-sm">Start recording to see transcript</p>
                    </div>
                ) : (
                    segments.map((segment, index) => (
                        <div
                            key={index}
                            className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50 animate-in fade-in slide-in-from-bottom-2 duration-300"
                        >
                            <p className="text-gray-200 leading-relaxed">
                                {highlightMedicalTerms(segment.text)}
                            </p>
                            {segment.timestamp && (
                                <span className="text-xs text-gray-500 mt-1 block">
                                    {new Date(segment.timestamp).toLocaleTimeString()}
                                </span>
                            )}
                        </div>
                    ))
                )}

                {/* Recording indicator */}
                {isRecording && (
                    <div className="flex items-center gap-2 text-gray-400">
                        <div className="flex gap-1">
                            <span
                                className="w-1.5 h-4 bg-blue-500 rounded-full animate-pulse"
                                style={{ animationDelay: "0ms" }}
                            ></span>
                            <span
                                className="w-1.5 h-4 bg-blue-500 rounded-full animate-pulse"
                                style={{ animationDelay: "150ms" }}
                            ></span>
                            <span
                                className="w-1.5 h-4 bg-blue-500 rounded-full animate-pulse"
                                style={{ animationDelay: "300ms" }}
                            ></span>
                        </div>
                        <span className="text-xs">Listening...</span>
                    </div>
                )}
            </div>
        </div>
    );
}
