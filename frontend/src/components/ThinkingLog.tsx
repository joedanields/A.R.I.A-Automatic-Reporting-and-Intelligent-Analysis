"use client";

import { useRef, useEffect } from "react";
import { Terminal, Brain, Search, CheckCircle } from "lucide-react";

interface ThoughtEntry {
    message: string;
    timestamp: Date;
    type?: "info" | "success" | "warning" | "processing";
    agent?: string;
}

interface ThinkingLogProps {
    thoughts: ThoughtEntry[];
    isProcessing: boolean;
}

function getAgentIcon(agent?: string) {
    switch (agent?.toLowerCase()) {
        case "scribe":
            return <Terminal className="w-3.5 h-3.5" />;
        case "coder":
            return <Search className="w-3.5 h-3.5" />;
        case "auditor":
            return <CheckCircle className="w-3.5 h-3.5" />;
        default:
            return <Brain className="w-3.5 h-3.5" />;
    }
}

function getTypeColor(type?: string) {
    switch (type) {
        case "success":
            return "text-emerald-400";
        case "warning":
            return "text-amber-400";
        case "processing":
            return "text-blue-400";
        default:
            return "text-gray-400";
    }
}

export function ThinkingLog({ thoughts, isProcessing }: ThinkingLogProps) {
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [thoughts]);

    return (
        <div className="bg-gray-900/80 rounded-xl border border-gray-700/50 overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-700/50 bg-gray-800/50">
                <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                    <Brain className="w-4 h-4 text-purple-400" />
                    A.R.I.A. Thinking
                </h3>
                {isProcessing && (
                    <div className="flex items-center gap-1.5">
                        <div className="w-1.5 h-1.5 bg-purple-500 rounded-full animate-ping"></div>
                        <span className="text-xs text-purple-400">Processing</span>
                    </div>
                )}
            </div>

            {/* Log area - terminal style */}
            <div
                ref={scrollRef}
                className="font-mono text-xs overflow-y-auto bg-gray-950/50 p-3 space-y-1.5"
                style={{ maxHeight: "180px" }}
            >
                {thoughts.length === 0 ? (
                    <div className="text-gray-600 italic">
                        Waiting for transcript to process...
                    </div>
                ) : (
                    thoughts.map((thought, index) => (
                        <div
                            key={index}
                            className={`flex items-start gap-2 ${getTypeColor(thought.type)} animate-in fade-in duration-200`}
                        >
                            <span className="text-gray-600 shrink-0">
                                [{thought.timestamp.toLocaleTimeString()}]
                            </span>
                            <span className="shrink-0 opacity-70">
                                {getAgentIcon(thought.agent)}
                            </span>
                            <span className="break-words">{thought.message}</span>
                        </div>
                    ))
                )}

                {/* Processing animation */}
                {isProcessing && (
                    <div className="flex items-center gap-2 text-purple-400">
                        <span className="text-gray-600">
                            [{new Date().toLocaleTimeString()}]
                        </span>
                        <div className="flex gap-0.5">
                            <span
                                className="w-1 h-1 bg-purple-500 rounded-full animate-bounce"
                                style={{ animationDelay: "0ms" }}
                            />
                            <span
                                className="w-1 h-1 bg-purple-500 rounded-full animate-bounce"
                                style={{ animationDelay: "100ms" }}
                            />
                            <span
                                className="w-1 h-1 bg-purple-500 rounded-full animate-bounce"
                                style={{ animationDelay: "200ms" }}
                            />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
