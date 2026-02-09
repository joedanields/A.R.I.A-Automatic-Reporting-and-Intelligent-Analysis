"use client";

import { FileText, Stethoscope, ClipboardList, PillIcon } from "lucide-react";

interface SOAPSection {
    title: string;
    text: string;
    codes?: Array<{ code: string; description: string }>;
}

interface SOAPNoteData {
    resourceType?: string;
    type?: { text: string };
    encounter?: { date: string };
    section?: SOAPSection[];
}

interface SoapNoteProps {
    data: SOAPNoteData | null;
    icdCodes: Array<{ code: string; description: string; confidence?: string }>;
    isLoading: boolean;
}

function getSectionIcon(title: string) {
    switch (title.toLowerCase()) {
        case "subjective":
            return <Stethoscope className="w-4 h-4" />;
        case "objective":
            return <FileText className="w-4 h-4" />;
        case "assessment":
            return <ClipboardList className="w-4 h-4" />;
        case "plan":
            return <PillIcon className="w-4 h-4" />;
        default:
            return <FileText className="w-4 h-4" />;
    }
}

function getSectionColor(title: string) {
    switch (title.toLowerCase()) {
        case "subjective":
            return "border-blue-500/30 bg-blue-500/5";
        case "objective":
            return "border-purple-500/30 bg-purple-500/5";
        case "assessment":
            return "border-amber-500/30 bg-amber-500/5";
        case "plan":
            return "border-emerald-500/30 bg-emerald-500/5";
        default:
            return "border-gray-500/30 bg-gray-500/5";
    }
}

export function SoapNote({ data, icdCodes, isLoading }: SoapNoteProps) {
    return (
        <div className="bg-gray-900/80 rounded-xl border border-gray-700/50 overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-700/50 bg-gray-800/50">
                <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                    <FileText className="w-4 h-4 text-cyan-400" />
                    SOAP Note
                </h3>
                {data?.encounter?.date && (
                    <span className="text-xs text-gray-500">
                        {new Date(data.encounter.date).toLocaleDateString()}
                    </span>
                )}
            </div>

            {/* Content */}
            <div className="p-4 space-y-3">
                {isLoading ? (
                    <div className="space-y-3">
                        {["Subjective", "Objective", "Assessment", "Plan"].map(
                            (section) => (
                                <div
                                    key={section}
                                    className="h-16 bg-gray-800/50 rounded-lg animate-pulse"
                                />
                            )
                        )}
                    </div>
                ) : data?.section ? (
                    <>
                        {/* SOAP Sections */}
                        {data.section.map((section, index) => (
                            <div
                                key={index}
                                className={`rounded-lg border p-3 ${getSectionColor(section.title)}`}
                            >
                                <div className="flex items-center gap-2 mb-2">
                                    <span className="text-gray-400">
                                        {getSectionIcon(section.title)}
                                    </span>
                                    <h4 className="text-sm font-medium text-gray-200">
                                        {section.title}
                                    </h4>
                                </div>
                                <p className="text-sm text-gray-400 leading-relaxed">
                                    {section.text || "Pending..."}
                                </p>
                            </div>
                        ))}

                        {/* ICD-10 Codes */}
                        {icdCodes.length > 0 && (
                            <div className="mt-4">
                                <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
                                    ICD-10 Codes
                                </h4>
                                <div className="flex flex-wrap gap-2">
                                    {icdCodes.map((code, index) => (
                                        <div
                                            key={index}
                                            className="flex items-center gap-1.5 bg-gray-800 rounded-full px-2.5 py-1 border border-gray-700"
                                            title={code.description}
                                        >
                                            <span className="text-xs font-mono text-cyan-400">
                                                {code.code}
                                            </span>
                                            {code.confidence && (
                                                <span
                                                    className={`text-[10px] px-1 rounded ${code.confidence === "high"
                                                            ? "bg-emerald-500/20 text-emerald-400"
                                                            : code.confidence === "medium"
                                                                ? "bg-amber-500/20 text-amber-400"
                                                                : "bg-gray-500/20 text-gray-400"
                                                        }`}
                                                >
                                                    {code.confidence}
                                                </span>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </>
                ) : (
                    <div className="text-center py-8 text-gray-500">
                        <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
                        <p className="text-sm">SOAP note will appear after analysis</p>
                    </div>
                )}
            </div>
        </div>
    );
}
