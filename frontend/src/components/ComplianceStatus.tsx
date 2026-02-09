"use client";

import { Shield, ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";

interface ComplianceStatusProps {
    isCompliant: boolean;
    missingFields: string[];
    isChecking: boolean;
}

export function ComplianceStatus({
    isCompliant,
    missingFields,
    isChecking,
}: ComplianceStatusProps) {
    return (
        <div className="bg-gray-900/80 rounded-xl border border-gray-700/50 overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-700/50 bg-gray-800/50">
                <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                    <Shield className="w-4 h-4 text-indigo-400" />
                    ABDM Compliance
                </h3>
            </div>

            {/* Status indicator */}
            <div className="p-4">
                {isChecking ? (
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center">
                            <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                        </div>
                        <div>
                            <p className="text-sm font-medium text-gray-300">
                                Checking compliance...
                            </p>
                            <p className="text-xs text-gray-500">
                                Validating FHIR OPConsultRecord
                            </p>
                        </div>
                    </div>
                ) : isCompliant ? (
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
                            <ShieldCheck className="w-5 h-5 text-emerald-400" />
                        </div>
                        <div>
                            <p className="text-sm font-medium text-emerald-400">ABDM Ready</p>
                            <p className="text-xs text-gray-500">
                                All mandatory fields present
                            </p>
                        </div>
                    </div>
                ) : (
                    <div className="space-y-3">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center">
                                <ShieldAlert className="w-5 h-5 text-amber-400" />
                            </div>
                            <div>
                                <p className="text-sm font-medium text-amber-400">
                                    Incomplete Record
                                </p>
                                <p className="text-xs text-gray-500">
                                    Missing required FHIR fields
                                </p>
                            </div>
                        </div>

                        {/* Missing fields list */}
                        {missingFields.length > 0 && (
                            <div className="bg-amber-500/10 rounded-lg p-3 border border-amber-500/20">
                                <div className="flex items-center gap-1.5 mb-2">
                                    <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                                    <span className="text-xs font-medium text-amber-400">
                                        Missing Fields:
                                    </span>
                                </div>
                                <ul className="space-y-1">
                                    {missingFields.map((field, index) => (
                                        <li
                                            key={index}
                                            className="text-xs text-gray-400 flex items-center gap-1.5"
                                        >
                                            <span className="w-1 h-1 rounded-full bg-amber-500" />
                                            {field.replace(/_/g, " ")}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                )}

                {/* FHIR link */}
                <div className="mt-3 pt-3 border-t border-gray-800">
                    <a
                        href="https://nrces.in/ndhm/fhir/r4/StructureDefinition-OPConsultRecord.html"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
                    >
                        View FHIR OPConsultRecord Schema →
                    </a>
                </div>
            </div>
        </div>
    );
}
