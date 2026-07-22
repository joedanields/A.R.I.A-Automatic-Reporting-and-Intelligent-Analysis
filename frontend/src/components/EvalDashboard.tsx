"use client";

import React, { useState, useEffect, useCallback } from "react";

interface CaseResult {
  case_id: string;
  description: string;
  passed: boolean;
  wer: number;
  entity_f1: number;
  code_accuracy: number;
  soap_similarity: number;
  error: string | null;
}

interface EvalResult {
  success: boolean;
  run_id: string;
  timestamp: string;
  total_cases: number;
  passed_cases: number;
  metrics: {
    avg_wer: number;
    avg_entity_f1: number;
    avg_code_accuracy: number;
    avg_soap_similarity: number;
  };
  duration_seconds: number;
  cases: CaseResult[];
}

interface EvalHistory {
  run_id: string;
  timestamp: string;
  total_cases: number;
  passed_cases: number;
  avg_wer: number;
  avg_entity_f1: number;
  avg_code_accuracy: number;
  duration_seconds: number;
}

function MetricCard({ label, value, icon }: { label: string; value: string; icon: string }) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-lg">{icon}</span>
        <span className="text-sm font-medium text-gray-500">{label}</span>
      </div>
      <div className="text-2xl font-bold text-gray-900">{value}</div>
    </div>
  );
}

function CaseRow({ case_result }: { case_result: CaseResult }) {
  const statusColor = case_result.passed
    ? "bg-green-100 text-green-800"
    : "bg-red-100 text-red-800";

  return (
    <tr className="hover:bg-gray-50">
      <td className="px-4 py-3">
        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusColor}`}>
          {case_result.passed ? "PASS" : "FAIL"}
        </span>
      </td>
      <td className="px-4 py-3 text-sm font-medium text-gray-900">{case_result.case_id}</td>
      <td className="px-4 py-3 text-sm text-gray-500">{case_result.description}</td>
      <td className="px-4 py-3 text-sm text-gray-900">{(case_result.wer * 100).toFixed(1)}%</td>
      <td className="px-4 py-3 text-sm text-gray-900">{(case_result.entity_f1 * 100).toFixed(1)}%</td>
      <td className="px-4 py-3 text-sm text-gray-900">{(case_result.code_accuracy * 100).toFixed(1)}%</td>
      <td className="px-4 py-3 text-sm text-gray-900">{(case_result.soap_similarity * 100).toFixed(1)}%</td>
      {case_result.error && (
        <td className="px-4 py-3 text-sm text-red-600">{case_result.error}</td>
      )}
    </tr>
  );
}

function HistoryRow({ entry }: { entry: EvalHistory }) {
  return (
    <tr className="hover:bg-gray-50">
      <td className="px-4 py-3 text-sm text-gray-900">{entry.run_id}</td>
      <td className="px-4 py-3 text-sm text-gray-500">{new Date(entry.timestamp).toLocaleString()}</td>
      <td className="px-4 py-3 text-sm text-gray-900">{entry.passed_cases}/{entry.total_cases}</td>
      <td className="px-4 py-3 text-sm text-gray-900">{(entry.avg_wer * 100).toFixed(1)}%</td>
      <td className="px-4 py-3 text-sm text-gray-900">{(entry.avg_entity_f1 * 100).toFixed(1)}%</td>
      <td className="px-4 py-3 text-sm text-gray-900">{(entry.avg_code_accuracy * 100).toFixed(1)}%</td>
      <td className="px-4 py-3 text-sm text-gray-500">{entry.duration_seconds.toFixed(1)}s</td>
    </tr>
  );
}

export default function EvalDashboard() {
  const [currentResult, setCurrentResult] = useState<EvalResult | null>(null);
  const [history, setHistory] = useState<EvalHistory[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"current" | "history">("current");

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch("/api/eval/history");
      const data = await res.json();
      if (data.success) {
        setHistory(data.history);
      }
    } catch (e) {
      console.error("Failed to fetch history:", e);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const runEval = async () => {
    setIsRunning(true);
    setError(null);

    try {
      const res = await fetch("/api/eval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });

      const data = await res.json();
      if (data.success) {
        setCurrentResult(data);
        fetchHistory(); // Refresh history
      } else {
        setError(data.detail || "Evaluation failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Network error");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-gray-900">Evaluation Dashboard</h1>
          <p className="mt-1 text-sm text-gray-500">
            Measure WER, entity F1, code accuracy, and SOAP similarity against gold test cases.
          </p>
        </div>

        {/* Run Button */}
        <div className="mb-6">
          <button
            onClick={runEval}
            disabled={isRunning}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRunning ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Running...
              </>
            ) : (
              "Run Evaluation"
            )}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-md p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Evaluation Error</h3>
                <p className="mt-1 text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Metrics Cards */}
        {currentResult && (
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4 mb-6">
            <MetricCard
              icon="📝"
              label="Word Error Rate"
              value={`${(currentResult.metrics.avg_wer * 100).toFixed(1)}%`}
            />
            <MetricCard
              icon="🏷️"
              label="Entity F1 Score"
              value={`${(currentResult.metrics.avg_entity_f1 * 100).toFixed(1)}%`}
            />
            <MetricCard
              icon="💊"
              label="Code Accuracy"
              value={`${(currentResult.metrics.avg_code_accuracy * 100).toFixed(1)}%`}
            />
            <MetricCard
              icon="📋"
              label="SOAP Similarity"
              value={`${(currentResult.metrics.avg_soap_similarity * 100).toFixed(1)}%`}
            />
          </div>
        )}

        {/* Tabs */}
        <div className="border-b border-gray-200 mb-6">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab("current")}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === "current"
                  ? "border-indigo-500 text-indigo-600"
                  : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
              }`}
            >
              Current Run
            </button>
            <button
              onClick={() => setActiveTab("history")}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === "history"
                  ? "border-indigo-500 text-indigo-600"
                  : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
              }`}
            >
              History
            </button>
          </nav>
        </div>

        {/* Current Run Tab */}
        {activeTab === "current" && currentResult && (
          <div className="bg-white shadow rounded-lg overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900">
                Run {currentResult.run_id} — {currentResult.passed_cases}/{currentResult.total_cases} passed
              </h3>
              <p className="text-sm text-gray-500">
                Duration: {currentResult.duration_seconds.toFixed(1)}s
              </p>
            </div>
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Case</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Description</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">WER</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Entity F1</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Code Acc</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">SOAP Sim</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {currentResult.cases.map((cr) => (
                  <CaseRow key={cr.case_id} case_result={cr} />
                ))}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === "current" && !currentResult && (
          <div className="bg-white shadow rounded-lg p-8 text-center">
            <p className="text-gray-500">No evaluation results yet. Click "Run Evaluation" to start.</p>
          </div>
        )}

        {/* History Tab */}
        {activeTab === "history" && (
          <div className="bg-white shadow rounded-lg overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Run ID</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Timestamp</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Passed</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">WER</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Entity F1</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Code Acc</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Duration</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {history.map((entry) => (
                  <HistoryRow key={entry.run_id} entry={entry} />
                ))}
                {history.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                      No evaluation history yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
