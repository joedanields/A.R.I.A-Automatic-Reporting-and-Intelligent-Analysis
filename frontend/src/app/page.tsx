"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { Mic, MicOff, Sparkles, Activity, Wifi, WifiOff, Upload, FileText } from "lucide-react";

import { useAudioCapture } from "@/hooks/useAudioCapture";
import { useWebSocket } from "@/hooks/useWebSocket";
import { LiveTranscript } from "@/components/LiveTranscript";
import { ThinkingLog } from "@/components/ThinkingLog";
import { SoapNote } from "@/components/SoapNote";
import { ComplianceStatus } from "@/components/ComplianceStatus";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/listen";

interface TranscriptSegment {
  text: string;
  timestamp?: string;
}

interface ThoughtEntry {
  message: string;
  timestamp: Date;
  type?: "info" | "success" | "warning" | "processing";
  agent?: string;
}

interface IcdCode {
  code: string;
  description: string;
  confidence?: string;
}

export default function Home() {
  // WebSocket connection
  const {
    isConnected,
    connect,
    disconnect,
    sendAudio,
    sendAction,
    messages,
    error: wsError,
  } = useWebSocket(WS_URL);

  // State
  const [transcriptSegments, setTranscriptSegments] = useState<TranscriptSegment[]>([]);
  const [thoughts, setThoughts] = useState<ThoughtEntry[]>([]);
  const [soapNote, setSoapNote] = useState<Record<string, unknown> | null>(null);
  const [icdCodes, setIcdCodes] = useState<IcdCode[]>([]);
  const [isCompliant, setIsCompliant] = useState(false);
  const [missingFields, setMissingFields] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  // File upload ref
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Audio capture with WebSocket streaming
  const handleAudioChunk = useCallback(
    async (chunk: Blob) => {
      if (isConnected) {
        await sendAudio(chunk);
      }
    },
    [isConnected, sendAudio]
  );

  const {
    isRecording,
    isSupported,
    startRecording,
    stopRecording,
    error: audioError,
  } = useAudioCapture(handleAudioChunk, { chunkDuration: 2000 });

  // Process incoming WebSocket messages
  useEffect(() => {
    if (messages.length === 0) return;

    const latestMessage = messages[messages.length - 1];

    switch (latestMessage.type) {
      case "transcript":
        const transcriptData = latestMessage.data as {
          text: string;
          timestamp: string;
        };
        setTranscriptSegments((prev) => [
          ...prev,
          { text: transcriptData.text, timestamp: transcriptData.timestamp },
        ]);
        break;

      case "thought":
        const thoughtText = latestMessage.data as string;
        // Extract agent name from thought
        let agent = "system";
        if (thoughtText.toLowerCase().includes("scribe")) agent = "scribe";
        else if (thoughtText.toLowerCase().includes("coder")) agent = "coder";
        else if (thoughtText.toLowerCase().includes("auditor")) agent = "auditor";

        setThoughts((prev) => [
          ...prev,
          {
            message: thoughtText,
            timestamp: new Date(),
            type: "processing",
            agent,
          },
        ]);
        break;

      case "soap":
        setSoapNote(latestMessage.data as Record<string, unknown>);
        break;

      case "codes":
        setIcdCodes(latestMessage.data as IcdCode[]);
        break;

      case "complete":
        setIsProcessing(false);
        setThoughts((prev) => [
          ...prev,
          {
            message: "Analysis complete",
            timestamp: new Date(),
            type: "success",
            agent: "system",
          },
        ]);
        // Check compliance based on SOAP note
        if (soapNote) {
          const sections = (soapNote.section as unknown[]) || [];
          setIsCompliant(sections.length >= 4 && icdCodes.length > 0);
        }
        break;

      case "error":
        setThoughts((prev) => [
          ...prev,
          {
            message: latestMessage.data as string,
            timestamp: new Date(),
            type: "warning",
            agent: "system",
          },
        ]);
        setIsProcessing(false);
        break;
    }
  }, [messages, soapNote, icdCodes.length]);

  // Handle recording toggle
  const handleRecordingToggle = async () => {
    if (isRecording) {
      stopRecording();
      sendAction("stop");
      // Start processing after stopping
      setTimeout(() => {
        setIsProcessing(true);
        sendAction("process");
      }, 500);
    } else {
      if (!isConnected) {
        connect();
      }
      // Reset state for new session
      setTranscriptSegments([]);
      setThoughts([]);
      setSoapNote(null);
      setIcdCodes([]);
      setIsCompliant(false);
      setMissingFields([]);

      sendAction("start");
      await startRecording();
    }
  };

  // Demo mode - test with sample text
  const handleDemoMode = () => {
    if (!isConnected) {
      connect();
    }
    setIsProcessing(true);
    setThoughts([
      {
        message: "Starting demo analysis...",
        timestamp: new Date(),
        type: "info",
        agent: "system",
      },
    ]);

    setTimeout(() => {
      sendAction("test", {
        text: "Patient is a 45 year old male presenting with high sugars and elevated BP. He reports chakkar and occasional headache for the past week. Has history of diabetes for 5 years.",
      });
    }, 300);
  };

  // Handle file upload
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();

      if (!isConnected) {
        connect();
      }

      // Reset state
      setTranscriptSegments([{ text, timestamp: new Date().toISOString() }]);
      setThoughts([
        {
          message: `Uploaded file: ${file.name}`,
          timestamp: new Date(),
          type: "info",
          agent: "system",
        },
      ]);
      setSoapNote(null);
      setIcdCodes([]);
      setIsCompliant(false);
      setMissingFields([]);
      setIsProcessing(true);

      // Process the uploaded text
      setTimeout(() => {
        sendAction("test", { text });
      }, 300);

      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      console.error("Failed to read file:", err);
    }
  };

  // Connection status
  useEffect(() => {
    // Auto-connect on mount
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  const error = wsError || audioError;

  return (
    <main className="min-h-screen bg-gradient-animated">
      {/* Header */}
      <header className="glass-card border-b border-gray-700/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center glow-purple">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                A.R.I.A.
              </h1>
              <p className="text-xs text-gray-500">
                Automatic Reporting & Intelligent Analysis
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Connection status */}
            <div className="flex items-center gap-2 text-sm">
              {isConnected ? (
                <>
                  <Wifi className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-400">Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-4 h-4 text-gray-500" />
                  <span className="text-gray-500">Disconnected</span>
                </>
              )}
            </div>

            {/* Demo button */}
            <button
              onClick={handleDemoMode}
              disabled={isProcessing || isRecording}
              className="px-3 py-1.5 text-sm bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg transition-colors disabled:opacity-50"
            >
              Demo Mode
            </button>

            {/* Upload button */}
            <label className="cursor-pointer">
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.text"
                onChange={handleFileUpload}
                className="hidden"
                disabled={isProcessing || isRecording}
              />
              <span
                className={`flex items-center gap-2 px-3 py-1.5 text-sm bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg transition-all ${isProcessing || isRecording ? "opacity-50 cursor-not-allowed" : ""
                  }`}
              >
                <Upload className="w-4 h-4" />
                Upload Transcript
              </span>
            </label>

            {/* Record button */}
            <button
              onClick={handleRecordingToggle}
              disabled={!isSupported || isProcessing}
              className={`flex items-center gap-2 ${isRecording ? "btn-danger recording-pulse" : "btn-primary"
                }`}
            >
              {isRecording ? (
                <>
                  <MicOff className="w-4 h-4" />
                  Stop Recording
                </>
              ) : (
                <>
                  <Mic className="w-4 h-4" />
                  Start Recording
                </>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="bg-red-500/20 border-b border-red-500/30 px-4 py-2 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Main content - Split screen */}
      <div className="max-w-7xl mx-auto p-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left Panel - Live Transcript */}
        <div className="glass-card rounded-2xl overflow-hidden h-[calc(100vh-140px)]">
          <LiveTranscript
            segments={transcriptSegments}
            isRecording={isRecording}
          />
        </div>

        {/* Right Panel - A.R.I.A. Intelligence */}
        <div className="space-y-4 h-[calc(100vh-140px)] overflow-y-auto pr-2">
          {/* Section A: Live Thinking */}
          <ThinkingLog thoughts={thoughts} isProcessing={isProcessing} />

          {/* Section B: SOAP Note */}
          <SoapNote
            data={soapNote}
            icdCodes={icdCodes}
            isLoading={isProcessing && thoughts.length > 0}
          />

          {/* Section C: Compliance Status */}
          <ComplianceStatus
            isCompliant={isCompliant}
            missingFields={missingFields}
            isChecking={isProcessing}
          />

          {/* Processing indicator */}
          {isProcessing && (
            <div className="glass-card rounded-xl p-4 flex items-center justify-center gap-3">
              <Activity className="w-5 h-5 text-purple-400 animate-pulse" />
              <span className="text-sm text-gray-400">
                AI agents processing...
              </span>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
