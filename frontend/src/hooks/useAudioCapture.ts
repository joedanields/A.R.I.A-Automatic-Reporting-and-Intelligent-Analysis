"use client";

import { useState, useRef, useCallback, useEffect } from "react";

interface AudioCaptureOptions {
  chunkDuration?: number; // ms
  mimeType?: string;
}

interface AudioCaptureState {
  isRecording: boolean;
  isSupported: boolean;
  error: string | null;
}

interface UseAudioCaptureReturn extends AudioCaptureState {
  startRecording: () => Promise<void>;
  stopRecording: () => void;
}

export function useAudioCapture(
  onAudioChunk: (chunk: Blob) => void,
  options: AudioCaptureOptions = {}
): UseAudioCaptureReturn {
  const { chunkDuration = 2000, mimeType = "audio/webm;codecs=opus" } = options;

  const [state, setState] = useState<AudioCaptureState>({
    isRecording: false,
    isSupported: false, // Initialize false to avoid SSR hydration mismatch
    error: null,
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // Check browser support
  useEffect(() => {
    if (typeof window !== "undefined") {
      setState((prev) => ({
        ...prev,
        isSupported: !!navigator.mediaDevices?.getUserMedia,
      }));
    }
  }, []);

  const startRecording = useCallback(async () => {
    if (!state.isSupported) {
      setState((prev) => ({
        ...prev,
        error: "Audio recording not supported in this browser",
      }));
      return;
    }

    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      });

      streamRef.current = stream;
      chunksRef.current = [];

      // Determine supported MIME type
      let selectedMimeType = mimeType;
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        // Fallback options
        const fallbacks = [
          "audio/webm",
          "audio/ogg;codecs=opus",
          "audio/mp4",
          "",
        ];
        selectedMimeType =
          fallbacks.find((type) => MediaRecorder.isTypeSupported(type)) || "";
      }

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: selectedMimeType,
        audioBitsPerSecond: 128000,
      });

      mediaRecorderRef.current = mediaRecorder;

      // Handle data available
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
          onAudioChunk(event.data);
        }
      };

      // Handle errors
      mediaRecorder.onerror = (event) => {
        console.error("MediaRecorder error:", event);
        setState((prev) => ({
          ...prev,
          error: "Recording error occurred",
          isRecording: false,
        }));
      };

      // Handle stop
      mediaRecorder.onstop = () => {
        setState((prev) => ({ ...prev, isRecording: false }));
      };

      // Start recording with timeslice for chunked data
      mediaRecorder.start(chunkDuration);

      setState((prev) => ({
        ...prev,
        isRecording: true,
        error: null,
      }));
    } catch (err) {
      console.error("Failed to start recording:", err);
      setState((prev) => ({
        ...prev,
        error:
          err instanceof Error ? err.message : "Failed to access microphone",
        isRecording: false,
      }));
    }
  }, [state.isSupported, mimeType, chunkDuration, onAudioChunk]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && state.isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    setState((prev) => ({ ...prev, isRecording: false }));
  }, [state.isRecording]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return {
    ...state,
    startRecording,
    stopRecording,
  };
}
