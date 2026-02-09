"use client";

import { useState, useRef, useCallback, useEffect } from "react";

type MessageType =
    | "transcript"
    | "thought"
    | "soap"
    | "codes"
    | "status"
    | "error"
    | "complete";

interface WebSocketMessage {
    type: MessageType;
    data: unknown;
}

interface UseWebSocketReturn {
    isConnected: boolean;
    connect: () => void;
    disconnect: () => void;
    sendAudio: (chunk: Blob) => Promise<void>;
    sendAction: (action: string, data?: Record<string, unknown>) => void;
    messages: WebSocketMessage[];
    error: string | null;
}

export function useWebSocket(url: string): UseWebSocketReturn {
    const [isConnected, setIsConnected] = useState(false);
    const [messages, setMessages] = useState<WebSocketMessage[]>([]);
    const [error, setError] = useState<string | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        try {
            const ws = new WebSocket(url);

            ws.onopen = () => {
                console.log("WebSocket connected");
                setIsConnected(true);
                setError(null);
            };

            ws.onclose = () => {
                console.log("WebSocket disconnected");
                setIsConnected(false);
            };

            ws.onerror = (event) => {
                console.error("WebSocket error:", event);
                setError("Connection error");
                setIsConnected(false);
            };

            ws.onmessage = (event) => {
                try {
                    const message: WebSocketMessage = JSON.parse(event.data);
                    setMessages((prev) => [...prev, message]);
                } catch (e) {
                    console.error("Failed to parse message:", e);
                }
            };

            wsRef.current = ws;
        } catch (e) {
            console.error("Failed to connect:", e);
            setError("Failed to connect to server");
        }
    }, [url]);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }

        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        setIsConnected(false);
    }, []);

    const sendAudio = useCallback(async (chunk: Blob) => {
        if (wsRef.current?.readyState !== WebSocket.OPEN) {
            console.warn("WebSocket not connected");
            return;
        }

        try {
            const arrayBuffer = await chunk.arrayBuffer();
            wsRef.current.send(arrayBuffer);
        } catch (e) {
            console.error("Failed to send audio:", e);
        }
    }, []);

    const sendAction = useCallback(
        (action: string, data?: Record<string, unknown>) => {
            if (wsRef.current?.readyState !== WebSocket.OPEN) {
                console.warn("WebSocket not connected");
                return;
            }

            wsRef.current.send(JSON.stringify({ action, ...data }));
        },
        []
    );

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            disconnect();
        };
    }, [disconnect]);

    return {
        isConnected,
        connect,
        disconnect,
        sendAudio,
        sendAction,
        messages,
        error,
    };
}
