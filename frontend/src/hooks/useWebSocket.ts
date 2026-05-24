import { useEffect, useRef, useCallback, useState } from 'react';
import type { WsMessage } from '../types';

interface UseWebSocketOptions {
  onMessage: (msg: WsMessage) => void;
  onClose?: () => void;
  onError?: (err: Event) => void;
}

export function useWebSocket(url: string | null, options: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  const close = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setConnected(false);
    }
  }, []);

  useEffect(() => {
    if (!url) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WsMessage;
        options.onMessage(msg);
      } catch {
        console.error('[WS] Failed to parse message:', event.data);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      options.onClose?.();
    };

    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
      options.onError?.(err);
    };

    return () => {
      ws.close();
      wsRef.current = null;
      setConnected(false);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  const send = useCallback((data: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      console.warn('[WS] Cannot send message, socket is not open');
    }
  }, []);

  return { connected, close, send };
}
