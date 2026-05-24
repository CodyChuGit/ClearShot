import { useEffect, useRef, useCallback, useState } from 'react';
import type { WsMessage } from '../types';

interface UseWebSocketOptions {
  onMessage: (msg: WsMessage) => void;
  onClose?: () => void;
  onError?: (err: Event) => void;
}

export function useWebSocket(url: string | null, options: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const pendingMessagesRef = useRef<unknown[]>([]);
  const [connected, setConnected] = useState(false);
  const [reconnectToken, setReconnectToken] = useState(0);

  const close = useCallback(() => {
    pendingMessagesRef.current = [];
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setConnected(false);
    }
  }, []);

  useEffect(() => {
    if (!url) {
      pendingMessagesRef.current = [];
      return;
    }

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      while (pendingMessagesRef.current.length > 0 && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(pendingMessagesRef.current.shift()));
      }
    };

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
  }, [url, reconnectToken]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else if (wsRef.current && wsRef.current.readyState === WebSocket.CONNECTING) {
      pendingMessagesRef.current.push(data);
    } else if (url) {
      pendingMessagesRef.current.push(data);
      setReconnectToken((token) => token + 1);
    } else {
      console.warn('[WS] Cannot send message, socket is not open');
    }
  }, [url]);

  return { connected, close, send };
}
