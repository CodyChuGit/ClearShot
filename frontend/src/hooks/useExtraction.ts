import { useState, useCallback } from 'react';
import type { ExtractionSettings, ExtractionStats, VideoMeta, WsMessage } from '../types';
import { DEFAULT_SETTINGS } from '../types';
import { uploadVideo, downloadUrl, getWsUrl } from '../services/api';
import { useWebSocket } from './useWebSocket';

export type ExtractionPhase = 'idle' | 'uploading' | 'ready' | 'downloading' | 'downloaded' | 'extracting' | 'complete' | 'error';

interface ExtractionState {
  phase: ExtractionPhase;
  jobId: string | null;
  videoMeta: VideoMeta | null;
  settings: ExtractionSettings;
  progress: number;
  stage: string;
  stats: ExtractionStats | null;
  results: string[];
  error: string | null;
  wsUrl: string | null;
}

export function useExtraction() {
  const [state, setState] = useState<ExtractionState>({
    phase: 'idle',
    jobId: null,
    videoMeta: null,
    settings: { ...DEFAULT_SETTINGS },
    progress: 0,
    stage: '',
    stats: null,
    results: [],
    error: null,
    wsUrl: null,
  });

  const handleWsMessage = useCallback((msg: WsMessage) => {
    if (msg.type === 'progress') {
      setState((s) => ({
        ...s,
        progress: msg.progress,
        stage: msg.stage,
      }));
    } else if (msg.type === 'download_complete') {
      setState((s) => ({
        ...s,
        phase: 'downloaded',
        progress: 1,
        stage: 'downloaded',
        videoMeta: { ...s.videoMeta!, is_url: false }, // After download, treat it as a local file for extraction UI
      }));
    } else if (msg.type === 'complete') {
      setState((s) => ({
        ...s,
        phase: 'complete',
        progress: 1,
        stage: 'complete',
        stats: msg.stats,
        results: msg.results,
        wsUrl: null,
      }));
    } else if (msg.type === 'error') {
      setState((s) => ({
        ...s,
        phase: 'error',
        error: msg.error,
        wsUrl: null,
      }));
    }
  }, []);

  const { send } = useWebSocket(state.wsUrl, {
    onMessage: handleWsMessage,
    onError: () => {
      setState((s) => ({
        ...s,
        phase: 'error',
        error: 'WebSocket connection lost',
        wsUrl: null,
      }));
    },
  });

  const upload = useCallback(async (file: File) => {
    setState((s) => ({ ...s, phase: 'uploading', error: null }));
    try {
      const res = await uploadVideo(file);
      setState((s) => ({
        ...s,
        phase: 'ready',
        jobId: res.job_id,
        videoMeta: res.meta,
        wsUrl: getWsUrl(res.job_id),
      }));
    } catch (err) {
      setState((s) => ({
        ...s,
        phase: 'error',
        error: err instanceof Error ? err.message : 'Upload failed',
      }));
    }
  }, []);

  const importUrl = useCallback(async (url: string) => {
    setState((s) => ({ ...s, phase: 'uploading', error: null }));
    try {
      const res = await downloadUrl(url);
      setState((s) => ({
        ...s,
        phase: 'ready',
        jobId: res.job_id,
        videoMeta: { ...res.meta, is_url: true },
        wsUrl: getWsUrl(res.job_id),
      }));
    } catch (err) {
      setState((s) => ({
        ...s,
        phase: 'error',
        error: err instanceof Error ? err.message : 'URL import failed',
      }));
    }
  }, []);

  const startDownload = useCallback((formatId?: string) => {
    if (!state.jobId) return;
    setState((s) => ({
      ...s,
      phase: 'downloading',
      progress: 0,
      stage: 'starting download',
      error: null,
    }));
    send({ action: 'download', format_id: formatId });
  }, [state.jobId, send]);

  const extract = useCallback(async () => {
    if (!state.jobId) return;

    setState((s) => ({
      ...s,
      phase: 'extracting',
      progress: 0,
      stage: 'starting',
      stats: null,
      results: [],
      error: null,
    }));
    
    send({ action: 'extract', settings: state.settings });
  }, [state.jobId, state.settings, send]);

  const updateSettings = useCallback((update: Partial<ExtractionSettings>) => {
    setState((s) => ({
      ...s,
      settings: { ...s.settings, ...update },
    }));
  }, []);

  const reset = useCallback(() => {
    setState({
      phase: 'idle',
      jobId: null,
      videoMeta: null,
      settings: { ...DEFAULT_SETTINGS },
      progress: 0,
      stage: '',
      stats: null,
      results: [],
      error: null,
      wsUrl: null,
    });
  }, []);

  return {
    ...state,
    upload,
    importUrl,
    startDownload,
    extract,
    updateSettings,
    reset,
  };
}
