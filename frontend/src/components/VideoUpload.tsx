import { useCallback, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Upload, Film, X, Link as LinkIcon, ArrowRight } from 'lucide-react';
import type { VideoMeta } from '../types';
import type { ExtractionPhase } from '../hooks/useExtraction';
import { getVideoUrl } from '../services/api';

interface Props {
  phase: ExtractionPhase;
  jobId: string | null;
  videoMeta: VideoMeta | null;
  onUpload: (file: File) => void;
  onUrlImport: (url: string) => void;
  onReset: () => void;
}

export function VideoUpload({ phase, jobId, videoMeta, onUpload, onUrlImport, onReset }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [url, setUrl] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('video/')) {
        onUpload(file);
      }
    },
    [onUpload]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onUpload(file);
    },
    [onUpload]
  );

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (url.trim()) {
      onUrlImport(url.trim());
    }
  };

  const formatDuration = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  const showThumbnail = videoMeta?.is_url && (phase === 'ready' || phase === 'downloading');
  const showVideo = jobId && (!videoMeta?.is_url || phase === 'downloaded' || phase === 'extracting' || phase === 'complete');

  return (
    <div className="upload-section">
      <AnimatePresence mode="wait">
        {phase === 'idle' || phase === 'error' ? (
          <motion.div
            key="input-methods"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: 0.2 }}
            style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}
          >
            <div
              className={`dropzone ${isDragging ? 'dropzone--active' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
            >
              <Upload size={36} className="dropzone__icon" />
              <p className="dropzone__text">Drop video here or click to browse</p>
              <div className="dropzone__formats">
                {['MP4', 'MOV', 'AVI', 'MKV', 'WEBM'].map((f) => (
                  <span key={f} className="format-pill">{f}</span>
                ))}
              </div>
              <input
                ref={inputRef}
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                hidden
              />
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <div style={{ flex: 1, height: '1px', background: 'var(--border-subtle)' }} />
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>OR</span>
              <div style={{ flex: 1, height: '1px', background: 'var(--border-subtle)' }} />
            </div>

            <form onSubmit={handleUrlSubmit} style={{ display: 'flex', gap: '0.5rem' }}>
              <div style={{ position: 'relative', flex: 1 }}>
                <LinkIcon size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                <input
                  type="url"
                  placeholder="Paste YouTube or video URL..."
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 12px 10px 36px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-subtle)',
                    background: 'var(--bg-card)',
                    color: 'var(--text-primary)',
                    fontSize: '0.85rem',
                    outline: 'none',
                  }}
                />
              </div>
              <button
                type="submit"
                className="btn btn--primary"
                disabled={!url.trim()}
                style={{ height: 'auto', padding: '0 16px' }}
              >
                <ArrowRight size={16} />
              </button>
            </form>
          </motion.div>
        ) : phase === 'uploading' ? (
          <motion.div
            key="uploading"
            className="upload-status"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="upload-spinner" />
            <p>Uploading video...</p>
          </motion.div>
        ) : (
          <motion.div
            key="video-info"
            className="video-info"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
          >
            <div className="preview-container">
              {showThumbnail && videoMeta?.thumbnail_url && (
                <img 
                  src={videoMeta.thumbnail_url} 
                  alt="Video thumbnail preview" 
                  className="preview-thumbnail"
                />
              )}
              
              {showVideo && jobId && (
                <video 
                  src={getVideoUrl(jobId)} 
                  controls 
                  className="preview-video"
                  preload="metadata"
                />
              )}
            </div>
            
            <div className="video-info__header">
              <Film size={18} />
              <span className="video-info__label">Video loaded</span>
              {phase !== 'downloading' && phase !== 'extracting' && (
                <button className="video-info__reset" onClick={onReset} title="Remove video">
                  <X size={14} />
                </button>
              )}
            </div>
            {videoMeta && (
              <div className="video-info__meta">
                <span>{formatDuration(videoMeta.duration)}</span>
                <span className="meta-dot">•</span>
                <span>{videoMeta.fps.toFixed(2)} FPS</span>
                <span className="meta-dot">•</span>
                <span>
                  {videoMeta.is_url 
                    ? `Max Res: ${videoMeta.width}x${videoMeta.height}` 
                    : `${videoMeta.width}x${videoMeta.height}`}
                </span>
                <span className="meta-dot">•</span>
                <span>{videoMeta.frame_count.toLocaleString()} frames</span>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
