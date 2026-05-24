import { useState } from 'react';
import { Play, Download, RotateCcw, XCircle } from 'lucide-react';
import { motion } from 'motion/react';
import { Header } from './components/Header';
import { VideoUpload } from './components/VideoUpload';
import { SettingsPanel } from './components/SettingsPanel';
import { ProgressBar } from './components/ProgressBar';
import { Gallery } from './components/Gallery';
import { StatsCard } from './components/StatsCard';
import { useExtraction } from './hooks/useExtraction';
import { getDownloadUrl, getVideoDownloadUrl } from './services/api';
import './index.css';

function App() {
  const {
    phase,
    jobId,
    videoMeta,
    settings,
    progress,
    stage,
    stats,
    results,
    error,
    upload,
    importUrl,
    startDownload,
    abortDownload,
    extract,
    updateSettings,
    reset,
  } = useExtraction();

  const [selectedFormat, setSelectedFormat] = useState<{ jobId: string | null; formatId: string }>({
    jobId: null,
    formatId: '',
  });
  const selectedFormatId = selectedFormat.jobId === jobId
    ? selectedFormat.formatId
    : videoMeta?.format_id ?? '';
  const setSelectedFormatId = (formatId: string) => {
    setSelectedFormat({ jobId, formatId });
  };

  const isExtracting = phase === 'extracting';
  const isDownloading = phase === 'downloading';
  const isWorking = isExtracting || isDownloading;

  // Settings are visible for local uploads immediately, but for URLs only after they are downloaded.
  const isUrlReadyToDownload = videoMeta?.is_url && phase === 'ready';
  const showSettings = (phase === 'ready' && !videoMeta?.is_url) || phase === 'downloaded' || phase === 'extracting' || phase === 'complete';
  const showProgress = phase === 'extracting' || phase === 'downloading';
  const showResults = phase === 'complete' && results.length > 0;
  const showError = phase === 'error' && error;
  const canDownloadVideo = Boolean(jobId && videoMeta?.downloaded_from_url);

  return (
    <div className="app">
      <Header />

      <main className="main">
        <div className="layout">
          {/* Left panel */}
          <aside className="panel panel--left">
            <VideoUpload
              phase={phase}
              jobId={jobId}
              videoMeta={videoMeta}
              onUpload={upload}
              onUrlImport={importUrl}
              onReset={reset}
            />

            {showSettings && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="settings-wrapper"
              >
                <SettingsPanel
                  settings={settings}
                  onChange={updateSettings}
                  disabled={isWorking}
                  maxTargetFps={videoMeta?.fps}
                />

                <div className="action-bar">
                  {(phase === 'ready' || phase === 'downloaded') && (
                    <button
                      className="btn btn--primary btn--lg"
                      style={canDownloadVideo ? undefined : { width: '100%' }}
                      onClick={extract}
                      disabled={isWorking || !jobId}
                    >
                      {isWorking ? (
                        <span className="upload-spinner" style={{ width: 18, height: 18, border: '2px solid rgba(255,255,255,0.3)', borderTopColor: '#fff' }} />
                      ) : (
                        <Play size={20} />
                      )}
                      <span>{isWorking ? 'Processing...' : 'Extract Frames'}</span>
                    </button>
                  )}
                  {canDownloadVideo && phase === 'downloaded' && jobId && (
                    <a
                      className="btn btn--secondary btn--lg"
                      href={getVideoDownloadUrl(jobId)}
                      download
                    >
                      <Download size={20} />
                      Download Video
                    </a>
                  )}
                  {phase === 'complete' && (
                    <>
                      <button className="btn btn--primary" onClick={extract}>
                        <RotateCcw size={14} />
                        Re-extract
                      </button>
                      {canDownloadVideo && jobId && (
                        <a
                          className="btn btn--secondary"
                          href={getVideoDownloadUrl(jobId)}
                          download
                        >
                          <Download size={14} />
                          Video
                        </a>
                      )}
                      {jobId && (
                        <a
                          className="btn btn--secondary"
                          href={getDownloadUrl(jobId)}
                          download
                          title="Download results ZIP"
                          aria-label="Download results ZIP"
                        >
                          <Download size={14} />
                          ZIP
                        </a>
                      )}
                    </>
                  )}
                </div>
              </motion.div>
            )}

            {isUrlReadyToDownload && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="settings-wrapper"
              >
                {videoMeta?.available_formats && videoMeta.available_formats.length > 0 && (
                  <div className="settings-group">
                    <label className="settings-label">
                      Video Quality
                      <span className="settings-tooltip" title="Select download resolution">ⓘ</span>
                    </label>
                    <select 
                      className="settings-select"
                      value={selectedFormatId}
                      onChange={(e) => setSelectedFormatId(e.target.value)}
                      disabled={isWorking}
                    >
                      {videoMeta.available_formats.map(fmt => (
                        <option key={fmt.format_id} value={fmt.format_id}>
                          {fmt.resolution} ({fmt.width}x{fmt.height})
                        </option>
                      ))}
                    </select>
                  </div>
                )}
                
                <div className="action-bar" style={{ marginTop: '1.5rem' }}>
                  {isDownloading ? (
                    <button
                      className="btn btn--danger btn--lg"
                      style={{ width: '100%', background: 'rgba(239, 68, 68, 0.2)', color: '#ef4444', borderColor: 'rgba(239, 68, 68, 0.5)' }}
                      onClick={abortDownload}
                    >
                      <XCircle size={20} />
                      <span>Cancel Download</span>
                    </button>
                  ) : (
                    <button
                      className="btn btn--primary btn--lg"
                      style={{ width: '100%' }}
                      onClick={() => startDownload(selectedFormatId || videoMeta?.format_id)}
                      disabled={isWorking || !jobId}
                    >
                      <Download size={20} />
                      <span>Download Video</span>
                    </button>
                  )}
                </div>
              </motion.div>
            )}
          </aside>

          {/* Right panel */}
          <section className="panel panel--right">
            {showError && (
              <div className="error-banner">
                <p>{error}</p>
                <button className="btn btn--secondary btn--sm" onClick={reset}>
                  Try again
                </button>
              </div>
            )}

            {showProgress && (
              <ProgressBar progress={progress} stage={stage} />
            )}

            {showResults && stats && (
              <StatsCard stats={stats} />
            )}

            <Gallery results={results} />

            {!showResults && !showProgress && !showError && (
              <div className="empty-state">
                <div className="empty-state__icon">🎯</div>
                <h2 className="empty-state__title">Ready to extract</h2>
                <p className="empty-state__text">
                  Upload a video and configure your settings to begin extracting sharp, face-focused training data.
                </p>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
