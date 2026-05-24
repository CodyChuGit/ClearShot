import { Play, Download, RotateCcw } from 'lucide-react';
import { motion } from 'motion/react';
import { Header } from './components/Header';
import { VideoUpload } from './components/VideoUpload';
import { SettingsPanel } from './components/SettingsPanel';
import { ProgressBar } from './components/ProgressBar';
import { Gallery } from './components/Gallery';
import { StatsCard } from './components/StatsCard';
import { useExtraction } from './hooks/useExtraction';
import { getDownloadUrl } from './services/api';
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
    extract,
    updateSettings,
    reset,
  } = useExtraction();

  const isExtracting = phase === 'extracting';
  const isDownloading = phase === 'downloading';
  const isWorking = isExtracting || isDownloading;

  // Settings are visible for local uploads immediately, but for URLs only after they are downloaded.
  const isUrlReadyToDownload = videoMeta?.is_url && phase === 'ready';
  const showSettings = (phase === 'ready' && !videoMeta?.is_url) || phase === 'downloaded' || phase === 'extracting' || phase === 'complete';
  const showProgress = phase === 'extracting' || phase === 'downloading';
  const showResults = phase === 'complete' && results.length > 0;
  const showError = phase === 'error' && error;

  return (
    <div className="app">
      <Header />

      <main className="main">
        <div className="layout">
          {/* Left panel */}
          <aside className="panel panel--left">
            <VideoUpload
              phase={phase}
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
                />

                <div className="action-bar">
                  {(phase === 'ready' || phase === 'downloaded') && (
                    <button
                      className="btn btn--primary btn--lg"
                      style={{ width: '100%' }}
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
                  {phase === 'complete' && (
                    <>
                      <button className="btn btn--primary" onClick={extract}>
                        <RotateCcw size={14} />
                        Re-extract
                      </button>
                      {jobId && (
                        <a
                          className="btn btn--secondary"
                          href={getDownloadUrl(jobId)}
                          download
                        >
                          <Download size={14} />
                          Download ZIP
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
                <div className="action-bar">
                  <button
                    className="btn btn--primary btn--lg"
                    style={{ width: '100%' }}
                    onClick={startDownload}
                    disabled={isWorking || !jobId}
                  >
                    {isWorking ? (
                      <span className="upload-spinner" style={{ width: 18, height: 18, border: '2px solid rgba(255,255,255,0.3)', borderTopColor: '#fff' }} />
                    ) : (
                      <Download size={20} />
                    )}
                    <span>{isWorking ? 'Downloading...' : 'Download Video'}</span>
                  </button>
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
