import type { ExtractionSettings, VideoMeta } from '../types';

interface Props {
  settings: ExtractionSettings;
  onChange: (update: Partial<ExtractionSettings>) => void;
  disabled?: boolean;
  maxTargetFps?: number;
  videoMeta?: VideoMeta | null;
}

const MAX_SHARPNESS = 100;
const MIN_DETECTION_CONFIDENCE = 0.3;
const MAX_DETECTION_CONFIDENCE = 0.8;

export function SettingsPanel({ settings, onChange, disabled, maxTargetFps = 15, videoMeta }: Props) {
  const targetFpsMax = Math.max(0.5, maxTargetFps);
  const targetFpsValue = Math.min(settings.target_fps, targetFpsMax);
  const detectionConfidenceValue = Math.min(
    Math.max(settings.detection_confidence, MIN_DETECTION_CONFIDENCE),
    MAX_DETECTION_CONFIDENCE,
  );

  return (
    <div className="settings">
      <h2 className="settings__title">Settings</h2>

      <div className="setting-group">
        <label className="setting-label">
          Sample Rate (Frames/Sec)
          <span className="setting-value">{settings.target_fps}</span>
        </label>
        <input
          type="range"
          className="setting-slider"
          min={0.5} max={targetFpsMax} step={0.5}
          value={targetFpsValue}
          onChange={(e) => onChange({ target_fps: Number(e.target.value) })}
          disabled={disabled}
        />
        <div className="setting-range-labels">
          <span>0.5</span><span>{targetFpsMax.toFixed(targetFpsMax % 1 === 0 ? 0 : 2)}</span>
        </div>
      </div>

      <div className="setting-group">
        <label className="setting-label">
          Minimum Sharpness
          <span className="setting-value">{settings.blur_threshold}%</span>
        </label>
        <input
          type="range"
          className="setting-slider"
          min={0} max={MAX_SHARPNESS} step={5}
          value={Math.min(settings.blur_threshold, MAX_SHARPNESS)}
          onChange={(e) => onChange({ blur_threshold: Number(e.target.value) })}
          disabled={disabled}
        />
        <div className="setting-range-labels">
          <span>Allow Blurry</span><span>Require Crisp</span>
        </div>
      </div>

      <div className="setting-group">
        <label className="setting-label">
          Detection Confidence
          <span className="setting-value">{(settings.detection_confidence * 100).toFixed(0)}%</span>
        </label>
        <input
          type="range"
          className="setting-slider"
          min={MIN_DETECTION_CONFIDENCE} max={MAX_DETECTION_CONFIDENCE} step={0.05}
          value={detectionConfidenceValue}
          onChange={(e) => onChange({ detection_confidence: Number(e.target.value) })}
          disabled={disabled}
        />
        <div className="setting-range-labels">
          <span>30%</span><span>80%</span>
        </div>
      </div>

      <div className="setting-group">
        <label className="setting-label">Crop Mode</label>
        <div className="segmented-control">
          {(['face', 'body'] as const).map((mode) => (
            <button
              key={mode}
              className={`segment ${settings.crop_mode === mode ? 'segment--active' : ''}`}
              onClick={() => onChange({ crop_mode: mode })}
              disabled={disabled}
            >
              {mode === 'face' ? '👤 Face' : '🧍 Body'}
            </button>
          ))}
        </div>
      </div>

      <div className="setting-group">
        <label className="setting-label">
          Occlusion Strictness
          <span className="setting-value">
            {settings.occlusion_threshold === 0 ? 'Off' : `${settings.occlusion_threshold}%`}
          </span>
        </label>
        <input
          type="range"
          className="setting-slider"
          min={0} max={100} step={5}
          value={settings.occlusion_threshold}
          onChange={(e) => onChange({ occlusion_threshold: Number(e.target.value) })}
          disabled={disabled}
        />
        <div className="setting-range-labels">
          <span>Off</span><span>Strict</span>
        </div>
      </div>

      <div className="setting-group">
        <label className="setting-label">
          Padding
          <span className="setting-value">{settings.padding_pct}%</span>
        </label>
        <input
          type="range"
          className="setting-slider"
          min={0} max={200} step={5}
          value={settings.padding_pct}
          onChange={(e) => onChange({ padding_pct: Number(e.target.value) })}
          disabled={disabled}
        />
        <div className="setting-range-labels">
          <span>0</span><span>200</span>
        </div>
      </div>

      <div className="setting-group">
        <label className="setting-label">Square Method</label>
        <div className="segmented-control">
          {([
            { value: 'center_crop', label: 'Center Crop' },
            { value: 'letterbox', label: 'Letterbox' },
          ] as const).map(({ value, label }) => (
            <button
              key={value}
              className={`segment ${settings.square_method === value ? 'segment--active' : ''}`}
              onClick={() => onChange({ square_method: value })}
              disabled={disabled}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="setting-group">
        <label className="setting-label">Output Size</label>
        <div className="segmented-control segmented-control--quad">
          {[256, 512, 768, 1024, 1280, 1536, 1920].map((size) => {
            const minVidDim = videoMeta ? Math.min(videoMeta.width, videoMeta.height) : Infinity;
            // If the video min dimension is 0 (some error), we just allow it. Otherwise restrict.
            const isTooLarge = (minVidDim > 0) && (size > minVidDim);
            return (
              <button
                key={size}
                className={`segment ${settings.output_size === size ? 'segment--active' : ''}`}
                onClick={() => onChange({ output_size: size })}
                disabled={disabled || isTooLarge}
                title={isTooLarge ? `Video resolution is too small (${videoMeta?.width}x${videoMeta?.height})` : undefined}
              >
                {size}
              </button>
            );
          })}
        </div>
      </div>

      <div className="setting-group">
        <label className="setting-label">Format</label>
        <div className="segmented-control">
          {(['png', 'jpg'] as const).map((fmt) => (
            <button
              key={fmt}
              className={`segment ${settings.output_format === fmt ? 'segment--active' : ''}`}
              onClick={() => onChange({ output_format: fmt })}
              disabled={disabled}
            >
              {fmt.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <div className="setting-group">
        <label className="setting-label">
          De-duplication
          <span className="setting-value">
            {settings.dedup_threshold === 0 ? 'Off' : `${Math.round((settings.dedup_threshold / 16) * 100)}%`}
          </span>
        </label>
        <input
          type="range"
          className="setting-slider"
          min={0} max={100} step={1}
          value={Math.round((settings.dedup_threshold / 16) * 100)}
          onChange={(e) => onChange({ dedup_threshold: Math.round((Number(e.target.value) / 100) * 16) })}
          disabled={disabled}
        />
        <div className="setting-range-labels">
          <span>Off</span><span>Aggressive</span>
        </div>
      </div>
    </div>
  );
}
