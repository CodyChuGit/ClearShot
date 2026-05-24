import type { ExtractionSettings } from '../types';

interface Props {
  settings: ExtractionSettings;
  onChange: (update: Partial<ExtractionSettings>) => void;
  disabled?: boolean;
}

export function SettingsPanel({ settings, onChange, disabled }: Props) {
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
          min={0.5} max={15} step={0.5}
          value={settings.target_fps}
          onChange={(e) => onChange({ target_fps: Number(e.target.value) })}
          disabled={disabled}
        />
        <div className="setting-range-labels">
          <span>0.5</span><span>15</span>
        </div>
      </div>

      <div className="setting-group">
        <label className="setting-label">
          Minimum Sharpness
          <span className="setting-value">{settings.blur_threshold}</span>
        </label>
        <input
          type="range"
          className="setting-slider"
          min={20} max={500} step={10}
          value={settings.blur_threshold}
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
          min={0.3} max={1} step={0.05}
          value={settings.detection_confidence}
          onChange={(e) => onChange({ detection_confidence: Number(e.target.value) })}
          disabled={disabled}
        />
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
          Padding
          <span className="setting-value">{settings.padding_pct}%</span>
        </label>
        <input
          type="range"
          className="setting-slider"
          min={0} max={80} step={5}
          value={settings.padding_pct}
          onChange={(e) => onChange({ padding_pct: Number(e.target.value) })}
          disabled={disabled}
        />
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
          {[256, 512, 768, 1024, 1280, 1536, 1920].map((size) => (
            <button
              key={size}
              className={`segment ${settings.output_size === size ? 'segment--active' : ''}`}
              onClick={() => onChange({ output_size: size })}
              disabled={disabled}
            >
              {size}
            </button>
          ))}
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
            {settings.dedup_threshold === 0 ? 'Off' : settings.dedup_threshold}
          </span>
        </label>
        <input
          type="range"
          className="setting-slider"
          min={0} max={20} step={1}
          value={settings.dedup_threshold}
          onChange={(e) => onChange({ dedup_threshold: Number(e.target.value) })}
          disabled={disabled}
        />
        <div className="setting-range-labels">
          <span>Off</span><span>Aggressive</span>
        </div>
      </div>
    </div>
  );
}
