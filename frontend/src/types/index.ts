export interface VideoMeta {
  fps: number;
  duration: number;
  width: number;
  height: number;
  frame_count: number;
  format_id?: string;
  thumbnail_url?: string;
  is_url?: boolean;
  downloaded_from_url?: boolean;
  available_formats?: Array<{
    format_id: string;
    resolution: string;
    width: number;
    height: number;
  }>;
}

export interface UploadResponse {
  job_id: string;
  meta: VideoMeta;
}

export interface ExtractionSettings {
  target_fps: number;
  blur_threshold: number;
  detection_confidence: number;
  crop_mode: 'face' | 'body';
  padding_pct: number;
  square_method: 'center_crop' | 'letterbox';
  output_size: number;
  output_format: 'png' | 'jpg';
  dedup_threshold: number;
  occlusion_threshold: number;
  download_resolution: 'max' | '1080p' | '720p' | '480p';
}

export interface ExtractionStats {
  total_sampled: number;
  blurry_discarded: number;
  low_resolution_discarded: number;
  no_face_discarded: number;
  duplicate_discarded: number;
  occluded_discarded: number;
  extracted: number;
  gpu_backend: string;
}

export interface ProgressMessage {
  type: 'progress';
  progress: number;
  stage: string;
  message: string;
}

export interface CompleteMessage {
  type: 'complete';
  stats: ExtractionStats;
  results: string[];
  total: number;
}

export interface ErrorMessage {
  type: 'error';
  error: string;
}

export interface DownloadCompleteMessage {
  type: 'download_complete';
  video_path: string;
  meta: VideoMeta;
}

export interface DownloadAbortedMessage {
  type: 'download_aborted';
}

export type WsMessage = ProgressMessage | CompleteMessage | ErrorMessage | DownloadCompleteMessage | DownloadAbortedMessage;

export interface GpuInfo {
  backend: 'cuda' | 'coreml' | 'cpu';
  device: string;
  provider: string;
}

export const DEFAULT_SETTINGS: ExtractionSettings = {
  target_fps: 2,
  blur_threshold: 100,
  detection_confidence: 0.5,
  crop_mode: 'face',
  padding_pct: 20,
  square_method: 'center_crop',
  output_size: 512,
  output_format: 'png',
  dedup_threshold: 8,
  occlusion_threshold: 50,
  download_resolution: 'max',
};
