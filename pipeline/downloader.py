from __future__ import annotations

import os
import time
from typing import Callable
from pytubefix import YouTube

def probe_url(url: str) -> dict:
    """Fetch video metadata from URL without downloading and find the best format using pytubefix."""
    yt = YouTube(url, client='WEB')
    
    # Filter for adaptive video streams (video-only) and order by resolution
    video_streams = yt.streams.filter(adaptive=True, type='video')
    
    # Sort by height to find the maximum resolution
    best_stream = None
    max_height = 0
    
    for s in video_streams:
        # resolution is typically something like "1080p", "2160p"
        if hasattr(s, 'resolution') and s.resolution:
            h = int(s.resolution.replace('p', ''))
            if h > max_height:
                max_height = h
                best_stream = s
    
    if not best_stream:
        # Fallback to progressive if no adaptive streams are found
        best_stream = yt.streams.filter(progressive=True).order_by('resolution').desc().first()
        
    if not best_stream:
        raise ValueError("Could not find any video streams for this URL.")
        
    fps = best_stream.fps if hasattr(best_stream, 'fps') else 30.0
    duration = yt.length if hasattr(yt, 'length') else 0
    
    # Approximate width based on 16:9 ratio
    height = max_height if max_height > 0 else int(best_stream.resolution.replace('p', '')) if best_stream.resolution else 0
    width = int(height * 16 / 9) if height else 0
    
    return {
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
        "frame_count": int(fps * duration) if fps and duration else 0,
        "format_id": str(best_stream.itag)
    }

def download_video(
    url: str,
    output_dir: str,
    job_id: str,
    progress_callback: Callable[[float, str], None] | None = None,
    format_id: str = "bestvideo"
) -> str:
    """
    Download a video from a URL using pytubefix.

    Args:
        url: The video URL.
        output_dir: Directory to save the video.
        job_id: Job identifier for naming.
        progress_callback: Callback for download progress.
        format_id: Target itag extracted during analysis.

    Returns:
        The absolute path to the downloaded video file.
    """
            ydl_opts['format'] = 'best'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return ydl.prepare_filename(info)
        raise e
