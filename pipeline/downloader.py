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
    available_formats = {}
    
    for s in video_streams:
        # resolution is typically something like "1080p", "2160p"
        if hasattr(s, 'resolution') and s.resolution:
            h = int(s.resolution.replace('p', ''))
            
            # Deduplicate by height, keeping the first (often best codec) we see
            if h not in available_formats:
                available_formats[h] = {
                    "format_id": str(s.itag),
                    "resolution": s.resolution,
                    "height": h,
                    "width": int(h * 16 / 9)
                }
                
            if h > max_height:
                max_height = h
                best_stream = s
    
    if not best_stream:
        # Fallback to progressive if no adaptive streams are found
        best_stream = yt.streams.filter(progressive=True).order_by('resolution').desc().first()
        if best_stream and hasattr(best_stream, 'resolution') and best_stream.resolution:
            h = int(best_stream.resolution.replace('p', ''))
            available_formats[h] = {
                "format_id": str(best_stream.itag),
                "resolution": best_stream.resolution,
                "height": h,
                "width": int(h * 16 / 9)
            }
        
    if not best_stream:
        raise ValueError("Could not find any video streams for this URL.")
        
    fps = best_stream.fps if hasattr(best_stream, 'fps') else 30.0
    duration = yt.length if hasattr(yt, 'length') else 0
    
    # Approximate width based on 16:9 ratio
    height = max_height if max_height > 0 else int(best_stream.resolution.replace('p', '')) if best_stream.resolution else 0
    width = int(height * 16 / 9) if height else 0
    
    # Sort formats from highest to lowest
    sorted_formats = [fmt for _, fmt in sorted(available_formats.items(), key=lambda x: x[0], reverse=True)]
    
    return {
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
        "frame_count": int(fps * duration) if fps and duration else 0,
        "format_id": str(best_stream.itag),
        "available_formats": sorted_formats
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
    os.makedirs(output_dir, exist_ok=True)
    
    def on_progress(stream, chunk, bytes_remaining):
        if progress_callback:
            total_size = stream.filesize
            if total_size > 0:
                pct = (total_size - bytes_remaining) / total_size
                progress_callback(pct, "Downloading video...")
                
    yt = YouTube(url, client='WEB', on_progress_callback=on_progress)
    
    # If a specific format_id was passed, use it, otherwise get the best video stream
    if format_id and format_id != "bestvideo":
        stream = yt.streams.get_by_itag(int(format_id))
    else:
        stream = yt.streams.filter(adaptive=True, type='video').order_by('resolution').desc().first()
        if not stream:
            stream = yt.streams.get_highest_resolution()
            
    if not stream:
        raise ValueError("Could not find a valid stream to download.")
        
    if progress_callback:
        progress_callback(0.0, "Starting download...")
        
    # Download the video
    out_file = stream.download(
        output_path=output_dir,
        filename=f"{job_id}.mp4"
    )
    
    if progress_callback:
        progress_callback(1.0, "Download finished, finalizing file...")
        
    return out_file
