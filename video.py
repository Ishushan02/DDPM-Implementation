import os
import subprocess

frame_folder = '/Users/ishananand/Desktop/DDPM-Implementation/default/samples'
output_video = '/Users/ishananand/Desktop/DDPM-Implementation/output_video.mp4'
frame_rate = 70

# List and sort frames in reverse order
frames = [f for f in os.listdir(frame_folder) if f.endswith('.png')]
frames.sort(reverse=True)  # Reverse the order of the frames

# Create frame pattern based on reversed frames
frame_pattern = os.path.join(frame_folder, "x0_%d.png")  # Assuming this pattern

output_video = '/Users/ishananand/Desktop/DDPM-Implementation/output_video.mp4'

# Now run ffmpeg with the reversed frame order
ffmpeg_command = [
    "ffmpeg",
    "-framerate", str(frame_rate),  # Set the frame rate
    "-i", frame_pattern,  # Input pattern (frame filenames)
    "-vf", "setpts=0.25*PTS",
    "-af", "areverse",         # Reverse the audio as well (if applicable)
    "-c:v", "libx264",         # Video codec
    "-preset", "fast",         # Encoding preset (optional)
    "-c:a", "aac",             # Audio codec (if applicable)
    "-strict", "experimental",
    "-vf", "reverse",
    "-c:v", "libx264",    # Video codec (libx264 for MP4)
    "-r", str(frame_rate), # Output frame rate
    "-pix_fmt", "yuv420p", # Pixel format for compatibility
    output_video           # Output video file
]

# Run the command
subprocess.run(ffmpeg_command)

print(f"Video saved as {output_video}")
