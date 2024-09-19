import torch
from text_to_video import SD3Transformer2DModel
import random
import numpy as np
import os
import argparse
from datetime import datetime
import moviepy.editor as mp

# Function to set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Function to save the video as an .mp4 file
def save_video(tensor):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path)
    return video_path

# Function to convert .mp4 video to .gif
def convert_to_gif(video_path):
    clip = mp.VideoFileClip(video_path)
    clip = clip.set_fps(8)
    clip = clip.resize(height=240)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path

# Argument parsing to allow for command-line inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from text prompt")

    # Defining the command-line arguments
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate video")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt to avoid in generation")
    parser.add_argument("--step", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--scale", type=float, default=3.5, help="Guidance scale for the model")
    parser.add_argument("--width", type=int, default=1024, help="Width of the output video")
    parser.add_argument("--height", type=int, default=1024, help="Height of the output video")
    parser.add_argument("--frames", type=int, default=25, help="Number of frames in the generated video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Setting the seed for reproducibility
    set_seed(args.seed)

    # Define device and load the model
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = SD3Transformer2DModel("stabilityai/stable-diffusion-3-medium-diffusers", device)

    # Generate video based on the prompt
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        video = pipe(
            args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.step,
            guidance_scale=args.scale,
            width=args.width,
            height=args.height,
            frames=args.frames
        )

    # Save the generated video
    video_path = save_video(video)
    print(f"Video saved at: {video_path}")

    # Convert the video to GIF
    gif_path = convert_to_gif(video_path)
    print(f"GIF saved at: {gif_path}")
