import torch
import random
import numpy as np
import os
from text_to_video import SD3Transformer2DModel
from datetime import datetime
import moviepy.editor as mp
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient."""
        # Define the device and load the model
        self.dtype = torch.float16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the model (Ensure SD3Transformer2DModel is available)
        self.pipe = SD3Transformer2DModel("v2ray/stable-diffusion-3-medium-diffusers", self.device)

    def set_seed(self, seed: int):
        """Set the seed for reproducibility."""
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def save_video(self, tensor) -> str:
        """Save the video as an .mp4 file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"./output/{timestamp}.mp4"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        export_to_video(tensor, video_path)  # Assuming export_to_video is defined
        return video_path

    def convert_to_gif(self, video_path: str) -> str:
        """Convert .mp4 video to .gif."""
        clip = mp.VideoFileClip(video_path)
        clip = clip.set_fps(8)
        clip = clip.resize(height=240)
        gif_path = video_path.replace(".mp4", ".gif")
        clip.write_gif(gif_path, fps=8)
        return gif_path

    def predict(
        self,
        prompt: str = Input(description="Text prompt to generate video"),
        negative_prompt: str = Input(description="Negative prompt to avoid in generation", default=""),
        step: int = Input(description="Number of inference steps", default=32),
        scale: float = Input(description="Guidance scale for the model", default=3.5),
        width: int = Input(description="Width of the output video", default=1024),
        height: int = Input(description="Height of the output video", default=1024),
        frames: int = Input(description="Number of frames in the generated video", default=25),
        seed: int = Input(description="Random seed for reproducibility", default=42),
    ) -> Path:
        """Run a single prediction."""
        self.set_seed(seed)

        # Generate video based on the prompt
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            video = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=step,
                guidance_scale=scale,
                width=width,
                height=height,
                frames=frames
            )

        # Save and convert video
        video_path = self.save_video(video)
        gif_path = self.convert_to_gif(video_path)

        return Path(gif_path)
