import scipy
import torch
from diffusers import AudioLDM2Pipeline
from diffusers import AudioLDMPipeline
import shutil
import os



# Constants
SR = 16000
DEVICE = "mps"
AUDIO_LENGTH_IN_S = 5
NUM_INFERENCE_STEPS = 10
GUIDANCE_SCALE = 2.5
SEED = 27463458782
NEGATIVE_PROMPT = "low quality, average quality, noise, high pitch, artefacts"
RESULTS_DIR = "results"

LDM_REPO_IDS = [
    "cvssp/audioldm-m-full",
    "cvssp/audioldm-l-full"
]

LDM2_REPO_IDS = [
    "cvssp/audioldm2-large",
    "cvssp/audioldm2-music"
]
generator = torch.Generator(DEVICE).manual_seed(SEED)

def initialize_pipeline(repo_ids, pipeline_type):
    pipelines = []
    for repo_id in repo_ids:
        pipe = pipeline_type.from_pretrained(repo_id, torch_dtype=torch.float32)
        pipe = pipe.to(DEVICE)
        pipe.enable_attention_slicing()
        pipe.name = repo_id.split('/')[-1]
        pipelines.append(pipe)
    return pipelines

def generate_audio_samples(prompts, pipes):
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)

    for prompt in prompts:
        prompt_underscores = prompt.replace(" ", "_")
        for pipe in pipes:
            audio = pipe(prompt=prompt,
                         audio_length_in_s=AUDIO_LENGTH_IN_S,
                         num_inference_steps=NUM_INFERENCE_STEPS,
                         guidance_scale=GUIDANCE_SCALE,
                         negative_prompt=NEGATIVE_PROMPT,
                         generator=generator).audios[0]

            # save sample to results folder
            os.makedirs(f"{RESULTS_DIR}/{prompt_underscores}", exist_ok=True)
            scipy.io.wavfile.write(f"{RESULTS_DIR}/{prompt_underscores}/{pipe.name}.mp3", SR, audio)


ldm_pipes = initialize_pipeline(LDM_REPO_IDS, AudioLDMPipeline)
ldm2_pipes = initialize_pipeline(LDM2_REPO_IDS, AudioLDM2Pipeline)
pipes = ldm_pipes + ldm2_pipes

with open("prompts.txt", "r") as f:
    prompts = f.read().splitlines()

generate_audio_samples(prompts, pipes)