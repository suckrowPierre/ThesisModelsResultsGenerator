import scipy
import torch
from diffusers import AudioLDM2Pipeline
from diffusers import AudioLDMPipeline
import shutil
import os



LDM_repo_ids = ["cvssp/audioldm-m-full", "cvssp/audioldm-l-full"]
LDM2_repo_ids = ["cvssp/audioldm2-large", "cvssp/audioldm2-music"]


sr = 16000
device = "mps"
audio_length_in_s = 5
num_inference_steps = 10
guidance_scale = 2.5
seed = 27463458782
negative_prompt = "low quality, average quality, noise, high pitch, artefacts"


pipes = []


for repo_id in LDM_repo_ids:
    pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.name = repo_id.split('/')[-1]
    pipes.append(pipe)

for repo_id in LDM2_repo_ids:
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.name = repo_id.split('/')[-1]
    pipes.append(pipe)


generator = torch.Generator(device).manual_seed(seed)

with open("prompts.txt", "r") as f:
    prompts = f.read().splitlines()

# delete folders in /results
shutil.rmtree("results")

# for each prompt, generate a folder in /results with the prompt as the name. Spaces are replaced with underscores
for prompt in prompts:
    prompt_underscores = prompt.replace(" ", "_")
    # generate sample for each pipe
    for pipe in pipes:
        audio = pipe(prompt=prompt,
                             audio_length_in_s=audio_length_in_s,
                             num_inference_steps=num_inference_steps,
                             guidance_scale=guidance_scale,
                             negative_prompt=negative_prompt,
                             generator=generator
                             ).audios[0]
        # save sample to results folder
        os.makedirs(f"results/{prompt_underscores}", exist_ok=True)
        scipy.io.wavfile.write(f"results/{prompt_underscores}/{pipe.name}.mp3", sr, audio)

