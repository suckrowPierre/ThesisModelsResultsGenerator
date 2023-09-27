import torch
from torch import mps
from pathlib import Path
from diffusers import AudioLDM2Pipeline
from diffusers import AudioLDMPipeline
from scipy.io import wavfile
import shutil
import gc
import logging
from typing import List, Dict, Tuple, Any, Optional

logging.basicConfig(level=logging.INFO)

SR = 16000
RESULTS_DIR = "results"
torch_dtype = torch.float32

LDM_REPO_IDS = [
    "cvssp/audioldm-s-full-v2",
    "cvssp/audioldm-m-full",
    "cvssp/audioldm-l-full"
]

LDM2_REPO_IDS = [
    "cvssp/audioldm2",
    "cvssp/audioldm2-large",
    "cvssp/audioldm2-music"
]


def initialize_pipeline(repo_id: str, device: str, torch_dtype: torch.dtype):
    pipeline_type = AudioLDMPipeline if repo_id in LDM_REPO_IDS else AudioLDM2Pipeline
    pipe = pipeline_type.from_pretrained(repo_id, torch_dtype=torch_dtype).to(device)
    pipe.name = repo_id.split('/')[-1]
    generator = torch.Generator(device)
    return pipe, generator


def write_audio(path: Path, audio, sr: int = SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_suffix(".wav")
    wavfile.write(path, sr, audio)


def validate_params(params: dict):
    required_params = [
        "prompt",
        "audio_length_in_s",
        "guidance_scale",
        "num_inference_steps",
        "negative_prompt",
        "num_waveforms_per_prompt",
        "generator",
    ]
    for param in required_params:
        if params.get(param) is None:
            raise ValueError(f"{param} must be specified")

    for key in params:
        if key not in required_params:
            raise ValueError(f"Parameter {key} is not a valid parameter")


def text2audio(pipe: any, generator: any, parameters, seed: int):
    parameters['generator'] = generator.manual_seed(seed)
    validate_params(parameters)
    logging.info(f"Generating audio for prompt: {parameters.get('prompt')} using {pipe.name}")
    waveforms = pipe(**parameters)["audios"]
    return waveforms[0]


def clean_pipeline(pipe: any, device: str):
    del pipe
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    if device == "cuda":
        torch.cuda.empty_cache()


def generate_audio(path: Path, device: str, prompt: str, pipe: Any, generator: Any, params: Dict[str, Any], seed: int,
                   parameter_variation: Tuple[str, any] = None):
    prompt_underscores = prompt.replace(" ", "_")
    path_audio = path / device / prompt_underscores / pipe.name
    if parameter_variation:
        variation_name, variation_values = parameter_variation
        for variation in variation_values:
            params[variation_name] = variation
            audio = text2audio(pipe, generator, params, seed)
            write_audio(path_audio / str(variation), audio)
    else:
        audio = text2audio(pipe, generator, params, seed)
        write_audio(path_audio, audio)


def generate_evaluation(
        path: Path,
        devices: List[str],
        models: List[Any],
        prompts: List[str],
        params: Dict[str, Any],
        seed: int,
        parameter_variation: Optional[Tuple[str, Any]] = None,
        delete_old_files_flag: bool = True
):
    if delete_old_files_flag and path.exists():
        logging.info(f"Deleting old files in {path}")
        shutil.rmtree(path)
    for device in devices:
        logging.info(f"Generating audios for device: {device}")
        for model in models:
            pipe, generator = initialize_pipeline(model, device, torch_dtype)
            for prompt in prompts:
                params['prompt'] = prompt
                generate_audio(path, device, prompt, pipe, generator, params, seed, parameter_variation)
            clean_pipeline(pipe, device)


def device_evaluation():
    path = Path(RESULTS_DIR + "/device_evaluation")
    devices = ["mps", "cpu"]
    models = ["cvssp/audioldm-m-full", "cvssp/audioldm2-music"]
    prompts = ["A blues harmonica", "Ambient ocean waves"]
    seed = 2304559601318669088
    params = {
        "audio_length_in_s": 5,
        "guidance_scale": 3,
        "num_inference_steps": 50,
        "negative_prompt": "low quality, average quality, noise, high pitch, artefacts",
        "num_waveforms_per_prompt": 1,
    }
    generate_evaluation(path, devices, models, prompts, params, seed)


def duration_evaluation():
    path = Path(RESULTS_DIR + "/duration_evaluation")
    devices = ["mps"]
    models = ["cvssp/audioldm-m-full", "cvssp/audioldm2-music"]
    prompts = ["A flute ensemble", "A choir pad", "An ambient electronic pad"]
    seed = 8094681003267709068
    params = {
        "guidance_scale": 3,
        "num_inference_steps": 50,
        "negative_prompt": "low quality, average quality, noise, high pitch, artefacts",
        "num_waveforms_per_prompt": 1,
    }
    parameter_variation = (
        "audio_length_in_s", [5, 10, 20, 30]
    )

    generate_evaluation(path, devices, models, prompts, params, seed, parameter_variation)


def main():
    device_evaluation()
    # duration_evaluation()


if __name__ == '__main__':
    main()
