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
    write_params_to_file(path, devices, models, prompts, params, seed, parameter_variation)
    for device in devices:
        logging.info(f"Generating audios for device: {device}")
        for model in models:
            pipe, generator = initialize_pipeline(model, device, torch_dtype)
            for prompt in prompts:
                params['prompt'] = prompt
                generate_audio(path, device, prompt, pipe, generator, params, seed, parameter_variation)
            clean_pipeline(pipe, device)


def write_params_to_file(
        path: Path,
        devices: List[str],
        models: List[Any],
        prompts: List[str],
        params: Dict[str, Any],
        seed: int,
        parameter_variation: Optional[Tuple[str, Any]] = None
):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "parameters.txt", "w") as f:
        f.write(f"seed: {seed}\n")
        f.write(f"devices: {devices}\n")
        models = [model.split('/')[1] for model in models]
        f.write(f"models: {models}\n")
        if prompts:
            f.write(f"prompts: {prompts}\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        if parameter_variation:
            key, value = parameter_variation
            f.write(f"{key}: {value}\n")
def device_evaluation():
    path = Path(RESULTS_DIR + "/device_evaluation")
    devices = ["mps", "cpu"]
    models = LDM_REPO_IDS + LDM2_REPO_IDS

    prompts = ["A string orchestra", "Ambient ocean waves", "A hip hop beat", "Smooth synth lead", "Warm organ chord", "Bright bell sound"]
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
    models = ["cvssp/audioldm-m-full", "cvssp/audioldm2"]
    prompts = ["Chirping birds at dawn", "A choir pad", "An ambient electronic pad"]
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


def guidance_scale_evaluation():
    path = Path(RESULTS_DIR + "/guidance_scale_evaluation")
    devices = ["mps"]
    models = LDM_REPO_IDS + LDM2_REPO_IDS
    prompts = ["Gentle guitar strum", "Tribal African drum circle", "vocal harmonies"]
    seed = 7484926261153206834
    params = {
        "audio_length_in_s": 5,
        "num_inference_steps": 50,
        "negative_prompt": "low quality, average quality, noise, high pitch, artefacts",
        "num_waveforms_per_prompt": 1,
    }
    parameter_variation = (
        "guidance_scale", [1, 2, 3, 4, 5]
    )

    generate_evaluation(path, devices, models, prompts, params, seed, parameter_variation)


def num_inference_steps_evaluation():
    path = Path(RESULTS_DIR + "/num_inference_steps_evaluation")
    devices = ["mps"]
    models = LDM_REPO_IDS + LDM2_REPO_IDS
    prompts = ["ambient texture", "techno kickdrum", "Long evolving drone"]
    seed = 12705433011279635488
    params = {
        "audio_length_in_s": 5,
        "guidance_scale": 3,
        "negative_prompt": "low quality, average quality, noise, high pitch, artefacts",
        "num_waveforms_per_prompt": 1,
    }
    parameter_variation = (
        "num_inference_steps", [5, 10, 20, 50, 100, 200, 400]
    )
    generate_evaluation(path, devices, models, prompts, params, seed, parameter_variation)

def negative_prompt_evaluation():
    path = Path(RESULTS_DIR + "/negative_prompt_evaluation")
    devices = ["mps"]
    models = ["cvssp/audioldm-m-full", "cvssp/audioldm2-music"]
    prompts = ["Soft flute note", "Choir", "A muted trumpet"]
    seed = 15159449351027230159
    params = {
        "audio_length_in_s": 5,
        "guidance_scale": 3,
        "num_inference_steps": 50,
        "num_waveforms_per_prompt": 1,
    }
    parameter_variation = ( "negative_prompt",
    ["low quality", "average quality", "harsh noise", "dissonant chords", "distorted sounds", "clashing frequencies", "feedback loop", "clattering", "inharmonious", "average quality", "noise", "high pitch", "artefacts"])

    generate_evaluation(path, devices, models, prompts, params, seed, parameter_variation)

def prompts_evaluation():

    devices = ["mps"]
    models = ["cvssp/audioldm-m-full", "cvssp/audioldm-l-full", "cvssp/audioldm2", "cvssp/audioldm2-music"]
    seed = 12057337057645377627
    params = {
        "audio_length_in_s": 5,
        "guidance_scale": 3,
        "num_inference_steps": 100,
        "negative_prompt": "low quality, average quality, noise, harsh noise",
        "num_waveforms_per_prompt": 1,
    }
    write_params_to_file(Path(RESULTS_DIR + "/prompts_evaluation"), devices, models, [], params, seed)

    # Single Events Prompts
    path = Path(RESULTS_DIR + "/prompts_evaluation/single_events_prompts")
    prompts = ["A kickdrum", "A snare", "A single Light triangle ting", "Loud clap sound", "A gong hit"]
    generate_evaluation(path, devices, models, prompts, params, seed)

    # Instrument Specific Prompts
    path = Path(RESULTS_DIR + "/prompts_evaluation/instrument_specific_prompts")
    prompts = ["FM synthesis bells", "mellotron chords", "A bagpipe melody", "A guitar string", "A piano chord"]
    generate_evaluation(path, devices, models, prompts, params, seed)

    # Emotion Specific Prompts
    path = Path(RESULTS_DIR + "/prompts_evaluation/emotion_specific_prompts")
    prompts = ["Dark pad sound", "An ethereal, shimmering synth pad", "An angelic choir", "dreamy nostalgic strings", "a sad violin solo"]
    generate_evaluation(path, devices, models, prompts, params, seed)

    # Effect Specific Prompts
    path = Path(RESULTS_DIR + "/prompts_evaluation/effect_specific_prompts")
    prompts = ["Long sustain snare hit", "A fluttering harp with crystal echoes", "A Synth with a delay effect", "echoing synth stabs", "A distorted synth",
               "A detuned synth", "Reverse cymbal", "A kickdrum with a lot of reverb"]
    generate_evaluation(path, devices, models, prompts, params, seed)

    # Music Production Specific Prompts
    path = Path(RESULTS_DIR + "/prompts_evaluation/music_production_specific_prompts")
    prompts = ["an 808 kickdrum", "the amen break", "a 909 snare", "a 303 baseline", "A jungle drum break", "A Juno-106 pad", "Oberheimer OB-Xa string pads"]
    generate_evaluation(path, devices, models, prompts, params, seed)


def main():
    prompts_evaluation()

if __name__ == '__main__':
    main()
