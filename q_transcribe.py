import typer
from rich import print
from rich.progress import track
from typing_extensions import Annotated
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path
from PIL import Image
import torch
import re

def natural_sort_key(filename: str):
    """Generate a sorting key that sorts numbers in a human-friendly way."""
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]

def process_image(image_file: Path, model, processor, prompt: str, device: str):
    """Process a single image file."""
    output_file = image_file.with_suffix(".md")  # Set output file path

    # Check if output file already exists
    if output_file.exists():
        print(f"[yellow]Skipping {image_file.name} - file already exists.[/yellow]")
        return

    image = Image.open(image_file)
    image.thumbnail((750, 750))

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to the appropriate device
    inputs = inputs.to(device)

    # Inference: Generate the output
    generated_ids = model.generate(**inputs, max_new_tokens=1280)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Write output to Markdown file
    output_file.write_text(output_text[0])
    print(f"[green]Saved {output_file.name}")

def transcribe_images_recursive(
    folder: Annotated[Path, typer.Argument(help="Folder containing image files or subfolders")],
    model_name: Annotated[str, typer.Argument(help="Choose Qwen model (e.g., 'Qwen/Qwen2-VL-2B-Instruct' or 'Qwen/Qwen2-VL-7B-Instruct')")] = "Qwen/Qwen2-VL-2B-Instruct"
):
    prompt = """Extract text into markdown format. SAY NOTHING ELSE."""

    print(f"[green]Transcribing images in {folder}")
    print(f"[cyan]Using model {model_name}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # Determine the appropriate device
    if torch.cuda.is_available():
        device = "cuda"
        print("[cyan]Using CUDA for inference.[/cyan]")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[cyan]Using MPS for inference.[/cyan]")
    else:
        device = "cpu"
        print("[cyan]Using CPU for inference.[/cyan]")

    # Ensure the folder exists
    if not folder.exists():
        print("[red]The specified folder does not exist.")
        return

    # Find all image files in the folder recursively
    image_formats = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for fmt in image_formats:
        image_files.extend(folder.rglob(fmt))  # Recursively search through all subdirectories

    # Sort files in natural order
    image_files.sort(key=lambda f: natural_sort_key(f.stem))

    if not image_files:
        print("[yellow]No image files found in the folder or subfolders.")
        return

    # Process each image
    for image_file in track(image_files, total=len(image_files)):
        process_image(image_file, model, processor, prompt, device)

if __name__ == "__main__":
    typer.run(transcribe_images_recursive)
