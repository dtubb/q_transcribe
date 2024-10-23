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
from pdf2image import convert_from_path  # Import for PDF to image conversion

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

def process_pdf(pdf_file: Path, model, processor, prompt: str, device: str):
    """Extract and process images from a PDF file, placing them in a subfolder."""
    pdf_name = pdf_file.stem  # Get the name of the PDF without the extension
    output_dir = pdf_file.parent / pdf_name  # Create a subfolder named after the PDF
    output_dir.mkdir(exist_ok=True)  # Ensure the folder exists

    # Convert each page of the PDF into an image
    images = convert_from_path(pdf_file)
    for i, image in enumerate(images):
        image_filename = output_dir / f"{pdf_name}_{i + 1:03d}.png"  # Name like IMC_AR_1966_001.png
        image.save(image_filename, "PNG")
        process_image(image_filename, model, processor, prompt, device)

def transcribe_images_recursive(
    folder: Annotated[Path, typer.Argument(help="Folder containing image or PDF files or subfolders")],
    model_name: Annotated[str, typer.Argument(help="Choose Qwen model (e.g., 'Qwen/Qwen2-VL-2B-Instruct' or 'Qwen/Qwen2-VL-7B-Instruct')")] = "Qwen/Qwen2-VL-2B-Instruct"
):
    prompt = """Extract text into markdown format. SAY NOTHING ELSE."""

    print(f"[green]Transcribing images and PDFs in {folder}")
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

    # Find all image and PDF files in the folder recursively
    image_formats = ["*.jpg", "*.jpeg", "*.png"]
    pdf_format = ["*.pdf"]
    image_files = []
    pdf_files = []
    for fmt in image_formats:
        image_files.extend(folder.rglob(fmt))  # Recursively search through all subdirectories for images
    for fmt in pdf_format:
        pdf_files.extend(folder.rglob(fmt))  # Recursively search through all subdirectories for PDFs

    # Sort files in natural order
    image_files.sort(key=lambda f: natural_sort_key(f.stem))
    pdf_files.sort(key=lambda f: natural_sort_key(f.stem))

    if not image_files and not pdf_files:
        print("[yellow]No image or PDF files found in the folder or subfolders.")
        return

    # Process each image
    for image_file in track(image_files, total=len(image_files)):
        process_image(image_file, model, processor, prompt, device)

    # Process each PDF
    for pdf_file in track(pdf_files, total=len(pdf_files)):
        process_pdf(pdf_file, model, processor, prompt, device)

if __name__ == "__main__":
    typer.run(transcribe_images_recursive)
