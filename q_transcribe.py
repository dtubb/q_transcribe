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
from pdf2image import convert_from_path

def natural_sort_key(filename: str):
    """Generate a sorting key that sorts numbers in a human-friendly way."""
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]

def process_image(image_file: Path, model, processor, prompt: str, device: str, jpg_output_dir: Path, output_dir: Path, output_format: str, max_tokens: int, thumbnail_size: int):
    """Process a single image file and save output to the designated JPG and output folders."""
    
    image_output_file = jpg_output_dir / image_file.name
    image = Image.open(image_file)
    image.save(image_output_file, "JPEG")
    print(f"[green]Saved full-size image {image_output_file.name}")

    thumbnail_image = image.copy()
    thumbnail_image.thumbnail((thumbnail_size, thumbnail_size))
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": thumbnail_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

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

    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    output_file = output_dir / image_file.with_suffix(f".{output_format}").name
    output_file.write_text(output_text[0])
    print(f"[green]Saved {output_file.name}")

def process_pdf(pdf_file: Path, model, processor, prompt: str, device: str, output_format: str, max_tokens: int, thumbnail_size: int, suffix: str, base_folder: Path):
    """Extract and process images from a PDF file, placing them in appropriate transcription folders."""
    
    pdf_name = pdf_file.stem
    jpg_output_dir = base_folder / f"{pdf_name}_transcription_jpg"
    if jpg_output_dir.exists():
        print(f"[yellow]Skipping JPG extraction for {pdf_name} - already done.[/yellow]")
    else:
        jpg_output_dir.mkdir(parents=True, exist_ok=True)
        images = convert_from_path(pdf_file)
        for i, image in enumerate(images):
            image_filename = jpg_output_dir / f"{pdf_name}_page_{i + 1:03d}.jpg"
            image.save(image_filename, "JPEG")

    suffix_str = f"_{suffix}" if suffix else ""
    output_dir = base_folder / f"{pdf_name}_transcription_{output_format}{suffix_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_file in jpg_output_dir.glob("*.jpg"):
        process_image(image_file, model, processor, prompt, device, jpg_output_dir, output_dir, output_format, max_tokens, thumbnail_size)

def transcribe_images_recursive(
    folder: Annotated[Path, typer.Argument(help="Folder containing image or PDF files or subfolders")],
    model_name: Annotated[str, typer.Option(help="Choose Qwen model (e.g., 'Qwen/Qwen2-VL-2B-Instruct' or 'Qwen/Qwen2-VL-7B-Instruct')")] = "Qwen/Qwen2-VL-2B-Instruct",
    output_format: Annotated[str, typer.Option(help="Output format: 'html' or 'md'")] = "html",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens for the model output")] = 1280,
    thumbnail_size: Annotated[int, typer.Option(help="Thumbnail size for the images")] = 750,
    suffix: Annotated[str, typer.Option(help="Additional suffix for subfolders")] = "",
    prompt: Annotated[str, typer.Option(help="Custom prompt for transcription")] = None
):
    if prompt is None:
        if output_format == "md":
            prompt = "Extract text into markdown format. SAY NOTHING ELSE."
        else:
            prompt = "Transcribe text and text elements into valid HTML format. SAY NOTHING ELSE."

    print(f"[green]Transcribing images and PDFs in {folder}")
    print(f"[cyan]Using model {model_name} with {max_tokens} max tokens and thumbnail size {thumbnail_size}px")
    print(f"[cyan]Using custom folder suffix: '{suffix}'")
    print(f"[cyan]Using custom prompt: '{prompt}'")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    if torch.cuda.is_available():
        device = "cuda"
        print("[cyan]Using CUDA for inference.[/cyan]")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[cyan]Using MPS for inference.[/cyan]")
    else:
        device = "cpu"
        print("[cyan]Using CPU for inference.[/cyan]")

    if not folder.exists():
        print("[red]The specified folder does not exist.")
        return

    base_folder = folder.parent / f"{folder.stem}_transcription"
    base_folder.mkdir(parents=True, exist_ok=True)
    
    image_formats = ["*.jpg", "*.jpeg", "*.png"]
    pdf_format = ["*.pdf"]
    image_files = []
    pdf_files = []
    for fmt in image_formats:
        image_files.extend(folder.rglob(fmt))
    for fmt in pdf_format:
        pdf_files.extend(folder.rglob(fmt))

    image_files.sort(key=lambda f: natural_sort_key(f.stem))
    pdf_files.sort(key=lambda f: natural_sort_key(f.stem))

    if not image_files and not pdf_files:
        print("[yellow]No image or PDF files found in the folder or subfolders.")
        return

    for image_file in track(image_files, total=len(image_files)):
        doc_name = image_file.stem
        jpg_output_dir = base_folder / f"{doc_name}_transcription_jpg"
        output_dir = base_folder / f"{doc_name}_transcription_{output_format}{('_' + suffix) if suffix else ''}"
        
        if jpg_output_dir.exists():
            print(f"[yellow]Skipping JPG extraction for {doc_name} - already done.[/yellow]")
        else:
            jpg_output_dir.mkdir(parents=True, exist_ok=True)
            process_image(image_file, model, processor, prompt, device, jpg_output_dir, output_dir, output_format, max_tokens, thumbnail_size)

    for pdf_file in track(pdf_files, total=len(pdf_files)):
        process_pdf(pdf_file, model, processor, prompt, device, output_format, max_tokens, thumbnail_size, suffix, base_folder)

if __name__ == "__main__":
    typer.run(transcribe_images_recursive)
