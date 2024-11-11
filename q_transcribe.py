import typer
from rich import print
from rich.progress import track
from typing_extensions import Annotated
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path
from PIL import Image, ImageOps
import torch
import re
from pdf2image import convert_from_path

def natural_sort_key(filename: str):
    """Generate a sorting key that sorts numbers in a human-friendly way."""
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]

def crop_image(image: Image.Image, crop_percentages: tuple):
    """Crop image by specified percentages from each side (top, right, bottom, left)."""
    width, height = image.size
    top, right, bottom, left = crop_percentages
    left_px = int(left / 100 * width)
    right_px = int(right / 100 * width)
    top_px = int(top / 100 * height)
    bottom_px = int(bottom / 100 * height)
    return image.crop((left_px, top_px, width - right_px, height - bottom_px))

def split_image(image: Image.Image, split_percentage: float = 50.0):
    """Split image vertically at specified percentage."""
    width, height = image.size
    split_point = int((split_percentage / 100) * width)
    left_image = image.crop((0, 0, split_point, height))
    right_image = image.crop((split_point, 0, width, height))
    return left_image, right_image

def binarize_image(image: Image.Image, threshold: int = 128):
    """Convert image to binary (black and white) mode with adjustable threshold."""
    return image.convert("L").point(lambda x: 0 if x < threshold else 255, '1')

def process_image(image_file: Path, model, processor, prompt: str, device: str, output_dir: Path, output_format: str, max_tokens: int, thumbnail_size: int, crop_percentages: tuple = (0, 0, 0, 0), split: bool = False, split_percentage: float = 50.0, skip_transcription: bool = False, skip_existing_images: bool = True, rotate_percentage: float = 0, binarize: bool = False, binarize_threshold: int = 50):
    """Process a single image file with cropping, optional splitting, rotation, and binarization, and save output."""

    image_output_file = output_dir / f"{image_file.stem}.jpg"
    
    if skip_existing_images and image_output_file.exists():
        print(f"[yellow]Skipping existing image {image_output_file.name}[/yellow]")
        return

    image = Image.open(image_file)

    # Apply cropping if specified
    if any(crop_percentages):
        image = crop_image(image, crop_percentages)
        print(f"[green]Cropped image {image_file.name} with margins {crop_percentages}")

    # Apply rotation if specified
    if rotate_percentage:
        image = image.rotate(rotate_percentage, expand=True)
        print(f"[green]Rotated image {image_file.name} by {rotate_percentage} degrees")

    # Apply binarization if enabled
    if binarize:
        image = binarize_image(image, binarize_threshold)
        print(f"[green]Binarized image {image_file.name} with threshold {binarize_threshold}")

    if split:
        # Split image if enabled, process each part separately
        left_image, right_image = split_image(image, split_percentage)
        for part, part_name in zip((left_image, right_image), ("left", "right")):
            part_output_file = output_dir / f"{image_file.stem}_{part_name}.jpg"
            # Only save if the part does not already exist
            if not part_output_file.exists():
                part.save(part_output_file, "JPEG")
                print(f"[green]Saved split image {part_output_file.name}")
                if not skip_transcription:
                    process_thumbnail_and_text(part, part_output_file.stem, model, processor, prompt, device, output_dir, output_format, max_tokens, thumbnail_size)
            else:
                print(f"[yellow]Skipping existing split image {part_output_file.name}[/yellow]")
    else:
        # Save the full-size processed image if splitting is not requested
        image.save(image_output_file, "JPEG")
        print(f"[green]Saved full-size image {image_output_file.name}")
        if not skip_transcription:
            process_thumbnail_and_text(image, image_file.stem, model, processor, prompt, device, output_dir, output_format, max_tokens, thumbnail_size)

def process_thumbnail_and_text(image: Image.Image, filename: str, model, processor, prompt: str, device: str, output_dir: Path, output_format: str, max_tokens: int, thumbnail_size: int):
    """Generate thumbnail, run model inference, and save output."""
    
    # Ensure the image is opened properly
    if isinstance(image, Path):
        image = Image.open(image)  # Open the image from the path
    
    thumbnail_image = image.copy()  # Now this should work because 'image' is a PIL.Image object
    thumbnail_image.thumbnail((thumbnail_size, thumbnail_size))

    messages = [{"role": "user", "content": [{"type": "image", "image": thumbnail_image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    output_file = output_dir / f"{filename}.{output_format}"
    output_file.write_text(output_text[0])
    print(f"[green]Saved transcription {output_file.name}")

def process_pdf(pdf_file: Path, model, processor, prompt: str, device: str, output_format: str, max_tokens: int, thumbnail_size: int, suffix: str, base_folder: Path, crop_percentages: tuple = (0, 0, 0, 0), split: bool = False, split_percentage: float = 50.0, skip_transcription: bool = False, skip_existing_images: bool = True, rotate_percentage: float = 0, binarize: bool = False, binarize_threshold: int = 50):
    """Process images extracted from a PDF, with optional cropping, splitting, rotation, and binarization."""

    pdf_name = pdf_file.stem
    jpg_output_dir = base_folder / f"{pdf_name}_transcription"
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
        process_image(image_file, model, processor, prompt, device, jpg_output_dir, output_dir, output_format, max_tokens, thumbnail_size, crop_percentages, split, split_percentage, skip_transcription, skip_existing_images, rotate_percentage, binarize, binarize_threshold)

def transcribe_images_recursive(
    folder: Annotated[Path, typer.Argument(help="Folder containing image or PDF files or subfolders")],
    model_name: Annotated[str, typer.Option(help="Choose Qwen model (e.g., 'Qwen/Qwen2-VL-2B-Instruct' or 'Qwen/Qwen2-VL-7B-Instruct')")] = "Qwen/Qwen2-VL-2B-Instruct",
    output_format: Annotated[str, typer.Option(help="Output format: 'html' or 'md'")] = "html",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens for the model output")] = 1280,
    thumbnail_size: Annotated[int, typer.Option(help="Thumbnail size for the images")] = 750,
    suffix: Annotated[str, typer.Option(help="Additional suffix for subfolders")] = "",
    prompt: Annotated[str, typer.Option(help="Custom prompt for transcription")] = None,
    crop_percentages: Annotated[str, typer.Option(help="Crop percentages as 'top,right,bottom,left'")] = "0,0,0,0",
    split: Annotated[bool, typer.Option(help="Whether to split the image in half")] = False,
    split_percentage: Annotated[float, typer.Option(help="Percentage to split the image at, if splitting is enabled")] = 50.0,
    skip_transcription: Annotated[bool, typer.Option(help="Skip transcription and model loading for testing image processing")] = False,
    skip_existing_images: Annotated[bool, typer.Option(help="Skip writing images if they already exist")] = True,
    rotate_percentage: Annotated[float, typer.Option(help="Rotate image by specified degrees")] = 0,
    binarize: Annotated[bool, typer.Option(help="Binarize the image (black and white)")] = False,
    binarize_threshold: Annotated[int, typer.Option(help="Threshold for binarization (default is 50, adjust for text brightness/darkness)")] = 50,
    skip_image_export: Annotated[bool, typer.Option(help="Skip image export, just run transcription")] = False  # New flag
):
    if prompt is None:
        prompt = "Extract text into markdown format. SAY NOTHING ELSE." if output_format == "md" else "Transcribe text and text elements into valid HTML format. SAY NOTHING ELSE."
    
    crop_percentages_tuple = tuple(map(int, crop_percentages.split(',')))
    print(f"[green]Transcribing images and PDFs in {folder}")
    print(f"[cyan]Using model {model_name} with {max_tokens} max tokens and thumbnail size {thumbnail_size}px")
    print(f"[cyan]Using custom folder suffix: '{suffix}'")
    print(f"[cyan]Using custom prompt: '{prompt}'")
    print(f"[cyan]Cropping margins with percentages (top, right, bottom, left): {crop_percentages_tuple}")
    print(f"[cyan]Image splitting is {'enabled' if split else 'disabled'} at {split_percentage}%")
    print(f"[cyan]Transcription is {'skipped' if skip_transcription else 'enabled'} for testing image processing")
    print(f"[cyan]Skipping existing images is {'enabled' if skip_existing_images else 'disabled'}")
    print(f"[cyan]Image rotation by {rotate_percentage} degrees is {'enabled' if rotate_percentage else 'disabled'}")
    print(f"[cyan]Binarization is {'enabled' if binarize else 'disabled'} with threshold {binarize_threshold}")

    if not skip_transcription:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[cyan]Using {device.upper()} for inference.[/cyan]")
    else:
        model = processor = device = None  # Skip model loading

    base_folder = folder.parent / f"{folder.stem}_transcription"
    base_folder.mkdir(parents=True, exist_ok=True)

    # If skip_image_export is enabled, look for files directly in the output folder (transcription folder)
    image_files = list(base_folder.glob("*.jpg"))
    pdf_files = list(folder.rglob("*.pdf"))

    image_files.sort(key=lambda f: natural_sort_key(f.stem))
    pdf_files.sort(key=lambda f: natural_sort_key(f.stem))

    if not image_files and not pdf_files:
        print("[yellow]No image or PDF files found in the folder or subfolders.")
        return

    for image_file in track(image_files, total=len(image_files), description="Working on image:"):
        # In skip_image_export mode, only process images in the transcription folder
        process_thumbnail_and_text(image_file, image_file.stem, model, processor, prompt, device, base_folder, output_format, max_tokens, thumbnail_size)

    for pdf_file in track(pdf_files, total=len(pdf_files)):
        process_pdf(pdf_file, model, processor, prompt, device, output_format, max_tokens, thumbnail_size, suffix, base_folder, crop_percentages_tuple, split, split_percentage, skip_transcription, skip_existing_images, rotate_percentage, binarize, binarize_threshold)

if __name__ == "__main__":
    typer.run(transcribe_images_recursive)
