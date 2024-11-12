import typer
import os
from difflib import SequenceMatcher
import json
import re
from pathlib import Path
from PIL import Image
from rich import print
from rich.progress import track
from typing_extensions import Annotated
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from pdf2image import convert_from_path

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def natural_sort_key(filename: str):
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]

def crop_image(image: Image.Image, crop_percentages: tuple):
    width, height = image.size
    top, right, bottom, left = crop_percentages
    return image.crop((int(left / 100 * width), int(top / 100 * height), width - int(right / 100 * width), height - int(bottom / 100 * height)))

def chunk_image(image: Image.Image, num_chunks: int = 3, overlap_percentage: float = 5.0):
    width, height = image.size
    chunk_height = height // num_chunks
    overlap = int(chunk_height * (overlap_percentage / 100))
    return [image.crop((0, max(0, i * chunk_height - overlap), width, min(height, (i + 1) * chunk_height + overlap))) for i in range(num_chunks)]

def binarize_image(image: Image.Image, threshold: int = 128):
    return image.convert("L").point(lambda x: 0 if x < threshold else 255, '1')

def remove_duplicate_lines(texts):
    cleaned_texts = []
    previous_last_line = None
    for text in texts:
        lines = text.splitlines()
        if previous_last_line and lines and lines[0] == previous_last_line:
            lines = lines[1:]
        if lines:
            previous_last_line = lines[-1]
        cleaned_texts.append("\n".join(lines))
    return cleaned_texts

from difflib import SequenceMatcher
import json

def clean_up_text_output(combined_text, output_format, fuzzy_threshold=0.8):
    # Join combined text list into a single string and split into lines for comparison
    lines = "\n".join(combined_text).splitlines()
    cleaned_lines = []
    prev_line = None

    for line in lines:
        # Check for similarity with the previous line using the fuzzy matching threshold
        # Split the line into significant segments (phrases) if long and check within phrases
        if prev_line:
            # If either line is long, split into phrases for more fine-grained comparison
            prev_phrases = prev_line.split(", ") if len(prev_line) > 50 else [prev_line]
            curr_phrases = line.split(", ") if len(line) > 50 else [line]
            match_found = any(
                SequenceMatcher(None, p, q).ratio() > fuzzy_threshold
                for p in prev_phrases for q in curr_phrases
            )
            if match_found:
                continue  # Skip line if it is highly similar to the previous line

        cleaned_lines.append(line)
        prev_line = line

    # Join the cleaned lines back into a single text based on the desired format
    final_output = "\n".join(cleaned_lines)

    # Format-specific adjustments if needed (json, md, html, txt)
    if output_format == "json":
        try:
            final_output = json.dumps({"text": final_output}, indent=2)
        except json.JSONDecodeError:
            raise ValueError("Failed to format text as JSON.")
    elif output_format == "md":
        final_output = f"```\n{final_output}\n```"
    elif output_format == "html":
        final_output = "<html><body><p>{}</p></body></html>".format(final_output.replace("\n", "</p><p>"))

    return final_output.strip()


def process_image(image_file: Path, model, processor, prompt: str, device: str, output_dir: Path, output_format: str, max_tokens: int, thumbnail_size: int, crop_percentages: tuple = (0, 0, 0, 0), chunk: bool = False, num_chunks: int = 3, overlap_percentage: float = 5.0, skip_transcription: bool = False):
    image = Image.open(image_file)
    if any(crop_percentages):
        image = crop_image(image, crop_percentages)
        print(f"[green]Cropped image {image_file.name} with margins {crop_percentages}")

    output_texts = []
    if chunk:
        chunks = chunk_image(image, num_chunks=num_chunks, overlap_percentage=overlap_percentage)
        for idx, chunk in enumerate(chunks):
            chunk_label = f"{image_file.stem}_chunk_{idx+1}"
            chunk_output_file = output_dir / f"{chunk_label}.txt"
            
            if chunk_output_file.exists():
                print(f"[yellow]Skipping transcription for {chunk_label} as it already exists.[/yellow]")
                with open(chunk_output_file, 'r') as file:
                    output_texts.append(file.read())
            else:
                print(f"[blue]Processing chunk {idx + 1}/{num_chunks} of image {image_file.name} as {chunk_label}[/blue]")
                if not skip_transcription:
                    chunk_output_text = process_thumbnail_and_text(chunk, chunk_label, model, processor, prompt, device, output_dir, output_format, max_tokens, thumbnail_size)
                    output_texts.append(chunk_output_text)
                    with open(chunk_output_file, 'w') as file:
                        file.write(chunk_output_text)
                    print(f"[green]Saved transcription for {chunk_label} as {chunk_output_file}[/green]")
                else:
                    print(f"[yellow]Skipping transcription for chunk {idx + 1} of {image_file.name} due to skip_transcription flag.")

        if not skip_transcription:
            combined_text = remove_duplicate_lines(output_texts)
            # Update the call to clean_up_text_output to match the new signature
            final_output = clean_up_text_output(combined_text, output_format)
            
            output_file = output_dir / f"{image_file.stem}.{output_format}"
            if output_format == "json":
                json_data = json.loads(final_output)
                output_file.write_text(json.dumps(json_data, indent=2))
            else:
                output_file.write_text(final_output)
            print(f"[green]Saved consolidated transcription {output_file.name}")

    elif not skip_transcription:
        process_thumbnail_and_text(image, image_file.stem, model, processor, prompt, device, output_dir, output_format, max_tokens, thumbnail_size)
    
    if skip_transcription or not chunk:
        output_image_file = output_dir / f"{image_file.stem}.jpg"
        image.save(output_image_file, "JPEG")
        print(f"[green]Saved image to {output_image_file}")

def process_thumbnail_and_text(image: Image.Image, filename: str, model, processor, prompt: str, device: str, output_dir: Path, output_format: str, max_tokens: int, thumbnail_size: int):
    thumbnail_image = image.copy()
    thumbnail_image.thumbnail((thumbnail_size, thumbnail_size))
    messages = [{"role": "user", "content": [{"type": "image", "image": thumbnail_image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0].strip()

def transcribe_images_recursive(
    folder: Annotated[Path, typer.Argument(help="Folder containing image or PDF files or subfolders")],
    model_name: Annotated[str, typer.Option(help="Choose Qwen model (e.g., 'Qwen/Qwen2-VL-2B-Instruct' or 'Qwen/Qwen2-VL-7B-Instruct')")] = "Qwen/Qwen2-VL-2B-Instruct",
    output_format: Annotated[str, typer.Option(help="Output format: 'html', 'md', 'json', or 'txt'")] = "html",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens for the model output")] = 1280,
    thumbnail_size: Annotated[int, typer.Option(help="Thumbnail size for the images")] = 750,
    suffix: Annotated[str, typer.Option(help="Additional suffix for subfolders")] = "",
    prompt: Annotated[str, typer.Option(help="Custom prompt for transcription")] = None,
    crop_percentages: Annotated[str, typer.Option(help="Crop percentages as 'top,right,bottom,left'")] = "0,0,0,0",
    chunk: Annotated[bool, typer.Option(help="Whether to chunk the image into overlapping parts")] = False,
    num_chunks: Annotated[int, typer.Option(help="Number of chunks if chunking is enabled")] = 3,
    overlap_percentage: Annotated[float, typer.Option(help="Percentage overlap between chunks")] = 5.0,
    skip_transcription: Annotated[bool, typer.Option(help="Skip transcription and model loading for testing image processing")] = False
):
    if prompt is None:
        if output_format == "md":
            prompt = "Extract text lines. RETURN ONLY VALID MARKDOWN. SAY NOTHING ELSE."
        elif output_format == "json":
            prompt = "Extract text lines. RETURN ONLY VALID JSON. SAY NOTHING ELSE."
        elif output_format == "txt":
            prompt = "Extract text lines. RETURN ONLY PLAIN TEXT. SAY NOTHING ELSE."
        else:
            prompt = "Extract text lines. RETURN ONLY VALID HTML. SAY NOTHING ELSE."
    
    crop_percentages_tuple = tuple(map(int, crop_percentages.split(',')))
    print(f"[green]Transcribing images and PDFs in {folder} with {output_format} format.")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto") if not skip_transcription else None
    processor = AutoProcessor.from_pretrained(model_name) if not skip_transcription else None
    
    suffix_str = f"_{suffix}" if suffix else ""
    base_folder = folder.parent / f"{folder.stem}_transcription{suffix_str}"
    base_folder.mkdir(parents=True, exist_ok=True)
    image_files = sorted(folder.rglob("*.jpg"), key=lambda f: natural_sort_key(f.stem))
    for image_file in track(image_files, description="Processing images"):
        process_image(image_file, model, processor, prompt, device, base_folder, output_format, max_tokens, thumbnail_size, crop_percentages_tuple, chunk, num_chunks, overlap_percentage, skip_transcription)

if __name__ == "__main__":
    typer.run(transcribe_images_recursive)
