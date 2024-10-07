import typer
from rich import print
from rich.progress import track
from typing_extensions import Annotated
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path
from PIL import Image
import torch

def transcribe_images(
    folder: Annotated[Path, typer.Argument(help="Folder containing JPGs")],
    model_name: Annotated[str, typer.Argument(help="HF model name")] = "Qwen/Qwen2-VL-7B-Instruct",
):
    prompt = """Recognize handwritten text in the provided document, along with relevant metadata in typescript. Identify blocks of text that are grouped together, and separate each line of text. Extract all text. SAY NOTHING ELSE. RETURN ONLY PLAIN TEXT."""
    
    print(f"[green]Transcribing images in {folder}")
    print(f"[cyan]Using model {model_name}")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Ensure the folder exists
    if not folder.exists() or not folder.is_dir():
        print("[red]The specified folder does not exist or is not a directory.")
        return
    
    # Find all JPG files in the folder
    jpg_files = list(folder.glob("*.jpg"))

    if not jpg_files:
        print("[yellow]No JPG files found in the folder.")
        return

    # Process each JPG
    for jpg_file in track(jpg_files, total=len(jpg_files)):
        output_file = jpg_file.with_suffix(".md")  # Set output file path

        # Check if output file already exists
        if output_file.exists():
            print(f"[yellow]Skipping {jpg_file.name} - file already exists.[/yellow]")
            continue  # Skip to the next image if the file exists

        image = Image.open(jpg_file)
        image.thumbnail((1000, 1000))
        
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
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "mps")

        # Inference: Generate the output
        generated_ids = model.generate(**inputs, max_new_tokens=1000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Write output to Markdown file
        output_file.write_text(output_text[0])
        print(f"[green]Saved {output_file.name}")

if __name__ == "__main__":
    typer.run(transcribe_images)
