
# q_transcribe 📸📝: Transcribe Typed and Handwritten Text in Images with QWEN VL Models

- Transcribe typed and handwritten text from individual images, entire folders, or nested folders containing images.
- Extracted text from images is saved as individual Markdown or HTML files.
- Choose between different Qwen models for transcription, including Qwen2 and Qwen 7B.

q_transcribe is a Python-based command-line tool that transcribes text from images using QWEN VL models. It supports extracting text from JPG, JPEG, PNG, and PDF formats into Markdown or HTML files.

## Installation Instructions

### Requirements

- **Python 3.7 or later**: Ensure you have Python installed. You can check your version by running `python --version` or `python3 --version` in your terminal.

### Installation Steps

1. **Clone the Repository**: Open your terminal and run the following commands to clone the Q_Transcribe repository:

    ```bash
    git clone https://github.com/dtubb/q_transcribe.git
    cd q_transcribe
    ```

2. **Create a Conda Environment** (recommended): Use the following commands:

    ```bash
    conda create --name q_transcribe python=3.8
    conda activate q_transcribe
    ```

3. **Install Required Packages**: Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Tool**: You can now run q_transcribe using the command line. Use the following command to get a list of available options:

    ```bash
    python q_transcribe.py --help
    ```

## Usage

To use q_transcribe, run the command line interface with the following syntax:

```bash
python q_transcribe.py input_folder --model_name "Qwen/Qwen2-VL-2B-Instruct"
```

Or, you can use the script to run transcription on a single image file:

```bash
python q_transcribe.py input_image.jpg --model_name "Qwen/Qwen2-VL-7B-Instruct"
```

### Arguments

- **input_folder** (Path): Path to the folder containing image or PDF files.
- **--model_name** (str, optional): Specify the Qwen model for transcription (e.g., "Qwen/Qwen2-VL-2B-Instruct"). Defaults to `"Qwen/Qwen2-VL-2B-Instruct"`.

### Options

- **--output_format** (str): Format for output files, either `md` (Markdown) or `html` (HTML). Defaults to Markdown.
- **--max_tokens** (int): Maximum tokens for model output. Default is `1280`.
- **--thumbnail_size** (int): Thumbnail size for image input to the model. Default is `750`.
- **--suffix** (str): Adds an additional suffix to subfolder names for organizational purposes.
- **--prompt** (str): Custom prompt for transcription.
- **--crop_percentages** (str): Specify crop margins as a comma-separated string (top, right, bottom, left).
- **--split**: Enables splitting of images, separating them vertically.
- **--split_percentage** (float): Split percentage (default `50.0`) when splitting images.
- **--skip_transcription**: Skip transcription, enabling testing of image processing without invoking the model.
- **--skip_existing_images**: Skip writing image files if they already exist.
- **--rotate_percentage** (float): Rotate images by a specified degree angle.
- **--binarize**: Enable binarization (black-and-white conversion) with an adjustable threshold.
- **--binarize_threshold** (int): Threshold for binarization (default `50`).
- **--skip_image_export**: Skip image export, only running transcription.

## Image Formats Supported

q_transcribe supports the following image formats:

- `.jpg`
- `.jpeg`
- `.png`
- `.pdf` (For PDFs, each page is extracted into individual images and saved in a subfolder.)

## Output

The transcribed text is saved as individual Markdown or HTML files in designated output directories. Files are named after the input images, with suffixes if specified.

## Credits

This project integrates folder handling and advanced image processing with [Andy Janco's](https://github.com/apjanco) CLI wrapper to Qwen2-VL. For more details, visit the [Qwen2-VL repository](https://github.com/QwenLM/Qwen2-VL).
