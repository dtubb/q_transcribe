# q_transcribe 📸📝: Transcribe Images with QWEN VL Models

- Flexible Input: Transcribe individual images, entire folders, or nested folders containing images.

- Markdown Output: Extracted text from images is saved as individual Markdown files.

- Model Selection: Choose between different Qwen models for transcription, including Qwen2 and Qwen 7B.

q_transcribe is (really simple) Python-based command-line tool that will transcribe text from images using QWEN VL models. This tool extracts text from JPG, JPEG, PNG formats into markdown files

## Installation Instructions

### Requirements

- Python 3.7 or later: Ensure you have Python installed. You can check your version by running `python --version` or `python3 --version` in your terminal.

### Installation Steps

1. **Clone the Repository**: Open your terminal and run the following command to clone the Q_Transcribe repository:

    ```bash
    git clone https://github.com/dtubb/q_transcribe.git
    cd q_transcribe
    ```

2. **Create a Conda Environment**: It is recommended to create a Conda environment to manage dependencies. You can create one using the following command:

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

Or, you can use the script to run transcription on just a single image file:

```bash
python q_transcribe.py input_image.jpg --model_name "Qwen/Qwen2-VL-7B-Instruct"
```

### Arguments

- **input_folder** (str): Path to the input folder containing image files.
- **input_image** (str): Path to a single image file.
- **--model_name** (str, optional): Specify the model to use for transcription. Defaults to `Qwen/Qwen2-VL-2B-Instruct`.

### Options

- **--help**: Show this message and exit.

## Image Formats Supported

Q_Transcribe supports multiple image formats, including:

- `.jpg`
- `.jpeg`
- `.png`

## Output

The transcribed text is saved as individual Markdown files in the same directory as the input images, with the same base name.
