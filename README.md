This project is a web-based application built with Streamlit that allows users to upload PDF and Word documents or input text manually to generate summaries using a pre-trained GPT-2 model. The app processes the input text, splits it into manageable chunks, and then generates a summary for each chunk.

## Features

- **PDF and Word Document Processing**: Load and process documents in PDF and DOCX formats.
- **Text Summarization**: Generate summaries using the GPT-2 model, with customizable chunk size and overlap.
- **Interactive Web Interface**: Users can upload files, input text, and adjust settings directly through a web-based interface powered by Streamlit.

## Installation

To get started with this project, you'll need to clone the repository and install the necessary dependencies.

### Create a Virtual Environment

In the root directory of your project, run the following command to create a virtual environment:

```bash
python3 -m venv newenv

### Prerequisites
- Python 3.7 or higher
- `pip` package manager

### Install Dependencies

First, clone the repository:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

### Then, install the required Python packages using pip:

```bash
pip install -r requirements.txt


###To run the application, use the following command:

bash
streamlit run final.py
