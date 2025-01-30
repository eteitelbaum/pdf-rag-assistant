## PDF RAG Assistant

An intelligent document assistant that helps researchers interact with academic papers and other PDF documents using RAG (Retrieval Augmented Generation) technology. This system allows users to query their document collection and receive contextually relevant responses based on the content of their papers.

## Features

- PDF document processing and storage
- Semantic search across academic papers
- Contextual question-answering using RAG
- Support for academic paper analysis and exploration

## Installation

1. Clone the repository

```bash
git clone https://github.com/pdf-rag-assistant
```

2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

1. Place your PDF documents in the `academic_papers` directory
2. Run the main script:

```bash
python main.py
```
3. Interact with your documents through natural language queries

## Project Structure

```text
├── main.py              # Main application entry point
├── academic_papers/     # Directory for PDF documents
├── academic_db/         # Vector store database
└── venv/                # Python virtual environment
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
