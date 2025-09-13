# RepoTalk

RepoTalk is a command-line interface (CLI) tool that allows you to chat with GitHub repositories using the power of OpenAI's language models.

## Features

- Index public GitHub repositories.
- Chat with indexed repositories to ask questions about the code.
- Uses a local vector store for efficient retrieval of information.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd repo-talk
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```

4.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up your OpenAI API key:**
    Create a `.env` file in the root of the project and add your OpenAI API key:
    ```
    OPENAI_API_KEY=your_api_key_here
    ```

## Usage

The CLI has three main commands:

- `index`: Index a GitHub repository.
- `chat`: Chat with an indexed repository.
- `list`: List all indexed repositories.

### `index`

To index a repository, use the `index` command followed by the repository's HTTPS URL:

```bash
python3 cli.py index https://github.com/user/repo
```

### `chat`

To chat with an indexed repository, use the `chat` command followed by the repository name and your question:

```bash
python3 cli.py chat repo "What is the main purpose of this repository?"
```

### `list`

To see a list of all indexed repositories, use the `list` command:

```bash
python3 cli.py list
```
