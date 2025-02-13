# Docs-Helper

Docs-Helper is a documentation assistant bot built using LangChain, Pinecone, and Streamlit. It helps users interact with documentation through natural language queries and provides relevant information from the documentation.

## Features

- **Natural Language Queries**: Ask questions in natural language and get relevant answers from the documentation.
- **Chat History**: Maintains a history of user queries and bot responses for context-aware interactions.
- **Source Tracking**: Provides sources for the information retrieved from the documentation.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/docs-helper.git
    cd docs-helper
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```


3. Set up environment variables:
    - Create a `.env` file in the root directory of the project.
    - Add the following environment variables:
        ```properties
        PINECONE_API_KEY=your_pinecone_api_key
        PINECONE_INDEX_NAME=your_pinecone_index_name
        LANGCHAIN_API_KEY=your_langchain_api_key
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_PROJECT=Docs-Helper
        ```

## Usage

1. **Ingest Documentation**:
    - Run the `ingestion.py` script to load and index the documentation:
        ```sh
        python ingestion.py
        ```

2. **Run the Streamlit App**:
    - Start the Streamlit app to interact with the documentation assistant bot:
        ```sh
        streamlit run main.py
        ```

3. **Interact with the Bot**:
    - Open the Streamlit app in your browser and enter your queries in the prompt input field. The bot will respond with relevant information from the documentation.

## Project Structure

- `main.py`: The main Streamlit app for interacting with the bot.
- `backend/core.py`: Core logic for running the language model and retrieving information.
- `ingestion.py`: Script for loading and indexing the documentation.
- `.env`: Environment variables configuration file.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
