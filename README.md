
# ChromaRAG

## Overview

ChromaRAG is a Retrieval-Augmented Generation (RAG) pipeline that supports multimodal data inputs, such as text, images, and URIs. It is built using FastAPI, ChromaDB, and Docker, with features for API key-based authentication, data validation, background processing, rate limiting, and compliance with GDPR, HIPAA, and SOC 2 regulations.

## Features

- **Multimodal Data Support**: Handles text, images, and URIs in the same embedding space.
- **Swagger/OpenAPI Documentation**: Automatically generated interactive API docs.
- **API Key-based Authentication**: Secures endpoints using API keys.
- **Data Validation**: Ensures data integrity with strict validation rules.
- **Consistency Checks**: Verifies data consistency before updates or deletes.
- **Background Processing**: Offloads time-consuming tasks to background processes.
- **Rate Limiting**: Prevents overloading the API with excessive requests.
- **Compliance**: Implements data encryption, anonymization, and access controls for GDPR, HIPAA, and SOC 2 compliance.

## Folder Structure

```
ChromaRAG/
├── app/
│   ├── main.py            # FastAPI application
│   ├── models.py          # Pydantic models
│   ├── chroma_client.py   # ChromaDB client setup and utilities
│   ├── dependencies.py    # Dependency handling for API key authentication
│   ├── routers/
│   │   ├── __init__.py    # Router initialization
│   │   ├── collections.py # Endpoints related to collections
│   │   └── documents.py   # Endpoints related to documents
│   ├── utils.py           # Utility functions, e.g., embedding functions
│   ├── security.py        # Security functions for encryption and anonymization
│   └── middleware.py      # Middleware for rate limiting
├── Dockerfile             # Dockerfile for the FastAPI app
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project overview and documentation
```

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/ChromaRAG.git
    cd ChromaRAG
    ```

2. **Setup Docker**:
    ```bash
    docker-compose up --build
    ```

3. **Access the API**:
    - The API will be available at `http://localhost:8000`.
    - Swagger UI is available at `http://localhost:8000/docs`.

## Authentication

The API is secured using API keys. Include the API key in the `Authorization` header as follows:

```bash
curl -H "Authorization: Bearer your-secure-api-key" http://localhost:8000/your-endpoint
```

## Data Encryption and Compliance

Sensitive data is encrypted and anonymized using the `cryptography` library to ensure compliance with GDPR, HIPAA, and SOC 2 regulations. Refer to `app/security.py` for implementation details.

## Contributing

Feel free to contribute by submitting a pull request. Ensure that your code adheres to the project's coding standards.

## License

TBA
