from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# Initialize the OpenCLIP embedding function
def get_openclip_embedding_function():
    return OpenCLIPEmbeddingFunction()

# Initialize the ImageLoader for multimodal collections
def get_image_loader():
    return ImageLoader()
