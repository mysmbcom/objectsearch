# ObjectSearch Class

A comprehensive Python class for end-to-end vector embeddings generation from images through image preprocessing.

## Introduction

The `ObjectSearch` class provides a complete pipeline for processing images and generating vector embeddings suitable for object search and similarity matching applications. The class implements a preprocessing pipeline that addresses common challenges in object matching usecases.

### Necessity of Preprocessing (Normalization)
Image preprocessing is crucial for consistent and reliable object search because:

- **Background removal** can interfere with object recognition and similarity matching
- **Varying color of object** affect comaparing object to object
- **Different image sizes and aspect ratios** require standardization for neural network inputs
- **Inconsistent object positioning** affect comaparing object to object

### Each Normalization Step

The preprocessing pipeline consists of three key normalization steps:

#### 1. Background Removal (`normalise_background()`)
- Removes distracting backgrounds using the Bria.ai API
- Converts images to PNG format with transparent backgrounds
- Supports input from bytes, file paths, or URLs

#### 2. Contrast Normalization (`normalise_contrast()`)
- Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Converts images to grayscale for consistent feature extraction

#### 3. Scale Normalization (`normalise_scale()`)
- Crops transparent margins to focus on the object
- Centers the object on a square canvas
- Resizes to target dimensions (default: 224×224 pixels)

### Vectorization

The final step generates high-dimensional vector embeddings using DINO-v2 (Vision Transformer):

- **Model**: DINO-v2 ViT-S/14 (384-dimensional embeddings)
- **Normalization**: L2-normalized vectors for cosine similarity
- **Device Support**: CPU and GPU (CUDA) modes
- **Flexibility**: Supports multiple DINO-v2 model variants, or other vector embeddings models by overriding the generate_vector_embeddings() method.

## Dependencies

The ObjectSearch class requires the following Python libraries:

### Core Dependencies
- **requests** (2.32.4) - HTTP requests for API calls and image downloads
- **numpy** (2.3.1) - Numerical computing and array operations
- **torch** (2.7.1) - PyTorch for deep learning models
- **torchvision** (0.22.1) - Computer vision utilities for PyTorch
- **opencv-python** (4.11.0.86) - Computer vision operations (CLAHE, color space conversion)
- **pillow** (11.2.1) - Image processing and format handling

### System Dependencies
- **libgl1** - OpenGL libraries required by OpenCV
- **libglib2.0-0** - GLib libraries for system operations


## Caveats

### Bria.ai API Limitations
- **Format Support**: Only accepts PNG and JPG image formats
- **API Token Required**: Requires a valid Bria.ai API token for background removal
- **Network Dependency**: Background removal requires internet connectivity
- **Rate Limits**: Subject to Bria.ai API rate limiting and usage quotas
- **Cost**: Background removal operations may incur API costs

### Technical Considerations
- **Memory Usage**: Large images may require significant memory during processing
- **Processing Time**: Background removal via API can add latency to the pipeline (Consifer asynchronous frmaework)

## Customization

The ObjectSearch class is designed for extensibility and customization:

### Overriding Background Removal

You can override the `normalise_background()` method to use open-source alternatives:

```python
class CustomObjectSearch(ObjectSearch):
    def normalise_background(self, image_bytes, **kwargs):
        # Use rembg (open-source background removal)
        import rembg
        from PIL import Image
        from io import BytesIO
        
        # Process image with rembg
        input_image = Image.open(BytesIO(image_bytes))
        output_image = rembg.remove(input_image)
        
        # Convert to bytes
        output_buffer = BytesIO()
        output_image.save(output_buffer, format="PNG")
        return output_buffer.getvalue()
```

### Overriding Vectorization

You can override the `generate_vector_embeddings()` method to use different embedding models:

```python
class CustomObjectSearch(ObjectSearch):
    def generate_vector_embeddings(self, image_bytes, **kwargs):
        # Use CLIP embeddings instead of DINO-v2
        import clip
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Preprocess image
        image = preprocess(Image.open(BytesIO(image_bytes))).unsqueeze(0).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        return image_features.cpu().numpy().astype("float32")
```

### Custom Preprocessing Pipeline

You can also customize the entire pipeline by overriding the `pipeline()` method:

```python
class CustomObjectSearch(ObjectSearch):
    def pipeline(self, **kwargs):
        # Custom preprocessing steps
        image_bytes = kwargs.get('image_bytes')
        
        # Add custom preprocessing
        image_bytes = self.custom_preprocessing(image_bytes)
        
        # Continue with standard pipeline
        return super().pipeline(image_bytes=image_bytes, **kwargs)
    
    def custom_preprocessing(self, image_bytes):
        # Add your custom preprocessing logic here
        return image_bytes
```

## Usage Example

```python
from object_search import ObjectSearch

# Initialize with your Bria.ai API token
api_token = "your_bria_api_token_here"
obj_search = ObjectSearch(api_token)

# Process an image through the complete pipeline
vector = obj_search.pipeline(
    image_path="path/to/your/image.jpg",
    normalize_background=True,
    normalize_contrast=True,
    normalize_scale=True,
    output_intermediate_images=False,
    output_dir="output/",
    model_name="dinov2_vits14",
    mode="cpu"
)

print(f"Generated vector shape: {vector.shape}")
print(f"Vector norm: {np.linalg.norm(vector):.6f}")
```