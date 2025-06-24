import requests
import numpy as np
import torch
from torchvision import transforms
from io import BytesIO
from typing import Optional, Any
import os
import time
import torch.nn.functional as F
from PIL import Image
import cv2


class ObjectSearch:
    """
    End-to-end helper for:
      1. Background removal (Bria API)
      2. Contrast equalisation
      3. Object-centred square resize ⟶ 224 × 224
      4. DINO-v2 ViT-S/14 embedding (384-D, L2-normalised)
    """

    def __init__(self, remove_background_api_token: str) -> None:
        """
        Initialize the ObjectSearch instance with Bria API credentials.

        Parameters
        ----------
        remove_background_api_token : str
            API token for the Bria background removal service.
        """
        self.remove_background_api_token = remove_background_api_token
        self.remove_background_api_url = (
            "https://engine.prod.bria-api.com/v1/background/remove"
        )

    def normalise_background(
        self,
        image_bytes: Optional[bytes] = None,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        output_image: bool = False,
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        Remove background from an image using the Bria API.

        This method supports multiple input types and removes the background
        from images, returning the result as PNG bytes with transparent background.
        The alpha channel is binarized with a threshold of 128.

        Parameters
        ----------
        image_bytes : bytes, optional
            Raw image bytes. Mutually exclusive with image_path and image_url.
        image_path : str, optional
            Path to the image file. Mutually exclusive with image_bytes and image_url.
        image_url : str, optional
            URL to the image. Mutually exclusive with image_bytes and image_path.
        output_image : bool, default=False
            If True, save the processed image to disk.
        output_path : str, optional
            Destination path for saved image. If None and output_image=True,
            a default name is generated based on input type.

        Returns
        -------
        bytes
            PNG bytes of the image with background removed (RGBA format).

        Raises
        ------
        ValueError
            If no image input is provided or multiple inputs are provided.
        Exception
            If the Bria API request fails or returns invalid response.
        """
        # Step 1: Prepare image input
        if image_bytes:
            # Handle binary input directly
            image_data = BytesIO(image_bytes)
        elif image_url:
            # Handle URL input
            image_data = BytesIO(requests.get(image_url).content)
        elif image_path:
            # Handle file path input
            with open(image_path, "rb") as f:
                image_data = BytesIO(f.read())
        else:
            raise ValueError(
                "Either image_bytes, image_path, or image_url must be provided"
            )

        # Step 2: Upload to Bria API
        files = {"file": ("image.jpg", image_data, "image/jpeg")}
        headers = {"api_token": self.remove_background_api_token}

        response = requests.post(
            self.remove_background_api_url, headers=headers, files=files
        )
        response.raise_for_status()

        # Step 3: Parse the result URL
        result_url = response.json().get("result_url")
        if not result_url:
            raise Exception("No 'result_url' found in API response.")

        # Step 4: Download the actual image from the CDN
        img_response = requests.get(result_url)
        img_response.raise_for_status()
        image = Image.open(BytesIO(img_response.content)).convert("RGBA")

        # Step 5: Binarize alpha channel with hardcoded threshold of 128
        data = np.array(image)
        threshold = 128  # Hardcoded default value
        data[:, :, 3] = np.where(data[:, :, 3] > threshold, 255, 0)

        # Step 6: Return result as PNG bytes
        output_buffer = BytesIO()
        Image.fromarray(data).save(output_buffer, format="PNG")
        result_bytes = output_buffer.getvalue()

        # Step 7: Save image if requested
        if output_image:
            if output_path is None:
                # Generate default output path based on input
                if image_path:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_path = f"{base_name}_no_bg.png"
                elif image_url:
                    # For URL inputs, use a timestamp-based name
                    timestamp = int(time.time())
                    output_path = f"url_image_{timestamp}_no_bg.png"
                elif image_bytes:
                    # For binary inputs, use a timestamp-based name
                    timestamp = int(time.time())
                    output_path = f"binary_image_{timestamp}_no_bg.png"
                else:
                    raise ValueError(
                        "output_path must be provided when output_image=True and no image_path is available"
                    )

            # Ensure the output directory exists
            os.makedirs(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True,
            )

            # Save the image
            with open(output_path, "wb") as f:
                f.write(result_bytes)
            print(f"✅ Background removed image saved to: {output_path}")

        return result_bytes

    def normalise_contrast(
        self,
        image_bytes: bytes,
        output_image: bool = False,
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        Convert the image to a contrast-equalised, single-channel (grayscale) PNG.

        Parameters
        ----------
        image_bytes : bytes
            Input image in bytes (e.g. from `remove_background()`).
        output_image : bool, optional
            If True, export the processed image to `output_path`.
        output_path : str, optional
            Destination path.  If omitted a timestamp-based name is generated.

        Returns
        -------
        bytes
            PNG bytes of the contrast-normalised image (grayscale,
            alpha preserved if present).
        """
        # Internal defaults for CLAHE parameters
        use_clahe: bool = True
        clip_limit: float = 2.0
        tile_grid_size: tuple[int, int] = (8, 8)

        # 1. Decode bytes ➞ ndarray (note: cv2 expects BGR)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img_bgr is None:
            raise ValueError("Could not decode image bytes – are they valid?")

        # Detect alpha
        has_alpha = img_bgr.shape[2] == 4 if img_bgr.ndim == 3 else False
        if has_alpha:
            bgr, alpha = img_bgr[:, :, :3], img_bgr[:, :, 3]
        else:
            bgr = img_bgr

        # 2. BGR ➞ grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 3. Equalise contrast
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            eq = clahe.apply(gray)
        else:
            eq = cv2.equalizeHist(gray)

        # 4. Re-assemble image (replicate grayscale ➞ 3-channel for ViT if needed)
        eq_bgr = cv2.merge([eq, eq, eq])

        if has_alpha:
            eq_rgba = cv2.merge([eq_bgr, alpha])
            final = eq_rgba
            cv2_type = cv2.IMWRITE_PNG_COMPRESSION
        else:
            final = eq_bgr
            cv2_type = cv2.IMWRITE_PNG_COMPRESSION

        # 5. Encode back to PNG bytes
        success, encoded = cv2.imencode(".png", final, [cv2_type, 9])
        if not success:
            raise RuntimeError("PNG encoding failed")
        result_bytes = encoded.tobytes()

        # 6. Optional file output
        if output_image:
            if output_path is None:
                ts = int(time.time())
                output_path = f"normalised_{ts}.png"
            os.makedirs(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True,
            )
            with open(output_path, "wb") as f:
                f.write(result_bytes)
            print(f"✅ Contrast-normalised image saved to: {output_path}")

        return result_bytes

    def normalise_scale(
        self,
        image_bytes: bytes,
        output_image: bool = False,
        output_path: Optional[str] = None,
        target_size: int = 224,
    ) -> bytes:
        """
        Crop away transparent margins, centre-pad to a square canvas, and
        resize to `target_size`×`target_size`.

        Parameters
        ----------
        image_bytes : bytes
            RGBA (or RGB) image data with transparent background.
        output_image : bool, optional
            If True, also save the result to disk.
        output_path : str, optional
            Path to write the file.  If omitted, a timestamped name is used.
        target_size : int, optional
            Final square side length (e.g. 224 for ViT).

        Returns
        -------
        bytes
            PNG bytes of the square, resized image.
        """
        # Internal defaults
        threshold = 0
        keep_alpha = True
        resample = Image.Resampling.BICUBIC

        # 1. Decode to PIL RGBA
        with BytesIO(image_bytes) as bf:
            img = Image.open(bf).convert("RGBA")

        data = np.array(img)
        alpha = data[..., 3]

        # 2. Find bounding box of non-transparent pixels
        ys, xs = np.where(alpha > threshold)
        if len(xs) == 0 or len(ys) == 0:  # fully transparent?
            raise ValueError("No non-transparent pixels found – cannot crop.")

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))

        # 3. Pad to square
        w, h = cropped.size
        side = max(w, h)
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        offset = ((side - w) // 2, (side - h) // 2)
        square.paste(cropped, offset, mask=cropped)

        # 4. Resize
        resized = square.resize((target_size, target_size), resample=resample)

        # 5. Optionally drop alpha
        if not keep_alpha:
            resized = resized.convert("RGB")

        # 6. Encode to bytes
        out_bf = BytesIO()
        resized.save(out_bf, format="PNG")
        result_bytes = out_bf.getvalue()

        # 7. Optional file save
        if output_image:
            if output_path is None:
                ts = int(time.time())
                postfix = "rgb" if not keep_alpha else "rgba"
                output_path = f"normalised_scale_{ts}_{postfix}.png"
            os.makedirs(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True,
            )
            with open(output_path, "wb") as f:
                f.write(result_bytes)
            print(f"✅ Scale-normalised image saved to: {output_path}")

        return result_bytes

    def generate_vector_embeddings(
        self,
        image_bytes: bytes,
        model_name: str = "dinov2_vits14",  # PyTorch Hub model name
        mode: str = "cpu",  # "cpu" or "gpu"
    ) -> np.ndarray:
        """
        Produce a single-vector, L2-normalised embedding from raw image bytes using PyTorch Hub.

        Parameters
        ----------
        image_bytes : bytes
            PNG/JPEG bytes of the image (background already removed / normalised).
        model_name : str, optional
            PyTorch Hub model name, e.g. "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14".
        mode : {'cpu', 'gpu'}, optional
            Compute device.  'gpu' picks the first CUDA device if available.

        Returns
        -------
        np.ndarray
            1-D float32 vector (already unit-length).
        """
        # 1. Select device
        device = (
            torch.device("cuda:0")
            if mode.lower() == "gpu" and torch.cuda.is_available()
            else torch.device("cpu")
        )

        # 2. Load model from PyTorch Hub
        model: Any = torch.hub.load("facebookresearch/dinov2", model_name)
        model.eval().to(device)

        # 3. Image preprocessing transform
        transform = transforms.Compose(
            [
                transforms.Resize(244),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        # 4. Prepare input
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_tensor: torch.Tensor = transform(img)  # Apply transform first
        img_tensor = img_tensor.unsqueeze(0).to(
            device
        )  # Add batch dimension and move to device

        # 5. Forward pass
        with torch.no_grad():
            outputs: torch.Tensor = model(img_tensor)

            # DINO-v2 models return features directly
            vec: torch.Tensor = outputs

            # 6. L2-normalise
            vec = F.normalize(vec, p=2, dim=-1)

        vec_cpu = vec.squeeze().cpu()
        return vec_cpu.numpy().astype("float32")

    def pipeline(
        self,
        image_bytes: Optional[bytes] = None,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None,
        normalize_background: bool = True,
        normalize_contrast: bool = True,
        normalize_scale: bool = True,
        output_intermediate_images: bool = False,
        output_dir: Optional[str] = None,
        model_name: str = "dinov2_vits14",
        mode: str = "cpu",
        **normalization_kwargs,
    ) -> np.ndarray:
        """
        Complete pipeline to process an image and generate vector embeddings.

        This method combines all preprocessing steps:
        1. Background removal (optional)
        2. Contrast normalization (optional)
        3. Scale normalization (optional)
        4. Vector embedding generation

        Parameters
        ----------
        image_bytes : bytes, optional
            Raw image bytes. Mutually exclusive with image_url and image_path.
        image_url : str, optional
            URL to the image. Mutually exclusive with image_bytes and image_path.
        image_path : str, optional
            Path to the image file. Mutually exclusive with image_bytes and image_url.
        normalize_background : bool, default=True
            Whether to remove background using Bria API.
        normalize_contrast : bool, default=True
            Whether to normalize contrast using CLAHE.
        normalize_scale : bool, default=True
            Whether to normalize scale (crop, pad, resize to 224x224).
        output_intermediate_images : bool, default=False
            Whether to save intermediate processed images.
        output_dir : str, optional
            Directory to save intermediate images. If None, uses current directory.
        model_name : str, default="dinov2_vits14"
            PyTorch Hub model name for embedding generation.
        mode : str, default="cpu"
            Compute device ("cpu" or "gpu").
        **normalization_kwargs
            Additional keyword arguments passed to normalization methods:
            - For contrast normalization: use_clahe, clip_limit, tile_grid_size
            - For scale normalization: target_size, threshold, keep_alpha, resample

        Returns
        -------
        np.ndarray
            1-D float32 vector (already unit-length).

        Raises
        ------
        ValueError
            If no image input is provided or multiple inputs are provided.
        """
        # Validate input parameters
        input_count = sum(
            [image_bytes is not None, image_url is not None, image_path is not None]
        )

        if input_count == 0:
            raise ValueError(
                "Must provide exactly one of: image_bytes, image_url, or image_path"
            )
        elif input_count > 1:
            raise ValueError(
                "Must provide exactly one of: image_bytes, image_url, or image_path"
            )

        # Initialize current image bytes
        current_image_bytes: bytes

        # Step 1: Get initial image bytes
        if image_bytes is not None:
            current_image_bytes = image_bytes
        elif image_url is not None:
            # Download image from URL
            response = requests.get(image_url)
            response.raise_for_status()
            current_image_bytes = response.content
        elif image_path is not None:
            # Read image from file
            with open(image_path, "rb") as f:
                current_image_bytes = f.read()
        else:
            # This should never happen due to validation above, but for type safety
            raise ValueError("No valid image input provided")

        # Step 2: Background removal (if enabled)
        if normalize_background:
            print("🔄 Removing background...")
            current_image_bytes = self.normalise_background(
                image_bytes=current_image_bytes,
                output_image=output_intermediate_images,
                output_path=(
                    f"{output_dir}/step1_background_removed.png" if output_dir else None
                ),
            )

        # Step 3: Contrast normalization (if enabled)
        if normalize_contrast:
            print("🔄 Normalizing contrast...")
            current_image_bytes = self.normalise_contrast(
                image_bytes=current_image_bytes,
                output_image=output_intermediate_images,
                output_path=(
                    f"{output_dir}/step2_contrast_normalized.png"
                    if output_dir
                    else None
                ),
            )

        # Step 4: Scale normalization (if enabled)
        if normalize_scale:
            print("🔄 Normalizing scale...")
            current_image_bytes = self.normalise_scale(
                image_bytes=current_image_bytes,
                output_image=output_intermediate_images,
                output_path=(
                    f"{output_dir}/step3_scale_normalized.png" if output_dir else None
                ),
            )

        # Step 5: Generate vector embeddings
        print("🔄 Generating vector embeddings...")
        vector = self.generate_vector_embeddings(
            image_bytes=current_image_bytes,
            model_name=model_name,
            mode=mode,
        )

        print("✅ Pipeline completed successfully!")
        return vector


# def test_background_removal():
#     """
#     Simple test function to remove background from ./data_raw/outback1.jpeg
#     and save the result as ./data_raw/outback1_no_bg.png
#     """
#     # Initialize ObjectSearch with your API token
#     # You'll need to replace this with your actual API token
#     api_token = "a9a26987be374cdeb1d14cdf231b5215"  # Replace with actual token
#     obj_search = ObjectSearch(api_token)

#     # Remove background from the image with automatic saving
#     image_bg_removed = obj_search.normalise_background(
#         image_path="project_subaru/data_raw/outback1.jpeg",
#         output_image=True,
#         output_path="project_subaru/output/outback1_no_bg.png",
#     )

#     # Normalise contrast from the image with automatic saving
#     image_contrast_normalised = obj_search.normalise_contrast(
#         image_bytes=image_bg_removed,
#         output_image=True,
#         output_path="project_subaru/output/outback1_no_bg_contrast_normalised.png",
#     )

#     image_scale_normalised = obj_search.normalise_scale(
#         image_bytes=image_contrast_normalised,
#         output_image=True,
#         output_path="project_subaru/output/outback1_no_bg_contrast_normalised_scale.png",
#     )

#     vector_embeddings = obj_search.generate_vector_embeddings(
#         image_bytes=image_scale_normalised,
#     )
#     print(vector_embeddings)


# def test_pipeline():
#     """
#     Test function to demonstrate the new pipeline method with different input types.
#     """
#     # Initialize ObjectSearch with your API token
#     api_token = "a9a26987be374cdeb1d14cdf231b5215"  # Replace with actual token
#     obj_search = ObjectSearch(api_token)

#     print("=== Testing Pipeline with Image Path ===")
#     # Test 1: Using image path
#     vector1 = obj_search.pipeline(
#         image_path="data/outback1.jpeg",
#         normalize_background=True,
#         normalize_contrast=True,
#         normalize_scale=True,
#         output_intermediate_images=True,
#         output_dir="project_subaru/output/pipeline_test",
#         model_name="dinov2_vits14",
#         mode="cpu",
#     )
#     print(f"Vector shape: {vector1.shape}")
#     print(f"Vector type: {type(vector1)}")
#     print(f"Vector norm: {np.linalg.norm(vector1):.6f}")

#     print("\n=== Testing Pipeline with Image URL ===")


# if __name__ == "__main__":
#     # test_background_removal()
#     test_pipeline()
