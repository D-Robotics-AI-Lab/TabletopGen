import os
import base64
import requests
from volcenginesdkarkruntime import Ark
from PIL import Image
import io

def adjust_image_aspect_ratio(image_path, max_ratio=2.5):
    """
    Adjust the image aspect ratio to be within the API's required range (0.33-3.00)
    """
    img = Image.open(image_path)
    width, height = img.size
    aspect_ratio = width / height

    if 0.4 <= aspect_ratio <= max_ratio:
        return img
    
    # Calculate required padding
    if aspect_ratio > max_ratio:
        # Image too wide, need to add padding on top and bottom
        new_height = int(width / max_ratio)
        padding = (new_height - height) // 2
        new_img = Image.new('RGB', (width, new_height), (128, 128, 128))
        new_img.paste(img, (0, padding))
    else:
        # Image too tall, need to add padding on left and right
        new_width = int(height * 0.4)
        padding = (new_width - width) // 2
        new_img = Image.new('RGB', (new_width, height), (128, 128, 128))
        new_img.paste(img, (padding, 0))
    
    return new_img

def image_to_base64(image_path):
    """Convert a local image to base64 encoding, automatically adjusting the aspect ratio"""

    img = adjust_image_aspect_ratio(image_path)

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    image_data = buffer.getvalue()

    base64_str = base64.b64encode(image_data).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"

def generate_object_image(image_path1, image_path2, object_name, output_path, is_main_obj=False, is_multi=False, ark_api_key=None):
    """
    Generate object images using Seedream 4.0
    """

    client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=ark_api_key or os.environ.get("ARK_API_KEY"),
    )
    
    image1_base64 = image_to_base64(image_path1)
    image2_base64 = image_to_base64(image_path2)
    
    if is_main_obj:
        prompt = f'''Please redraw the "{object_name}" object in the first reference image and generate a high-definition image of it. Its proportions, shape, texture, and other appearance features must remain unchanged. Its current front perspective must be maintained, it cannot be rotated. If it is obscured or overlapped by other objects, remove the other objects and draw only the "{object_name}" itself. But do not remove objects that come with it, such as drawers and sinks that come with the table. Fill the background with a solid color.
    If the object is obscured or partly missing, you can use the second scene image as a reference to better identify the object. However, be sure not to mistake other objects in the scene for the object!'''
    elif is_multi:
        prompt = f'''Please redraw the object **in the first reference image** and generate a high-definition image of it. Its proportions, shape, texture, and other appearance features must remain unchanged. If it is obscured or overlapped by other objects(usually the transparent mask in the first reference image is the obstructing object), remove the obstructing objects and draw only the object itself. Fill the background with a solid color, however, do not project the light and shadow of the background color onto the object. The object must maintain its own color, especially transparent objects.
    You can use the second scene image as a reference to better identify the object. However, be sure not to mistake other objects in the scene for the object!'''
    else:
        prompt = f'Please redraw the "{object_name}" object in the first reference image and generate a high-definition image of it. Its proportions, shape, texture, and other appearance features must remain unchanged. If it is obscured or overlapped by other objects(usually the transparent mask in the first reference image is the obstructing object), remove the obstructing objects and draw only the "{object_name}" itself. Fill the background with a solid color, however, do not project the light and shadow of the background color onto the object. The object must maintain its own color, especially transparent objects.'
    
    print(f"Generating object: {object_name}")
    
    # Call API to generate image
    images_response = client.images.generate(
        model="doubao-seedream-4-0-250828",
        prompt=prompt,
        image=[image1_base64, image2_base64],  # Multiple images input
        size="2K",
        response_format="url",  # Return URL format
        watermark=False,  # Do not add watermark
        sequential_image_generation="disabled"  # Disable sequential image generation, generate single image only
    )

    image_url = images_response.data[0].url
    
    response = requests.get(image_url, timeout=120)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f" Image saved to: {output_path}")
    else:
        raise RuntimeError(f"Failed to download image, status code: {response.status_code}")
    
    return image_url

if __name__ == "__main__":
    image_path1 = "input_images/object_1.png"
    image_path2 = "input_images/scene_1.png"
    object_name = "vase"
    output_path = "output_images/object_1_redraw.png"
    
    ark_api_key = "" 
    
    generate_object_image(
        image_path1=image_path1,
        image_path2=image_path2,
        object_name=object_name,
        output_path=output_path,
        is_main_obj=True,
        is_multi=False,
        ark_api_key=ark_api_key
    )
