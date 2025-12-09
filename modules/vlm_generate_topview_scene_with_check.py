"""
Generate a top view of the scene and use GPT to determine if the generated top view is acceptable.
"""

import os
import base64
import httpx
import openai
from PIL import Image
from io import BytesIO
import PIL.Image
from volcenginesdkarkruntime import Ark
import base64
import requests
from modules.setup_openai_client import setup_openai_client


def judge_topview_quality(original_image_path, generated_image_path, openai_client):
    """
    Use GPT to determine if the generated top view meets the standards
    """
    try:
        with open(original_image_path, "rb") as image_file:
            original_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
        with open(generated_image_path, "rb") as image_file:
            generated_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = """Please compare these two images: the first is the original scene, and the second is the generated top view. Determine if the generated top view meets the following criteria:
1. The viewpoint is a direct overhead vertical view (90 degrees)
2. The tabletop appears as a regular rectangle without perspective distortion
3. The table legs are completely invisible
4. The items on the tabletop maintain their original layout and position (compared to the first image) without any additional objects
5. The background is filled with a solid color

Please only answer "YES" or "NO". "YES" means all criteria are met, "NO" means one or more criteria are not met."""

        input_messages = [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{original_base64}" } },
                    { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{generated_base64}" } }
                ]
            }
        ]

        response = openai_client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=input_messages,
        )
        
        result = response.choices[0].message.content.strip().upper()
        return result == "YES"
        
    except Exception as e:
        print(f"GPT judgment failed: {e}")
        return False
    
def image_to_base64(image_path):
    """Convert a local image to base64 encoding"""
    with open(image_path, "rb") as f:
        image_data = f.read()

    ext = os.path.splitext(image_path)[1].lower().replace('.', '')
    if ext == 'jpg':
        ext = 'jpeg'
    base64_str = base64.b64encode(image_data).decode('utf-8')
    return f"data:image/{ext};base64,{base64_str}"
    
def generate_topview_with_seedream(image_path, output_path, prompt, ark_api_key=None):
    client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=ark_api_key or os.environ.get("ARK_API_KEY"),
    )
    
    image1_base64 = image_to_base64(image_path)

    images_response = client.images.generate(
        model="doubao-seedream-4-0-250828",
        prompt=prompt,
        image = [image1_base64],
        size="2K",
        response_format="url",
        watermark=False,
        sequential_image_generation="disabled"
    )
    
    image_url = images_response.data[0].url
    
    response = requests.get(image_url, timeout=120)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
    else:
        raise RuntimeError(f"Failed to download image, status code: {response.status_code}")
    
    return image_url


def generate_topview_scene(input_image_path, output_image_path, seedream_api_key, openai_api_key, proxy_url, base_url, max_attempts=10):
    """
    Generate a top-down view of the scene.
    """
    try:
        if not os.path.exists(input_image_path):
            print(f"Input image does not exist: {input_image_path}")
            return False
        
        openai_client = setup_openai_client(openai_api_key, proxy_url, base_url)
        
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        
        temp_files = []

        text_input = """Generate **a perfectly orthographic top-down shot of the table**, as if taken by a camera directly above at 90 degrees with no perspective distortion. 
        The tabletop should be a flat rectangle fully obscuring its legs. 
        **Keep all items exactly in their current positions and orientations. Do not add new objects. Do not change the aspect ratio of the table. ** 
        The background around the desk should be a solid neutral color."""
        
        for attempt in range(1, max_attempts + 1):
            try:
                base_path = output_image_path.rsplit('.', 1)
                temp_path = f"{base_path[0]}_{attempt}.{base_path[1]}"
                temp_files.append(temp_path)
                
                image_url = generate_topview_with_seedream(
                    input_image_path, 
                    temp_path, 
                    text_input, 
                    seedream_api_key
                )
                
                print(f"Seedream generation {attempt} saved to: {temp_path}")
                
                print("Using GPT to judge image quality...")
                is_qualified = judge_topview_quality(input_image_path, temp_path, openai_client)
                
                if is_qualified:
                    import shutil
                    shutil.copy2(temp_path, output_image_path)
                    print(f"✓ Final top view saved to: {output_image_path}")
                    return True
                else:
                    print(f"✗ Seedream generation {attempt} did not meet standards")
                    
            except Exception as e:
                print(f"Seedream generation {attempt} failed: {e}")
        
        print(f"✗ Seedream failed to generate a standard top view after {max_attempts} attempts")
        
        # Clean up temporary files generated by Seedream
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return False
        
    except Exception as e:
        print(f"Failed to generate top view: {e}")
        return False
    
    
if __name__ == '__main__':
    try:
        pipeline_dir = os.path.dirname(os.path.abspath(__file__))

        id=1
        input_image_path = f"output_scene/scene_{id}/comfy_image/refined_scene_image.png"
        output_image_path = f"output_scene/scene_{id}/output_assets/image/topview_scene.png"
        
        seedream_api_key = ""  # Replace with your Seedream API Key
        openai_api_key = "sk-xxxxxx"  # Replace with your OpenAI API Key
        proxy_url = "http://your-proxy-url:port"  # Replace with your proxy URL
        base_url = "https://api.openai.com/v1"  # Replace with your OpenAI API base URL

        print("Starting to generate scene top view...")
        success = generate_topview_scene(input_image_path, output_image_path, seedream_api_key, openai_api_key, proxy_url, base_url)
        
            
    except Exception as e:
        print(f"Main function execution failed: {e}")
