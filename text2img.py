"""
Generate scene images based on text input.
    1. Generate prompt for scene description.
    2. Call Seedream API to generate image.
    3. Save to scene_image directory and record in input_text.json.
"""

import json
import os
import requests
import argparse
from volcenginesdkarkruntime import Ark
from modules.setup_openai_client import setup_openai_client


def generate_scene_prompt(scene_description):
    """
    Generate asset list json based on scene description
    """
    client = setup_openai_client()
    prompt = f"""
You are a 3D scene layout prompt designer. Your task is to Infer the possible items based on the scene description and layout the items. 
**Rules**: 
- First of all, a complete and head-on(front) table is needed. Every parts of the table must be very easy to be seen, especially any of the table foot!
- The angle more than 50 degrees and less than 60 degrees above the table.
- The objects placed should preferably be individual objects, not combined objects.
- "completely white background", " a head-on(front) table and every parts of the table is needed" and "The longer side of the table must be level! The bottommost foot of the desk's front legs must be fully displayed" must be added in end of the prompt. 
- Make the layout of objects more natural, ensure semantic rationality, and conform to real-world logic. When two or more items are semantically related, arrange them according to real-world logic. For example, place the pen to the right of the notebook, the mouse to the right of the keyboard, and the laptop in front of the keyboard. Avoid lining up all objects side by side.
- Ensure object independence for segmentation: To facilitate later segmentation, avoid arrangements where one object is contained within another. For example, pens should be laid directly on the desk, NOT placed inside a cup or pen holder. 
- All items should be placed separately and adjacent to each other.But when there are two or more same items, they must not be placed too close to each other，avoiding being attached together.
- Make sure that all possible objects are on the table.And all items on the table can be clearly seen.
- Keep in mind that the full view of the table and all items can be seen completely.
- Make sure there are only table and all objects on the table in the picture, without chairs etc.
- Avoid items that are too thin, such as a piece of paper. Avoid the same thing. Avoid a single fragile label sticker.
- If no specific items are specified, the number of items should not exceed seven.
Only output a short final prompt. The scene description is :"{scene_description}" """

    response = client.chat.completions.create(
        model="openai/gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def get_next_filename(output_dir, specified_id=None):
    """
    Get the next available filename
    
    Args:
        output_dir (str): Output directory
        specified_id (int, optional): Specified ID, if provided, use this ID
    
    Returns:
        tuple: (filename, full file path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if specified_id is not None:
        # Use specified ID
        next_num = specified_id
    else:
        # Find existing files, get max number
        existing_files = [f for f in os.listdir(output_dir) if f.startswith("scene_image_") and f.endswith(".png")]
        
        if not existing_files:
            next_num = 1
        else:
            # Extract number and find max value
            numbers = []
            for f in existing_files:
                try:
                    num = int(f.replace("scene_image_", "").replace(".png", ""))
                    numbers.append(num)
                except ValueError:
                    continue
            next_num = max(numbers) + 1 if numbers else 1
    
    filename = f"scene_image_{next_num}.png"
    filepath = os.path.join(output_dir, filename)
    return filename, filepath


def update_input_text_json(output_dir, filename, input_text):
    """
    Update input_text.json file, record filename and corresponding input text
    
    Args:
        output_dir (str): Output directory
        filename (str): Generated image filename
        input_text (str): Input text description
    """
    json_path = os.path.join(output_dir, "input_text.json")
    
    # Read existing data
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}
    
    # Add new record
    data[filename] = input_text
    
    # Write back to file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Updated input_text.json: {filename} -> {input_text[:50]}...")


def generate_scene_image_with_seedream(prompt_text, output_dir, doubao_api_key, input_text, specified_id=None):
    """
    Generate scene image using Seedream API
    
    Args:
        prompt_text (str): Text-to-image prompt
        output_dir (str): Output directory
        doubao_api_key (str): Doubao API key
        input_text (str): Original input text
        specified_id (int, optional): Specified ID
    
    Returns:
        str: Generated image path, return None if failed
    """
    try:
        # Initialize client
        client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=doubao_api_key or os.environ.get("DOUBAO_API_KEY"),
        )
        
        # Get next filename
        filename, output_path = get_next_filename(output_dir, specified_id)
        
        print(f"Generating image: {filename}")
        
        # Call API to generate image
        images_response = client.images.generate(
            model="doubao-seedream-4-0-250828",
            prompt=prompt_text,
            size="2K",
            response_format="url",
            watermark=False,
            sequential_image_generation="disabled"
        )
        
        # Get generated image URL
        image_url = images_response.data[0].url
        
        # Download image and save
        response = requests.get(image_url, timeout=120)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"✓ Image saved to: {output_path}")
            
            # Update JSON record
            update_input_text_json(output_dir, filename, input_text)
            
            return output_path
        else:
            print(f"✗Failed to download image, status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"✗ Failed to generate image: {e}")
        return None


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate scene images using Doubao Seedream API')
    parser.add_argument('--doubao_api_key', type=str, required=True,
                       help='Doubao API key')
    parser.add_argument('--id', type=int, default=None,
                       help='Specify generated image ID (optional, auto-increment if not specified)')
    parser.add_argument('--text', type=str, 
                       default="A hobby desk with some model cars and tools.",
                       help='Scene description text')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Scene description
    input_text = args.text
    
    # Generate scene prompt
    print("Generating scene prompt...")
    scene_prompt = generate_scene_prompt(input_text)
    print(f"Scene prompt: {scene_prompt}")
    
    # Configure paths
    PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(PIPELINE_DIR, "scene_image")
    
    # Generate scene image
    print("\nGenerating scene image...")
    result_path = generate_scene_image_with_seedream(
        scene_prompt, 
        output_directory, 
        args.doubao_api_key,
        input_text,
        args.id
    )