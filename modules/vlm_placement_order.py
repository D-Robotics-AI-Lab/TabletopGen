"""
Analyze the placement order of objects in the scene using Gemini VLM.
"""

import json
import os
from google import genai
from google.genai import types
from modules.setup_openai_client import setup_openai_client


def get_object_names(segmentation_results_path):
    try:
        with open(segmentation_results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        object_names = data.get('object_names', [])
        return object_names
        
    except Exception as e:
        print(f"Failed to read segmentation results file: {e}")
        return []

def analyze_placement_order(scene_image_path, annotated_image_path, object_names, api_key, proxy_url, base_url):
    """Analyze object placement order using GPT"""
    try:

        prompt = f"""You are a layout designer. Given a list of objects, please output the placement order of each object in the image. The main object in the scene (such as a table) is ranked 1. Objects placed on top of it are determined by their stacking position. Generally, larger objects placed on top of the main object are placed first, with smaller objects placed later. Also, if object A is partially or completely placed above object B, A should be placed after B, and object B needs to be added to the output of object A.

Please analyze the image carefully and determine the logical placement order based on:
1. The main object (table) should be placed first (order 1)
2. Larger objects on the table should be placed before smaller objects
3. If one object is on top of another, the top object should have a higher order number
4. Consider the natural stacking and layering in the scene
5. **Analyze object overlap in scenes without bounding boxes**, and then map bounding box numbers to object names. Don't judge overlap directly from bounding boxes, as the bounding boxes will be larger and prone to false overlap.

Please output the placement order dictionary in JSON format, for example:
{{
    "pen_0": {{
        "order": 5,
        "on_top_of": "book_1"
    }},
    "book_1": {{
        "order": 2,
        "on_top_of": "table_2"
    }},
    "table_2": {{
        "order": 1,
        "on_top_of": null
    }}
}}

The object list is as follows:
{object_names}"""

        client = setup_openai_client(api_key, proxy_url, base_url)
        with open(scene_image_path, "rb") as f1:
            image1_data = f1.read()
        with open(annotated_image_path, "rb") as f2:
            image2_data = f2.read()
        import base64
        image1_base64 = base64.b64encode(image1_data).decode('utf-8')
        image2_base64 = base64.b64encode(image2_data).decode('utf-8')
        input_messages = [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{image1_base64}" } },
                    { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{image2_base64}" } }

                ]
            }
        ]
        response = client.chat.completions.create(
            model="openai/gpt-5",
            messages=input_messages
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            if '```json' in content:
                json_start = content.find('```json') + 7
                json_end = content.find('```', json_start)
                json_text = content[json_start:json_end].strip()
            elif '{' in content:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                json_text = content[json_start:json_end]
            else:
                json_text = content
            
            placement_order = json.loads(json_text)
            
            print(f"Parsed placement order: {placement_order}")
            return placement_order
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Original response: {content}")
            return None
            
    except Exception as e:
        print(f"Gemini analysis of placement order failed: {e}")
        return None

def validate_placement_order(placement_order, object_names):
    """Validate the rationality of the placement order"""
    try:
        # Check if all objects have an order
        missing_objects = []
        for obj_name in object_names:
            if obj_name not in placement_order:
                missing_objects.append(obj_name)
        
        if missing_objects:
            print(f"Warning: The following objects are missing placement order: {missing_objects}")
        
        # Check if order values are reasonable
        orders = []
        for obj_name, obj_data in placement_order.items():
            if isinstance(obj_data, dict):
                order = obj_data.get("order")
            else:
                order = obj_data
            orders.append(order)
        
        if not all(isinstance(order, int) for order in orders):
            print("Warning: Placement order contains non-integer values")
            return False
        
        if len(set(orders)) != len(orders):
            print("Warning: Duplicate values in placement order")
            return False
        
        print(f"Placement order validation passed, total {len(placement_order)} objects")
        return True
        
    except Exception as e:
        print(f"Error validating placement order: {e}")
        return False

def placement_order_main(output_assets_dir, comfy_image_dir, api_key, proxy_url, base_url):
    """Main function"""
    try:
        scene_image_path = os.path.join(comfy_image_dir, "scene_image.png")
        annotated_image_path = os.path.join(output_assets_dir, "image", "annotated_image_with_ids.jpg")
        segmentation_results_path = os.path.join(output_assets_dir, "image", "segmentation_results.json")

        output_dir = os.path.join(output_assets_dir, "layout_json")
        output_path = os.path.join(output_dir, "placement_order.json")
        
        if not os.path.exists(scene_image_path):
            print(f"Scene image does not exist: {scene_image_path}")
            return False
        
        if not os.path.exists(annotated_image_path):
            print(f"Annotated image does not exist: {annotated_image_path}")
            return False
        
        if not os.path.exists(segmentation_results_path):
            print(f"Segmentation results file does not exist: {segmentation_results_path}")
            return False
        
        print("=== Start analyzing object placement order ===")

        object_names = get_object_names(segmentation_results_path)
        if not object_names:
            print("Unable to extract object name list")
            return False

        placement_order = analyze_placement_order(scene_image_path, annotated_image_path, object_names, api_key, proxy_url, base_url)
        
        if not placement_order:
            print("Placement order analysis failed")
            return False

        if not validate_placement_order(placement_order, object_names):
            print("Placement order validation failed")

        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(placement_order, f, indent=4, ensure_ascii=False)
        
        print(f"✓ Placement order saved to: {output_path}")
        
        # 6. Output result summary
        print("\n=== Placement Order Results ===")
        # Sort by order and display
        sorted_objects = []
        for obj_name, obj_data in placement_order.items():
            if isinstance(obj_data, dict):
                order = obj_data.get("order", 999)
                on_top_of = obj_data.get("on_top_of", None)
            else:
                order = obj_data
                on_top_of = None
            sorted_objects.append((order, obj_name, on_top_of))
        
        sorted_objects.sort(key=lambda x: x[0])
        
        for order, obj_name, on_top_of in sorted_objects:
            if on_top_of:
                print(f"Order {order}: {obj_name} (Placed on: {on_top_of})")
            else:
                print(f"Order {order}: {obj_name}")
        
        return True
        
    except Exception as e:
        print(f"Main function execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    output_assets_dir = "output_scene/scene_1/output_assets"
    comfy_image_dir = "output_scene/scene_1/comfy_image"
    openai_api_key = "sk-xxxxxx"  # Replace with your OpenAI API Key
    proxy_url = "http://your-proxy-url:port"  # Replace with your proxy URL
    base_url = "https://api.openai.com/v1"  # Replace with your OpenAI API base URL


    placement_order_main(output_assets_dir, comfy_image_dir, openai_api_key, proxy_url, base_url)