"""
Analyze scene images and extract common sense dimensions and coordinate axis information of objects.
"""

import json
import os
from modules.setup_openai_client import setup_openai_client


PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_object_list(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        object_names = data.get('object_names', [])

        return object_names
        
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return []
    except json.JSONDecodeError:
        print(f"JSON file format error: {json_file_path}")
        return []
    except Exception as e:
        print(f"Error occurred while reading file: {e}")
        return []

def analyze_scene_images(output_assets_dir, comfy_image_dir, api_key, proxy_url, base_url):
    """
    Analyze scene images and extract object dimensions and axial information
    """

    scene_image_path = os.path.join(comfy_image_dir, "scene_image.png")
    annotated_image_path = os.path.join(output_assets_dir, "image", "annotated_image_with_ids.jpg")
    segmentation_json_path = os.path.join(output_assets_dir, "image", "segmentation_results.json")
    
    if not os.path.exists(scene_image_path):
        print(f"Scene image does not exist: {scene_image_path}")
        return None
    if not os.path.exists(annotated_image_path):
        print(f"Annotated image does not exist: {annotated_image_path}")
        return None
    if not os.path.exists(segmentation_json_path):
        print(f"Segmentation results file does not exist: {segmentation_json_path}")
        return None
    
    object_names = get_object_list(segmentation_json_path)
    if not object_names:
        print("Unable to get object list")
        return None
    
    try:
        prompt = f"""You are an image analysis expert. Please analyze the size of each object in the picture and give the corresponding coordinate axis direction at the same time. The specific tasks are: 
1. **Object size estimation** 
    - Match each object with the given list of objects and Figure 2. Each bounding box in Figure 2 corresponds to a single object (Note: Don't be misled by the plural names in the object list. Each serial number corresponds to only one item.). 
    - Estimate the size of its 3D bounding box according to the display state of the object: [length, width, height] (unit: cm). 
    - The size must strictly correspond to the state of the diagram. For example, if there is an open laptop in the diagram, the height should be the height after opening rather than the thickness in the closed state. 
2. **Axis Definition & Placement State** 
    - Give the direction description and placement state of the three axes of the object [x, y, z] (i.e. [length, width, height]). For example, a book has two placement states: 'lay flat' and 'stand up'. In the lay flat state, x is the length of the book when it is laid flat, y is the width of the book when it is laid flat, and z is the thickness when it is laid flat. 
    - Do not include any other reference objects in the direction description. It is required that the coordinate system can be aligned only by the object itself. 
    - The xy plane of all objects should be parallel to the ground/table, and the z axis should be the height/thickness of the displayed position. 
3. **Realistic Size Reference** 
    Base size estimates on common real-world proportions (e.g., smartphone ≈ 15cm long, coffee cup ≈ 10cm tall, open laptop screen height ≈ 20cm). 
    
Output in JSON format (no additional explanations). Include the object name (from the object list), size, and axis description. 
**Output example** 
{{ "book_1": {{"size":[20,10,2], "axis": {{ "placement":"lay flat", "x":"length of the book when it is lying flat", "y":"width of the book when it is lying flat", "z":"thickness of the book when it is lying flat"}} }}, "object_2": ... }} 

The **object list** in the figure is: {object_names}"""
        

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
            model="google/gemini-2.5-pro",
            messages=input_messages
        )
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON
        try:
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                json_text = response_text[json_start:json_end].strip()
            elif '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            result_dict = json.loads(json_text)
            
            output_dir = os.path.join(output_assets_dir, "layout_json")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "size_axis.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
            print(f"Analysis results saved to: {output_path}")
            return result_dict, output_path
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Original response: {response_text}")
            return None
            
    except Exception as e:
        print(f"API call failed: {e}")
        return None

def extract_axis_size(input_path, output_axis_path, output_size_path):
    """
    Extract axis and size attributes from size_axis.json and save them as axis.json and size.json respectively
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        axis_data = {}
        for item_name, item_info in data.items():
            if 'axis' in item_info:
                axis_data[item_name] = item_info['axis']
            else:
                print(f"Warning: {item_name} has no axis attribute")

        size_data = {}
        for item_name, item_info in data.items():
            if 'size' in item_info:
                size_data[item_name] = item_info['size']
            else:
                print(f"Warning: {item_name} has no size attribute")

        os.makedirs(os.path.dirname(output_axis_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_size_path), exist_ok=True)

        # Save axis to new file
        with open(output_axis_path, 'w', encoding='utf-8') as f:
            json.dump(axis_data, f, indent=4, ensure_ascii=False)

        print(f"Extracted axis info for {len(axis_data)} items, saved to: {output_axis_path}")

        # Save size attribute to new file
        with open(output_size_path, 'w', encoding='utf-8') as f:
            json.dump(size_data, f, indent=4, ensure_ascii=False)

        print(f"Extracted size info for {len(size_data)} items, saved to: {output_size_path}")

        return axis_data, size_data

    except FileNotFoundError:
        print(f"File not found: {input_path}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None, None
    except Exception as e:
        print(f"Error during processing: {e}")
        return None, None

def vlm_size_axis_main(json_path, output_dir, output_assets_dir, comfy_image_dir, api_key, proxy_url, base_url):
    get_object_list(json_path)
    
    result, size_axis_path = analyze_scene_images(output_assets_dir, comfy_image_dir, api_key, proxy_url, base_url)
    if result:
        axis_path = os.path.join(output_dir, "axis.json")
        size_path = os.path.join(output_dir, "size.json")
        
        axis_data, size_data = extract_axis_size(size_axis_path, axis_path, size_path)
        if axis_data and size_data:
            print("axis and size data extraction completed!")
        else:
            print("axis and size data extraction failed!")
    else:
        print("Failed to analyze axis and size!")


if __name__ == "__main__":
    try:
        json_path = "output_scene/scene_1/output_assets/image/segmentation_results.json"
        output_dir = "output_scene/scene_1/output_assets/layout_json"
        output_assets_dir = "output_scene/scene_1/output_assets"
        comfy_image_dir = "output_scene/scene_1/comfy_image"
        api_key = "sk-xxxxxx"  # Replace with your OpenAI API Key
        proxy_url = "http://your-proxy-url:port"  # Replace with your proxy URL
        base_url = "https://api.openai.com/v1"  # Replace with your OpenAI API base URL

        vlm_size_axis_main(json_path, output_dir, output_assets_dir, comfy_image_dir, api_key, proxy_url, base_url)
    except Exception as e:
        print(f"VLM size axis main failed: {e}")




