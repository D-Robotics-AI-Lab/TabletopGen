"""
Use doubao to perform object detection on the top view and get bounding boxes.
"""
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
import json
import os
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import base64
from volcenginesdkarkruntime import Ark
import io
from modules.setup_openai_client import setup_openai_client

# Doubao API configuration
DEFAULT_MODEL = "doubao-seed-1-6-thinking-250715"
BBOX_TAG_START = "<bbox>"
BBOX_TAG_END = "</bbox>"

def is_multi_object(class_name, all_class_names):
    """
    Determine if the current object is of the same type (number of objects of the same type is greater than 1)
    """
    if sum(1 for name in all_class_names if name == class_name) > 1:
        return True
    this_words = set(class_name.lower().split())
    for other_name in all_class_names:
        if other_name == class_name:
            continue
        other_words = set(other_name.lower().split())
        if this_words & other_words:
            return True
    return False

def get_object_info_from_json(json_path):
    """Extract object information from segmentation result JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        object_names = []
        object_classes = []
        class_counts = defaultdict(int)
        
        for i, obj in enumerate(data.get('objects', [])):
            class_name = obj.get('class_name', '')
            object_name = data.get('object_names', [])[i] if i < len(data.get('object_names', [])) else f"{class_name}_{i}"
            
            object_names.append(object_name)
            object_classes.append(class_name)
            class_counts[class_name] += 1
        multi_instance_classes = {}
        for cls, count in class_counts.items():
            if is_multi_object(cls, object_classes):
                multi_instance_classes[cls] = count
        return object_names, object_classes, multi_instance_classes
    except Exception as e:
        print(f"Failed to read JSON file: {e}")
        return [], [], {}

def generate_object_descriptions(scene_image_path, annotated_image_path, multi_instance_objects, api_key, proxy_url, base_url):
    """Generate descriptions for multiple objects of the same type"""
    if not multi_instance_objects:
        return {}
    
    try:
        objects_to_describe = []
        for obj_name in multi_instance_objects:
            objects_to_describe.append(obj_name)
        
        objects_list = ", ".join(objects_to_describe)
        
        prompt = f"""I have two images:
1. Original scene image showing objects clearly
2. Annotated image with numbered labels on objects

Please analyze these {len(objects_to_describe)} objects: {objects_list}

For each object, use **the numbered annotations in image 2** to locate it (the object names may be incorrect, but the serial numbers in the image are definitely accurate), then refer to image 1 to describe its distinctive features including:
- Color and texture
- Shape and size relative to others
- Position in the scene
- Any unique characteristics that distinguish it from similar objects

Return a JSON dictionary where keys are the object names and values are detailed descriptions.
Example: {{"pen_1": "blue ballpoint pen on the left side", "pen_2": "red pen in the center", "pen_3": "black pen on the right"}}"""
        
        print(f"Generating descriptions for {len(objects_to_describe)} objects of the same type...")


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
        content = response.choices[0].message.content.strip()

        # Extract JSON
        if '```json' in content:
            json_start = content.find('```json') + 7
            json_end = content.find('```', json_start)
            content = content[json_start:json_end].strip()
        elif '{' in content:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            content = content[json_start:json_end]

        descriptions = json.loads(content)
        print(f"Generated object descriptions: {descriptions}")
        
        return descriptions
        
    except Exception as e:
        print(f"Failed to generate object descriptions: {e}")
        return {}

def validate_detection_results(detection_results):
    """Validate if detection result coordinates are valid"""
    for obj_name, box in detection_results.items():
        for coord in box:
            if coord < 1:
                print(f"Invalid coordinates detected: Bounding box for {obj_name} contains coordinate values less than 1: {box}")
                return False
    return True


def reconcile_detection_labels_by_id(detection_results: Dict[str, List[int]], input_object_names: List[str]) -> Tuple[Dict[str, List[int]], bool]:
    """
    Normalize detection result labels to full input object names based on IDs.
    """
    import re

    # Build "ID -> Standard Name" mapping from input names
    id_to_expected = {}
    for name in input_object_names:
        m = re.search(r'(\d+)$', name)
        if m:
            id_to_expected[m.group(1)] = name

    normalized: Dict[str, List[int]] = {}
    seen_ids = set()
    duplicate_found = False

    for detected_label, box in detection_results.items():
        m = re.search(r'(\d+)$', detected_label)
        if not m:
            # No ID, cannot align, keep original name
            normalized[detected_label] = box
            continue

        idx = m.group(1)
        if idx in seen_ids:
            duplicate_found = True
        else:
            seen_ids.add(idx)

        expected_name = id_to_expected.get(idx)
        if expected_name:
            # Overwrite with input standard name (aligned by ID)
            normalized[expected_name] = box
        else:
            # ID not found in input, keep original name
            normalized[detected_label] = box

    return normalized, (not duplicate_found)

def detect_objects_with_doubao(image_path, object_names, doubao_api_key, object_descriptions=None, max_retries=3):
    """Use Doubao for object detection, supporting precise identification of objects of the same type"""
    expected_count = len(object_names)
    
    for retry_count in range(max_retries):
        try:
            if retry_count > 0:
                print(f"Attempting detection {retry_count + 1}...")
            
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
            if not doubao_api_key:
                raise ValueError("请设置ARK_API_KEY环境变量")
            
            client = Ark(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=doubao_api_key)
            
            image = Image.open(image_path)

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            width, height = image.size
            
            # Build prompt - decide detection mode based on whether object descriptions exist
            if object_descriptions and len(object_descriptions) > 0:
                # Precise detection mode - objects of the same type need special handling
                object_list = []
                for obj_name in object_names:
                    if obj_name in object_descriptions:
                        object_list.append(f"{obj_name}: {object_descriptions[obj_name]}")
                    else:
                        object_list.append(obj_name)
                
                objects_text = "\n".join(object_list)
                
                prompt = f"""Detect ALL the following specific objects in the image:

{objects_text}

For objects with descriptions, use the description to identify the correct instance. Each object should be detected separately even if they are of the same type.
Return results in JSON format as an array with each object having 'label' (use the exact object name) and 'bbox' fields.
Example: [{{"category": "pen_1", "bbox": "<bbox>150 200 350 400</bbox>"}}, {{"category": "pen_2", "bbox": "<bbox>500 100 700 300</bbox>"}}]
"""
                if retry_count == 0:
                    print("Using Doubao for precise mode object detection...")
            else:
                object_list = ", ".join(set(object_names))
                prompt = f"""Detect all of the prominent items in the image, specifically looking for: {object_list}. 
Return results in JSON format as an array with each object having 'label' (use the exact object name) and 'bbox' fields.
Example: [{{"category": "pen", "bbox": "<bbox>150 200 350 400</bbox>"}}, {{"category": "notebook", "bbox": "<bbox>500 100 700 300</bbox>"}}]
"""
                if retry_count == 0:
                    print("Using Doubao for standard mode object detection...")
            
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }, {
                        "type": "text",
                        "text": prompt
                    }]
                }]
            )
            
            response_text = response.choices[0].message.content
            
            detection_results = {}
            
            try:
                # Parse JSON
                import re
                # Use greedy matching to ensure matching the complete JSON array (including nested brackets)
                json_pattern = r'\[.*\]'
                json_match = re.search(json_pattern, response_text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group()
                    bounding_boxes = json.loads(json_str)
                    
                    for bounding_box in bounding_boxes:
                        try:
                            bbox_str = bounding_box.get("bbox", "")
                            object_name = bounding_box.get("category") or bounding_box.get("label", "unknown")
                            
                            # Support multiple bbox formats: <bbox>x y w h</bbox> or <bbox x y w h>
                            bbox_str = bbox_str.replace(BBOX_TAG_START, "").replace(BBOX_TAG_END, "")
                            bbox_str = bbox_str.replace("<bbox", "").replace(">", "").strip()
                            
                            # Parse coordinates
                            coords = list(map(int, bbox_str.split()))
                            
                            if len(coords) == 4:
                                x_min, y_min, x_max, y_max = coords
                                
                                # Convert coordinates to absolute coordinates
                                abs_x_min = int(x_min * width / 1000)
                                abs_y_min = int(y_min * height / 1000)
                                abs_x_max = int(x_max * width / 1000)
                                abs_y_max = int(y_max * height / 1000)
                                
                                converted_box = [abs_x_min, abs_y_min, abs_x_max, abs_y_max]
                                detection_results[object_name] = converted_box
                        
                        except Exception as e:
                            print(f"Failed to parse bounding box coordinates: {e}")
                            continue
                
                # If JSON parsing fails, fallback to line-by-line parsing
                if not detection_results:
                    lines = response_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if 'bbox' in line.lower():
                            # Extract object name and bounding box
                            # Support multiple formats
                            try:
                                # Try to extract label
                                label_match = re.search(r'"label"\s*:\s*"([^"]+)"', line)
                                if label_match:
                                    object_name = label_match.group(1)
                                else:
                                    continue
                                
                                # Extract bbox coordinates
                                bbox_match = re.search(r'<?\s*bbox[>\s]+(\d+\s+\d+\s+\d+\s+\d+)', line)
                                if bbox_match:
                                    coords = list(map(int, bbox_match.group(1).split()))
                                    if len(coords) == 4:
                                        x_min, y_min, x_max, y_max = coords
                                        
                                        # Convert coordinates to absolute coordinates
                                        abs_x_min = int(x_min * width / 1000)
                                        abs_y_min = int(y_min * height / 1000)
                                        abs_x_max = int(x_max * width / 1000)
                                        abs_y_max = int(y_max * height / 1000)
                                        
                                        converted_box = [abs_x_min, abs_y_min, abs_x_max, abs_y_max]
                                        detection_results[object_name] = converted_box
                            
                            except Exception as e:
                                print(f"Failed to parse line: {e}, line content: {line}")
                                continue
            
            except Exception as e:
                print(f"Failed to parse response: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"Detected {len(detection_results)} objects this time")
            
            # Name normalization and duplicate ID check (align names by ID)
            if detection_results:
                detection_results, name_ok = reconcile_detection_labels_by_id(detection_results, object_names)
                if not name_ok:
                    # Duplicate ID found: treat as "Detection result coordinate validation failed, preparing to retry..."
                    print("Duplicate ID detected, preparing to retry...")
                    if retry_count == max_retries - 1:
                        print(f"Reached maximum retries ({max_retries}), returning current results")
                        return image, detection_results, (width, height)
                    continue

            # Check detected object count
            detected_count = len(detection_results)
            if detected_count != expected_count:
                print(f"Detected object count mismatch: Expected {expected_count}, actually detected {detected_count}")
                if retry_count == max_retries - 1:
                    print(f"Reached maximum retries ({max_retries}), returning current results")
                    return image, detection_results, (width, height)
                print(f"Preparing to retry...")
                continue

            # Validate detection results
            if detection_results and validate_detection_results(detection_results):
                print(f"Detected {len(detection_results)} objects, coordinate validation passed")
                return image, detection_results, (width, height)
            else:
                print(f"Detection result coordinate validation failed, preparing to retry...")
                if retry_count == max_retries - 1:
                    print(f"Reached maximum retries ({max_retries}), returning current results")
                    return image, detection_results, (width, height)
                continue
                
        except Exception as e:
            print(f"Object detection attempt {retry_count + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            if retry_count == max_retries - 1:
                print(f"Reached maximum retries ({max_retries}), detection failed")
                return None, {}, (0, 0)

    return None, {}, (0, 0)

def draw_bounding_boxes_with_ids(image, detection_results, output_path):
    """Draw bounding boxes with IDs on the image"""
    try:
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", size=20)
        except IOError:
            font = ImageFont.load_default()
        
        colors = ["red", "blue", "green", "purple", "orange", "yellow", "cyan", "magenta"]
        
        for i, (object_name, box) in enumerate(detection_results.items()):
            color = colors[i % len(colors)]
            
            # Draw rectangle box
            draw.rectangle(box, outline=color, width=3)
            
            # Write full object name at the top-left corner of the box
            text_position = (box[0] + 5, box[1] + 5)
            draw.text(text_position, object_name, fill="white", font=font, stroke_width=1, stroke_fill="black")
            
            # Print bounding box info
            width = box[2] - box[0]
            height = box[3] - box[1]
            print(f"{object_name}: Position({box[0]}, {box[1]}) Size({width} x {height})")
        
        image.save(output_path)
        print(f"✓ Detection results saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Failed to draw bounding boxes: {e}")
        return False

def topview_bbox_enhanced_main(output_assets_dir, comfy_image_dir, doubao_api_key, openai_api_key, proxy_url, base_url):
    """Enhanced main function"""
    try:
        scene_image_path = os.path.join(output_assets_dir, "image", "topview_scene.png")
        annotated_image_path = os.path.join(output_assets_dir, "image", "annotated_image_with_ids.jpg")
        
        # Segmentation result JSON path
        json_path = os.path.join(output_assets_dir, "image", "segmentation_results.json")
        
        # Output image path
        output_image_path = os.path.join(comfy_image_dir, "doubao_detection_result.png")
        
        # Output JSON path
        output_json_path = os.path.join(output_assets_dir, "layout_json", "doubao_detection_boxes.json")
        

        if not os.path.exists(scene_image_path):
            print(f"Scene image does not exist: {scene_image_path}")
            return False
        
        if not os.path.exists(json_path):
            print(f"Segmentation result JSON does not exist: {json_path}")
            return False
        
        # Extract object information
        object_names, object_classes, multi_instance_classes = get_object_info_from_json(json_path)
        
        if not object_names:
            print("Unable to extract object information")
            return False
        
        print(f"All objects: {object_names}")
        print(f"Multiple objects of same type: {multi_instance_classes}")
        
        # Generate descriptions for objects of same type
        object_descriptions = {}
        if multi_instance_classes and os.path.exists(annotated_image_path):
            
            multi_instance_objects = []
            for obj_name in object_names:
                for class_name in multi_instance_classes:
                    if class_name in obj_name or obj_name.startswith(class_name):
                        multi_instance_objects.append(obj_name)
                        break
            
            if multi_instance_objects:
                object_descriptions = generate_object_descriptions(
                    scene_image_path, annotated_image_path, multi_instance_objects, openai_api_key, proxy_url, base_url
                )
        elif multi_instance_classes:
            print(f"Warning: Multiple objects of same type detected but numbered annotation image not found: {annotated_image_path}")
        
        # Use Doubao for object detection
        scene_image, detection_results, image_size = detect_objects_with_doubao(
            scene_image_path, object_names, doubao_api_key, object_descriptions=object_descriptions
        )
        
        if scene_image is None or not detection_results:
            print("Object detection failed or no objects detected")
            return False
        
        print(f"Successfully detected {len(detection_results)} objects:")
        for obj_name, box in detection_results.items():
            print(f"  {obj_name}: {box}")
        
        # Save JSON results
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(detection_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ JSON results saved to: {output_json_path}")
        
        # Draw bounding boxes and save
        success = draw_bounding_boxes_with_ids(scene_image, detection_results, output_image_path)
        
        if success:
            print(f"Successfully identified and detected {len(detection_results)} objects")
            return True
        else:
            print("✗ Failed to draw bounding boxes!")
            return False
            
    except Exception as e:
        print(f"Main function execution failed: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    output_assets_dir = "output_scene/scene_1/output_assets"
    comfy_image_dir = "output_scene/scene_1/comfy_image"
    doubao_api_key = "your_doubao_api_key"  # Replace with your Doubao API Key
    openai_api_key = "sk-xxxxxx"  # Replace with your OpenAI API Key
    proxy_url = "http://your-proxy-url:port"  # Replace with your proxy URL
    base_url = "https://api.openai.com/v1"  # Replace with your OpenAI API base URL