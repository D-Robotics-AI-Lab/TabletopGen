"""
Detection and Segmentation of Objects in an Image using Grounded-SAM-2 and GPT
"""
import os
import cv2
import json
import torch
import httpx
import openai
import numpy as np
import supervision as sv
from PIL import Image
import pycocotools.mask as mask_util
from pathlib import Path
import base64
import logging
from torchvision.ops import box_convert
import sys
from modules.setup_openai_client import setup_openai_client


MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(MODULES_DIR).parent
# Add Grounded-SAM-2 directory (now expected at repo root) to system path
GSAM_DIR = PROJECT_ROOT / "Grounded-SAM-2"
sys.path.append(str(GSAM_DIR))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import argparse
import math


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("get_asset_image")

def detect_objects_with_gpt(client, image_path):
    """
    Use GPT to identify object categories in the image and analyze detailed occlusion relationships simultaneously.
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Send request to GPT to analyze objects and detailed occlusion relationships in the image
        logger.info("Using GPT-4.1 to recognize objects and detailed occlusion relationships in the image...")
        messages=[
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text", 
                        "text": """
You are an image analysis expert. Please complete the following tasks based on the provided image:
1. Identify all object categories present in the **tabletop**, including the table itself. Return all category names in English, each followed by a period.
2. Each category refers to a major category, objects that are roughly similar belong to the same category. For example, "pen" includes "pencil", "cup" includes "mug", and "book" includes "notebook", etc.
- **Items that are integrated with the desktop do not belong to a separate category, such as drawers and sinks that come with the table.**
3. **This is a mandatory merging rule and must be strictly enforced**: When identifying object categories, you **must** merge objects with similar appearances or the same essence into **one** unified and broader main category. It is **absolutely prohibited** to list multiple similar sub - categories separately in the final result. For example:
- If you see both 'pen' and 'pencil' at the same time, only one category can be output in the end, such as 'pen'. It is **prohibited** to output both 'pen' and 'pencil'.
- Similarly, 'cup' and 'mug' **must** be merged into 'cup'.
- Similarly, 'book' , 'notebook' and 'journal' **must** be merged into 'notebook'. **But the sticky notes on the book are classified separately as 'sticky note'**.
- Dissimilar-looking items do not need to be merged. For example, "guitar" and "flute" are two different categories and do not need to be merged into "musical instrument".
4. Identify the main object of the scene. This is usually a foundational object that supports or contains other objects, such as a table or bookshelf.
5. **Analyze occlusion relationships with the strictest standard**: An object is only considered occluded if another object is physically **in front of it and covering a part of it** from the camera's perspective. Objects that are merely **next to each other do not count** as occlusion. You **must** meticulously inspect all object edges and contact points, as even the **most subtle overlap** (e.g., the tip of one object resting on the edge of another) **must be identified** as an occlusion. Describe in detail which objects are occluded, by which objects, and the location/direction of the occlusion.
6. Output the result in JSON format only, with no additional explanation, strictly as a JSON object.

Example return format:
{
  "objects": "table. cup. book. laptop. plant.",
  "main_object": "table",
  "occlusion_info": [
    {
      "occluded_object": "book",
      "occluded_by": ["laptop", "pen"],
      "description": "The laptop occludes the upper right part of the book, and the pen occludes the lower left part of the book."
    },
    {
      "occluded_object": "table",
      "occluded_by": ["cup", "book", "laptop"],
      "description": "The cup occludes a small part on the right side of the table, and the book and laptop occlude the central area of the table."
    }
  ]
}

Notes:
1. 'occlusion_info' only includes objects that are occluded. Objects that are not occluded do not need to be listed.
2. 'occluded_by' should be an array containing all objects that occlude the target object.
3. 'description' should describe all occlusion relationships.
4. Different instances of the same category being occluded should be listed as separate entries.
5. 'main_object' must return the main object in the scene, such as table or bookshelf. If there are multiple, return only the one with the largest coverage area.
"""
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ]

        response = client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=messages,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        print(content)
        if content is None or content.strip() == "":
            logger.error("GPT returned empty content, aborting.")
            raise RuntimeError("Failed to get response from GPT.")

        try:
            result = json.loads(content)
            objects = result.get("objects", "").strip()
            occlusion_info = result.get("occlusion_info", [])
            main_object = result.get("main_object", "unknown")
        except json.JSONDecodeError:
            logger.error("GPT returned content that could not be parsed as JSON, aborting.")
            raise RuntimeError("Failed to parse GPT response as JSON.")
        
        logger.info(f"Detected objects: {objects}")
        logger.info(f"Main object: {main_object}")
        if occlusion_info:
            # for info in occlusion_info:
            #     logger.info(f"Occlusion: {info.get('occluded_object')} occluded by {info.get('occluded_by')} - {info.get('description')}")
            logger.info(f"Detected {len(occlusion_info)} occlusion relationships.")
        else:
            logger.info("No occlusion relationships detected")
        
        # Extract list of occluded object categories
        occluded_categories = [info.get('occluded_object') for info in occlusion_info]
        
        return objects, occluded_categories, occlusion_info, main_object
    except Exception as e:
        logger.error(f"Failed to recognize objects with GPT: {str(e)}")
        raise

def create_annotated_image_with_ids(image, detections, class_names, ids):
    """
    Create annotated image with object ID identifiers.
    """
    annotated_image = image.copy()
    
    # Define different colors (BGR format)
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
        (255, 192, 203), # Pink
        (165, 42, 42),  # Brown
    ]
    
    label_info = []
    
    used_positions = []
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    text_thickness = 2
    
    # First pass: Draw bounding boxes only
    for i, box in enumerate(detections.xyxy):
        x1, y1, x2, y2 = map(int, box)
        
        color = colors[i % len(colors)]
        
        thickness = 3
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
    
    # Second pass: Calculate label positions and store information
    for i, box in enumerate(detections.xyxy):
        x1, y1, x2, y2 = map(int, box)
        
        color = colors[i % len(colors)]
        
        label_text = str(ids[i])

        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, text_thickness)

        potential_positions = [
            (x1, y1 - 10),                    # Above top-left corner
            (x2 - text_width, y1 - 10),       # Above top-right corner
            (x1, y2 + text_height + 10),      # Below bottom-left corner
            (x2 - text_width, y2 + text_height + 10),  # Below bottom-right corner
            (x1 - text_width - 10, y1),       # Left side
            (x2 + 10, y1),                    # Right side
            (x1 - text_width - 10, y2 - text_height),  # Bottom-left side
            (x2 + 10, y2 - text_height),      # Bottom-right side
        ]

        label_x, label_y = None, None
        for pos_x, pos_y in potential_positions:
            if (pos_x >= 0 and pos_y >= text_height and 
                pos_x + text_width <= annotated_image.shape[1] and 
                pos_y <= annotated_image.shape[0]):
                
                current_rect = (pos_x - 5, pos_y - text_height - 5, 
                              pos_x + text_width + 5, pos_y + baseline + 5)
                
                overlap = False
                for used_rect in used_positions:
                    if rectangles_overlap(current_rect, used_rect):
                        overlap = True
                        break
                
                if not overlap:
                    label_x, label_y = pos_x, pos_y
                    used_positions.append(current_rect)
                    break
        
        # If no non-overlapping position is found, use default position (inside bounding box)
        if label_x is None or label_y is None:
            label_x = x1 + 5
            label_y = y1 + text_height + 5
            current_rect = (label_x - 5, label_y - text_height - 5, 
                          label_x + text_width + 5, label_y + baseline + 5)
            used_positions.append(current_rect)

        brightness = sum(color) / 3
        text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
        
        label_info.append({
            'text': label_text,
            'x': label_x,
            'y': label_y,
            'width': text_width,
            'height': text_height,
            'baseline': baseline,
            'bg_color': color,
            'text_color': text_color
        })
    
    # Third pass: Draw all labels (ensure they are on the top layer)
    for info in label_info:
        cv2.rectangle(
            annotated_image,
            (info['x'] - 5, info['y'] - info['height'] - 5),
            (info['x'] + info['width'] + 5, info['y'] + info['baseline'] + 5),
            info['bg_color'],
            -1
        )

        cv2.putText(
            annotated_image,
            info['text'],
            (info['x'], info['y']),
            font,
            font_scale,
            info['text_color'],
            text_thickness
        )
    
    return annotated_image

def rectangles_overlap(rect1, rect2):
    """
    Check if two rectangles overlap.
    """
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2
    
    # Check if they do not overlap (disjoint); if not disjoint, they overlap
    return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)

def segment_image(image_path, text_prompt, output_dir, occlusion_info=None, box_threshold=0.3, text_threshold=0.2, confidence_threshold = 0.4):
    """Segment image using Grounded-SAM-2"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Grounded-SAM-2 configuration
        SAM_BASE_DIR = str(GSAM_DIR)
        
        current_dir = os.getcwd()
        
        output_dir_absolute = os.path.abspath(output_dir)
        
        seg_dir = os.path.join(output_dir_absolute, "seg_dir")
        os.makedirs(seg_dir, exist_ok=True)
        
        os.chdir(SAM_BASE_DIR)
        
        SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
        SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
        GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
        
        BOX_THRESHOLD = box_threshold
        TEXT_THRESHOLD = text_threshold
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {DEVICE}")
        logger.info(f"Loading SAM2 model: {SAM2_MODEL_CONFIG}")
        
        # Build SAM2 image predictor
        sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        logger.info("Loading Grounding DINO model")
        grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=DEVICE
        )
    
        full_image_path = os.path.join(current_dir, image_path)
        logger.info(f"Loading image: {full_image_path}")
        
        image_source, image = load_image(full_image_path)
        original_image = cv2.imread(full_image_path)
        sam2_predictor.set_image(image_source)
            # === New: Save original scene image to seg_dir ===
        scene_jpg_path = os.path.join(seg_dir, "scene.jpg")
        scene_png_path = os.path.join(seg_dir, "scene.png")
        
        cv2.imwrite(scene_jpg_path, original_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(scene_png_path, original_image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        
        # logger.info(f"Saved scene image to seg_dir: scene.jpg and scene.png")
        
        processed_prompt = text_prompt.replace('.', '.')
        
        try:
            with torch.cuda.amp.autocast(enabled=False):
                # Detect using Grounding DINO
                boxes, confidences, labels = predict(
                    model=grounding_model,
                    image=image,
                    caption=processed_prompt,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                )
        except Exception as e:
            error_msg = str(e)
            if "BFloat16" in error_msg or "ms_deform_attn" in error_msg or "_C" in error_msg:
                logger.warning(f"Failed to run in CUDA mode ({error_msg}), trying pure CPU mode...")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    del grounding_model
                    del sam2_model
                    del sam2_predictor
                    torch.cuda.synchronize()
                
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                
                # Reload model to CPU
                sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device="cpu")
                sam2_predictor = SAM2ImagePredictor(sam2_model)
                sam2_predictor.set_image(image_source)
                
                grounding_model = load_model(
                    model_config_path=GROUNDING_DINO_CONFIG,
                    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
                    device="cpu"
                )
                
                # Try detection again
                boxes, confidences, labels = predict(
                    model=grounding_model,
                    image=image,
                    caption=processed_prompt,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                )
                
                DEVICE = "cpu"
            else:
                raise
        
        # Add confidence filtering
        valid_indices = confidences >= confidence_threshold
        
        if valid_indices.sum() == 0:
            logger.warning(f"No objects with confidence >= {confidence_threshold} detected! Please try adjusting the prompt or lowering the threshold.")
            return {
                "image_path": image_path,
                "objects": [],
                "img_width": image_source.shape[1],
                "img_height": image_source.shape[0]
            }
        
        # Filter detection results
        boxes = boxes[valid_indices]
        confidences = confidences[valid_indices]
        labels = [labels[i] for i in range(len(labels)) if valid_indices[i]]
        
        # logger.info(f"After confidence filtering (>= {confidence_threshold}): {len(boxes)} objects remaining")
        
        if len(boxes) == 0:
            logger.warning("No objects detected! Please try adjusting the prompt or lowering the threshold.")
            return {
                "image_path": image_path,
                "objects": [],
                "img_width": image_source.shape[1],
                "img_height": image_source.shape[0]
            }
        
        logger.info(f"Detected {len(boxes)} objects")
        
        # Process bounding boxes
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        if DEVICE == "cuda":
            torch.autocast(device_type="cuda", dtype=torch.float32).__enter__()
            
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        logger.info("Generating segmentation masks...")
        # Predict masks
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        # Process mask shape
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        # Convert results to list format
        confidences = confidences.numpy().tolist()
        class_names = labels
        class_ids = np.array(list(range(len(class_names))))
        
        # Create labels
        label_texts = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]
        
        # Visualize detection results
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )
        
        # Create annotated image with ID identifiers
        annotated_frame = create_annotated_image_with_ids(
            original_image.copy(), 
            detections, 
            class_names, 
            list(range(len(class_names)))
        )
        
        # Save annotation results
        annotation_path = os.path.join(output_dir_absolute, "annotated_image_with_ids.jpg")
        cv2.imwrite(annotation_path, annotated_frame)
        # logger.info(f"Saved annotated image with IDs: {annotation_path}")
        
        # Also save standard annotated image
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        standard_annotated_frame = box_annotator.annotate(scene=original_image.copy(), detections=detections)
        standard_annotated_frame = label_annotator.annotate(scene=standard_annotated_frame, detections=detections, labels=label_texts)
        standard_annotated_frame = mask_annotator.annotate(scene=standard_annotated_frame, detections=detections)
        
        standard_annotation_path = os.path.join(output_dir_absolute, "annotated_image.jpg")
        cv2.imwrite(standard_annotation_path, standard_annotated_frame)
        
        # Extract images of each object
        extracted_objects = []
        
        obj_class_list = []
        for class_name in class_names:
            formatted_class = class_name.replace('. ', ',')
            if formatted_class not in obj_class_list:
                obj_class_list.append(formatted_class)
        
        # Create obj_class_bbox dictionary
        obj_class_bbox = {
            "obj_class": obj_class_list
        }

        # Create mapping from object ID to occlusion info
        occlusion_map = {}
        if occlusion_info:
            # Create mapping from category to instance IDs
            class_to_ids = {}
            for i, class_name in enumerate(class_names):
                if class_name not in class_to_ids:
                    class_to_ids[class_name] = []
                class_to_ids[class_name].append(i)
            
            # Find corresponding object ID for each occlusion relationship
            for info in occlusion_info:
                occluded_object = info.get('occluded_object')
                if occluded_object in class_to_ids:
                    for obj_id in class_to_ids[occluded_object]:
                        occlusion_map[obj_id] = info
        
        
        logger.info("Extracting each object image...")
        for i, (box, mask, class_name) in enumerate(zip(input_boxes, masks, class_names)):
            try:
                transparent_mask = np.zeros((h, w, 4), dtype=np.uint8)
                
                transparent_mask[:, :, :3] = original_image
                
                transparent_mask[:, :, 3] = (mask * 255).astype(np.uint8)
                
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                cropped_object = transparent_mask[y1:y2, x1:x2]
                
                output_path = os.path.join(output_dir_absolute, f"{class_name}_{i}.png")
                cv2.imwrite(output_path, cropped_object)
                
                seg_object_path = os.path.join(seg_dir, f"{class_name}_{i}.png")
                cv2.imwrite(seg_object_path, cropped_object)
                
                scene_mask = np.zeros((h, w), dtype=np.uint8)
                mask_bool = mask.astype(bool)
                scene_mask[mask_bool] = 255
                
                mask_path = os.path.join(seg_dir, f"{class_name}_{i}_mask.png")
                cv2.imwrite(mask_path, scene_mask)
                
                x1, y1, x2, y2 = map(int, box)
                obj_key = f"{class_name}_{i}"
                obj_class_bbox[obj_key] = [x1, y1, x2, y2]
                
                # logger.info(f"Saved to seg_dir: {class_name}_{i}.png and {class_name}_{i}_mask.png")
                
                is_occluded = i in occlusion_map
                
                image_output_path = None
                mask_output_path = None
                
                if is_occluded:
                    margin = 20
                    x1_expanded = max(0, x1 - margin)
                    y1_expanded = max(0, y1 - margin)
                    x2_expanded = min(w, x2 + margin)
                    y2_expanded = min(h, y2 + margin)
                    
                    cropped_image = original_image[y1_expanded:y2_expanded, x1_expanded:x2_expanded].copy()
                    
                    cropped_mask = np.ones((y2_expanded-y1_expanded, x2_expanded-x1_expanded, 4), dtype=np.uint8) * 255
                    
                    center_width = int((x2_expanded - x1_expanded) * 0.3)
                    center_height = int((y2_expanded - y1_expanded) * 0.3)
                    center_x = (x2_expanded - x1_expanded) // 2
                    center_y = (y2_expanded - y1_expanded) // 2
                    
                    start_x = max(0, center_x - center_width // 2)
                    end_x = min(x2_expanded - x1_expanded, center_x + center_width // 2)
                    start_y = max(0, center_y - center_height // 2)
                    end_y = min(y2_expanded - y1_expanded, center_y + center_height // 2)
                    
                    cropped_mask[start_y:end_y, start_x:end_x, 3] = 0
                    
                    image_output_path = os.path.join(output_dir_absolute, f"image_{class_name}_{i}.png")
                    mask_output_path = os.path.join(output_dir_absolute, f"mask_{class_name}_{i}.png")
                    
                    cv2.imwrite(image_output_path, cropped_image)
                    cv2.imwrite(mask_output_path, cropped_mask)
                    
                    # logger.info(f"Saved occluded object image: {image_output_path}")
                    # logger.info(f"Saved occlusion mask: {mask_output_path}")
                
                obj_info = {
                    "class_name": class_name,
                    "confidence": confidences[i],
                    "image_path": output_path,
                    "bbox": box.tolist(),
                    "size": {"width": x2-x1, "height": y2-y1},
                    "is_occluded": is_occluded
                }
                
                if is_occluded:
                    occlusion_data = occlusion_map[i]
                    obj_info["occluded_by"] = occlusion_data.get("occluded_by")
                    obj_info["occlusion_description"] = occlusion_data.get("description")
                    obj_info["inpainting_image_path"] = image_output_path
                    obj_info["inpainting_mask_path"] = mask_output_path
                
                extracted_objects.append(obj_info)
                
                # logger.info(f"Extracted object: {class_name} (ID: {i})" + (" [occluded]" if is_occluded else ""))
            except Exception as e:
                logger.error(f"Failed to extract object {class_name}_{i}: {str(e)}")
        
        bbox_json_path = os.path.join(seg_dir, "obj_class_bbox.json")
        with open(bbox_json_path, "w") as f:
            json.dump(obj_class_bbox, f, indent=4)
        
        # logger.info(f"Saved obj_class_bbox.json to: {bbox_json_path}")
        logger.info(f"Saved segmentation outputs to seg_dir: {seg_dir}")
        
        os.chdir(current_dir)
        
        if torch.cuda.is_available():
            del grounding_model
            del sam2_model
            del sam2_predictor
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Generate object_names list
        object_names = [f"{class_name}_{i}" for i, class_name in enumerate(class_names)]
        
        # Save segmentation results as JSON
        results = {
            "image_path": image_path,
            "object_names": object_names,
            "objects": extracted_objects,
            "img_width": w,
            "img_height": h
        }
        
        results_path = os.path.join(output_dir, "segmentation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Saved segmentation results: {results_path}")
        return results, annotation_path
    except Exception as e:
        if 'current_dir' in locals():
            os.chdir(current_dir)
        
        if torch.cuda.is_available():
            try:
                if 'grounding_model' in locals():
                    del grounding_model
                if 'sam2_model' in locals():
                    del sam2_model
                if 'sam2_predictor' in locals():
                    del sam2_predictor
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass
        
        logger.error(f"Failed to segment image: {str(e)}")
        raise

def analyze_occluded_objects(client, image_path, extracted_objects, occluded_categories, occlusion_info, output_dir_absolute, current_dir):
    """
    Analyze which specific object IDs are occluded, especially handling cases with multiple objects of the same category.
    """
    try:
        # If no occluded categories, return directly
        if not occluded_categories:
            logger.info("No occluded object categories detected, skipping detailed analysis")
            for obj in extracted_objects:
                obj["is_occluded"] = False
            return extracted_objects, []
        
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        category_objects = {}
        for i, obj in enumerate(extracted_objects):
            category = obj["class_name"]
            if category not in category_objects:
                category_objects[category] = []
            category_objects[category].append(i)
        
        # Determine which categories need further analysis (cases with multiple objects of the same category)
        categories_to_analyze = []
        for category in occluded_categories:
            if category in category_objects and len(category_objects[category]) > 1:
                categories_to_analyze.append(category)
                # logger.info(f"{category} has multiple objects (IDs: {category_objects[category]}), further analysis required")
        
        if not categories_to_analyze:
            logger.info("No multi-instance categories require further analysis")
            
            occlusion_map = {}
            for info in occlusion_info:
                occluded_object = info.get('occluded_object')
                if occluded_object in category_objects:
                    for obj_id in category_objects[occluded_object]:
                        occlusion_map[obj_id] = info
            
            for i, obj in enumerate(extracted_objects):
                if i in occlusion_map:
                    obj["is_occluded"] = True
                    info = occlusion_map[i]
                    obj["occluded_by"] = info.get("occluded_by")
                    obj["occlusion_description"] = info.get("description")
                else:
                    obj["is_occluded"] = False
            
            return extracted_objects, occlusion_info
        
        # Prepare object images for further analysis
        object_images = {}
        for category in categories_to_analyze:
            for obj_id in category_objects[category]:
                obj = extracted_objects[obj_id]
                obj_image_path = os.path.join(output_dir_absolute, f"{obj['class_name']}_{obj_id}.png")
                if os.path.exists(obj_image_path):
                    with open(obj_image_path, "rb") as img_file:
                        img_data = img_file.read()
                    object_images[obj_id] = base64.b64encode(img_data).decode('utf-8')

        object_info_text = "\n".join([
            f"Category '{category}' object IDs: {category_objects[category]}" 
            for category in categories_to_analyze
        ])
        
        content = [
            {"type": "text", "text": f"""Please analyze the original image and determine which specific object IDs are occluded.

The following categories may have occluded objects in the image, and each category has multiple instances:
{', '.join(categories_to_analyze)}

Object ID info:
{object_info_text}

Carefully check the segmentation image of each object and determine which specific IDs are occluded.

Return a JSON result in the following format:
```json
{{
  "detailed_occlusion_info": [
    {{
      "occluded_object_id": object_id,
      "occluded_object_class": "object_category",
      "occluded_by": ["occluder1", "occluder2"],
      "description": "Detailed description of the occlusion, including all occlusion relationships"
    }},
    ... 
  ]
}}
```

Notes:
1. Only consider an object occluded if part of it is blocked by another object.
2. If the object is fully visible, it is not considered occluded.
3. 'occluded_by' should be an array containing all other objects that occlude this object.
4. Return only JSON, no explanation.
"""}, 
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
        
        for obj_id, img_base64 in object_images.items():
            obj = extracted_objects[obj_id]
            content.append(
                {"type": "text", "text": f"Segmentation image of object ID {obj_id} ({obj['class_name']}):"}
            )
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            )
        
        logger.info("Using GPT-4.1 to analyze which specific object IDs are occluded...")
        response = client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        detailed_info = result.get("detailed_occlusion_info", [])
        
        occluded_ids = set()
        for info in detailed_info:
            obj_id = info.get("occluded_object_id")
            occluded_by = info.get("occluded_by", [])
            if obj_id is not None and 0 <= obj_id < len(extracted_objects) and occluded_by:
                occluded_ids.add(obj_id)
                # logger.info(f"GPT detected object ID {obj_id} ({info.get('occluded_object_class')}) is occluded by {occluded_by}: {info.get('description', '')}")
        
        # Update object information
        for i, obj in enumerate(extracted_objects):
            category = obj["class_name"]
            if category in categories_to_analyze:
                if i in occluded_ids:
                    obj["is_occluded"] = True
                    for info in detailed_info:
                        if info.get("occluded_object_id") == i:
                            obj["occluded_by"] = info.get("occluded_by")
                            obj["occlusion_description"] = info.get("description")
                            break
                else:
                    obj["is_occluded"] = False
            else:
                # For categories not requiring further analysis, use initial analysis results
                if category in occluded_categories:
                    if len(category_objects[category]) == 1:
                        obj["is_occluded"] = True
                        for info in occlusion_info:
                            if info.get("occluded_object") == category:
                                obj["occluded_by"] = info.get("occluded_by")
                                obj["occlusion_description"] = info.get("description")
                                break
                else:
                    obj["is_occluded"] = False
        
        # Merge results from initial and detailed analysis
        merged_occlusion_info = []

        for info in detailed_info:
            obj_id = info.get("occluded_object_id")
            if obj_id is not None and 0 <= obj_id < len(extracted_objects):
                merged_info = {
                    "occluded_object": info.get("occluded_object_class"),
                    "occluded_by": info.get("occluded_by"),
                    "description": info.get("description")
                }
                merged_occlusion_info.append(merged_info)
        
        for info in occlusion_info:
            category = info.get("occluded_object")
            if category not in categories_to_analyze:
                merged_occlusion_info.append(info)
        
        return extracted_objects, merged_occlusion_info
    except Exception as e:
        logger.error(f"Failed to analyze occluded objects: {str(e)}")
        # On error, use initial analysis results
        return extracted_objects, occlusion_info

def get_asset_image_main(image, api_key, proxy_url, base_url, output_assets_dir, prompt=None, main_object=None, box_threshold=0.3, text_threshold=0.2,confidence_threshold=0.3):
    """Main processing function"""
    try:
        image_path = os.path.abspath(image)

        if not os.path.exists(image_path):
            logger.error(f"Error: image file {image_path} does not exist")
            return

        output_dir = os.path.join(output_assets_dir, "image")
        os.makedirs(output_dir, exist_ok=True)
        
        seg_dir = os.path.join(output_dir, "seg_dir")
        os.makedirs(seg_dir, exist_ok=True)
        
        logger.info(f"Processing started for image: {image_path}")
        logger.info(f"Output directory: {output_dir}")
        
        # Detect objects and occlusion relationships
        if prompt:
            text_prompt = prompt
            occlusion_info = []
            occluded_categories = []
            main_object = main_object if main_object else "unknown"
            logger.info(f"Using user-provided prompt: {text_prompt}")
            logger.info(f"Main object of the scene: {main_object}")
        else:
            logger.info("Using GPT-4.1 to detect objects and occlusion relationships in the image...")
            client = setup_openai_client(api_key, proxy_url, base_url)
            text_prompt, occluded_categories, occlusion_info, main_object = detect_objects_with_gpt(client, image_path)
        
        # Segment image - first call does not include occlusion info
        logger.info(f"Segmenting image with prompt: {text_prompt}")
        logger.info(f"Box threshold: {box_threshold}, Text threshold: {text_threshold}")
        segment_result = segment_image(
            image_path, 
            text_prompt, 
            output_dir,
            occlusion_info=None,
            box_threshold=box_threshold, 
            text_threshold=text_threshold,
            confidence_threshold=confidence_threshold
        )
        
        if isinstance(segment_result, tuple) and len(segment_result) >= 1:
            results = segment_result[0]
            annotation_path = segment_result[1] if len(segment_result) > 1 else None
        else:
            results = segment_result
            annotation_path = None
        
        if occluded_categories:
            logger.info("Analyzing occluded objects...")
            client = setup_openai_client(api_key, proxy_url, base_url)

            current_dir = os.getcwd()
            output_dir_absolute = os.path.abspath(output_dir)
            
            # For multiple objects of the same category, use GPT for secondary analysis
            updated_objects, detailed_occlusion_info = analyze_occluded_objects(
                client,
                image_path,
                results["objects"],
                occluded_categories,
                occlusion_info,
                output_dir_absolute,
                current_dir
            )
            
            # Update results
            results["objects"] = updated_objects
            results["occlusion_info"] = detailed_occlusion_info
            results["main_object"] = main_object 
            
            # Call process_occlusion_images again to handle occlusion info, generating mask and image maps
            logger.info("Generating masks and images for occluded objects...")
            
            process_occlusion_images(
                image_path,
                updated_objects,
                detailed_occlusion_info,
                output_dir_absolute,
                current_dir,
                main_object
            )
            
            # Save updated results
            results_path = os.path.join(output_dir, "segmentation_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            
            logger.info(f"Updated segmentation results saved: {results_path}")
        else:
            results["main_object"] = main_object
        
        logger.info(f"Processing complete! Segmentation results saved to {output_dir}")
        logger.info(f"Total objects detected: {len(results['objects'])}")
        
        # Print each detected object
        for i, obj in enumerate(results['objects']):
            occlusion_status = ""
            if obj.get("is_occluded", False):
                occluders = obj.get('occluded_by', [])
                if isinstance(occluders, list):
                    occluders_str = ", ".join(occluders)
                else:
                    occluders_str = str(occluders)
                occlusion_status = f" [occluded by {occluders_str}]"
            logger.info(f"- ID {i}: {obj['class_name']} (confidence: {obj['confidence']:.2f}){occlusion_status}")
        
        return results
    except Exception as e:
        logger.error(f"Error occurred during execution: {str(e)}")
        raise

def process_occlusion_images(image_path, extracted_objects, occlusion_info, output_dir_absolute, current_dir, main_object="unknown"):
    """Generate mask and image maps for occluded objects"""
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            logger.error(f"Failed to read image: {image_path}")
            return
        
        h, w = original_image.shape[:2]
        
        object_masks = {}
        main_object_ids = [] 
        
        if main_object != "unknown":
            for i, obj in enumerate(extracted_objects):
                if obj["class_name"].lower() == main_object.lower():
                    main_object_ids.append(i)
                    # logger.info(f"Main object identified: {obj['class_name']} (ID: {i})")
        
        for i, obj in enumerate(extracted_objects):
            png_path = os.path.join(output_dir_absolute, f"{obj['class_name']}_{i}.png")
            if os.path.exists(png_path):
                object_img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
                if object_img is not None and object_img.shape[2] == 4:
                    mask = object_img[:, :, 3] > 0 

                    x1, y1, x2, y2 = map(int, obj["bbox"])
                    object_masks[i] = {
                        "mask": mask,
                        "bbox": [x1, y1, x2, y2],
                        "class_name": obj["class_name"],
                        "is_main_object": i in main_object_ids
                    }
        
        for i, obj in enumerate(extracted_objects):
            if obj.get("is_occluded", False):
                x1, y1, x2, y2 = map(int, obj["bbox"])
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                box_image = original_image.copy()
                cv2.rectangle(box_image, (x1, y1), (x2, y2), (0, 255, 0), 3) 
                
                box_output_path = os.path.join(output_dir_absolute, f"box_{obj['class_name']}_{i}.png")
                cv2.imwrite(box_output_path, box_image)
                # logger.info(f"Saved individual bounding box image: {box_output_path}")
                
                margin = 0
                x1_expanded = max(0, x1 - margin)
                y1_expanded = max(0, y1 - margin)
                x2_expanded = min(w, x2 + margin)
                y2_expanded = min(h, y2 + margin)
                
                cropped_image = original_image[y1_expanded:y2_expanded, x1_expanded:x2_expanded].copy()
                
                cropped_mask = np.ones((y2_expanded-y1_expanded, x2_expanded-x1_expanded, 4), dtype=np.uint8) * 255
                
                center_width = int((x2_expanded - x1_expanded) * 0.3)
                center_height = int((y2_expanded - y1_expanded) * 0.3)
                center_x = (x2_expanded - x1_expanded) // 2
                center_y = (y2_expanded - y1_expanded) // 2
                
                start_x = max(0, center_x - center_width // 2)
                end_x = min(x2_expanded - x1_expanded, center_x + center_width // 2)
                start_y = max(0, center_y - center_height // 2)
                end_y = min(y2_expanded - y1_expanded, center_y + center_height // 2)
                
                cropped_mask[start_y:end_y, start_x:end_x, 3] = 0
                
                occlusion_mask = np.zeros((y2_expanded-y1_expanded, x2_expanded-x1_expanded, 4), dtype=np.uint8)
                occlusion_mask[:, :, 3] = 255 
                
                if i in object_masks:
                    target_obj_mask = object_masks[i]
                    target_mask = target_obj_mask["mask"]
                    target_bbox = target_obj_mask["bbox"]
                    
                    occluders = obj.get("occluded_by", [])
                    if not isinstance(occluders, list):
                        occluders = [occluders] if occluders else []
                        
                    for other_id, other_mask_info in object_masks.items():
                        if other_id == i:
                            continue
                            
                        if other_mask_info.get("is_main_object", False):
                            logger.info(f"Skipping main object {other_mask_info['class_name']}_{other_id} as occluder")
                            continue
                            
                        other_class = other_mask_info["class_name"]
                        is_known_occluder = other_class in occluders
                        
                        ox1, oy1, ox2, oy2 = other_mask_info["bbox"]
                        
                        if (ox1 < x2 and ox2 > x1 and oy1 < y2 and oy2 > y1):
                            rel_ox1 = max(0, ox1 - x1_expanded)
                            rel_oy1 = max(0, oy1 - y1_expanded)
                            rel_ox2 = min(x2_expanded - x1_expanded, ox2 - x1_expanded)
                            rel_oy2 = min(y2_expanded - y1_expanded, oy2 - y1_expanded)
                            
                            orig_ox1 = max(0, rel_ox1 + x1_expanded - ox1)
                            orig_oy1 = max(0, rel_oy1 + y1_expanded - oy1)
                            orig_ox2 = orig_ox1 + (rel_ox2 - rel_ox1)
                            orig_oy2 = orig_oy1 + (rel_oy2 - rel_oy1)
                            
                            other_obj_mask = other_mask_info["mask"]
                            if (orig_ox2 > orig_ox1 and orig_oy2 > orig_oy1 and 
                                rel_ox2 > rel_ox1 and rel_oy2 > rel_oy1):
                                
                                occluder_mask_region = other_obj_mask[orig_oy1:orig_oy2, orig_ox1:orig_ox2]
                                
                                if (occluder_mask_region.shape[0] == rel_oy2 - rel_oy1 and 
                                    occluder_mask_region.shape[1] == rel_ox2 - rel_ox1):
                                    
                                    occlusion_mask[rel_oy1:rel_oy2, rel_ox1:rel_ox2, 3][occluder_mask_region] = 0
                                    
                                    # if is_known_occluder:
                                    #     logger.info(f"Confirmed {other_class}_{other_id} occludes {obj['class_name']}_{i}")
                                    # else:
                                    #     logger.info(f"Detected {other_class}_{other_id} may occlude {obj['class_name']}_{i}")
                
                image_output_path = os.path.join(output_dir_absolute, f"image_{obj['class_name']}_{i}.png")
                mask_output_path = os.path.join(output_dir_absolute, f"mask_{obj['class_name']}_{i}.png")
                cv2.imwrite(image_output_path, cropped_image)
                cv2.imwrite(mask_output_path, cropped_mask)
                
                occlusion_mask_path = os.path.join(output_dir_absolute, f"occlusion_mask_{obj['class_name']}_{i}.png")
                cv2.imwrite(occlusion_mask_path, occlusion_mask)
                
                # logger.info(f"Saved occluded object image: {image_output_path}")
                # logger.info(f"Saved regular mask: {mask_output_path}")
                # logger.info(f"Saved occlusion mask: {occlusion_mask_path}")
                
                obj["inpainting_image_path"] = image_output_path
                obj["inpainting_mask_path"] = mask_output_path
                obj["occlusion_mask_path"] = occlusion_mask_path
                obj["box_image_path"] = box_output_path
                
        return True
    except Exception as e:
        logger.error(f"Failed to generate masks and images for occluded objects: {str(e)}")
        return False

def test_imports():
    """Test if all modules are imported correctly"""
    try:
        # Test SAM2 module import
        logger.info("Testing SAM2 module import...")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Test Grounding DINO module import
        logger.info("Testing Grounding DINO module import...")
        from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
        
        logger.info("All modules imported successfully!")
        return True
    except ImportError as e:
        logger.error(f"Module import failed: {str(e)}")
        return False

if __name__ == "__main__":
    test = False

    image_file = "output_scene/scene_1/comfy_image/refined_scene_image.png"

    custom_output_dir = "output_scene/scene_1/output_assets/image"
    user_prompt = None
    scene_main_object = None

    box_thresh = 0.3
    text_thresh = 0.2
    proxy_url = "your_proxy_url"
    base_url = "your_base_url"
    api_key = "your_api_key"
    
    if test:
        test_imports()
    elif os.path.exists(image_file):
        get_asset_image_main(
            image=image_file,
            api_key=api_key,
            proxy_url=proxy_url,
            base_url=base_url,
            output_assets_dir=custom_output_dir,
            prompt=user_prompt,
            main_object=scene_main_object,
            box_threshold=box_thresh,
            text_threshold=text_thresh
        )
    else:
        logger.error(f"Input image not found: {image_file}")

