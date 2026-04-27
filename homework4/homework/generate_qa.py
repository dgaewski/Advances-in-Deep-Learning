import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    #setup our list of karts found
    karts = []

    #Load each detection
    for detection in info['detections'][view_index]:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        #we need to scale detection coordinates to match the scaling we do on the image from 600X400 -> 150X100 before doing calculations
        # Scale coordinates to fit the current image size
            # Calculate scaling factors
        scale_x = img_width / ORIGINAL_WIDTH
        scale_y = img_height / ORIGINAL_HEIGHT

        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue


        kart_name = info['karts'][track_id]

        #compute the centers for this kart
        x_center = (x2_scaled + x1_scaled) / 2
        y_center = (y2_scaled + y1_scaled) / 2

        center = (x_center, y_center)

        karts.append({
            'instance_id': track_id,
            'kart_name': kart_name,
            'center': center,
            'is_center_kart': False  #we will determine once all karts are processed
            })

    #for each kart in list, find out if its closest to center
    closest_kart = None
    dist = 50000
    img_center_x = img_width / 2    
    img_center_y = img_height / 2

    for i, kart in enumerate(karts):
        
        kart_x = kart['center'][0]
        kart_y = kart['center'][1]

        #find distance to center - distance formula
        k_dist = ( (kart_x - img_center_x) ** 2 + (kart_y - img_center_y) ** 2 ) ** 0.5

        if k_dist <= dist:
            #update closest dist
            dist = k_dist
            #update closest kart
            closest_kart = i

    #update the correct kart that is center kart
    if karts:
        karts[closest_kart]['is_center_kart'] = True

    #return a list with: instance_id, kart_name, center, is_center_kart (bool)
    return karts
    raise NotImplementedError("Not implemented")


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)
    
    return info['track']

    raise NotImplementedError("Not implemented")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """

    #extract info for this view to ask questions
    kart_obj = extract_kart_objects(info_path, view_index,img_width,img_height)
    track_info = extract_track_info(info_path)

    # No visible karts in this view — skip it
    if not kart_obj:
        return []

    # Find corresponding image file
    info_path = Path(info_path)
    base_name = info_path.stem.replace("_info", "")
    #image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]
    image_file = f"{info_path.parent.name}/{base_name}_{view_index:02d}_im.jpg"


    '''
    kart_obj structure:
    {'instance_id': 0, 'kart_name': 'nolok',  'center': (75, 78), 'is_center_kart': True},
    {'instance_id': 5, 'kart_name': 'konqi',  'center': (65, 50), 'is_center_kart': False},
    {'instance_id': 9, 'kart_name': 'gnu',    'center': (90, 45), 'is_center_kart': False},
    
    '''

    # 1. Ego car question
    # What kart is the ego car?

    #the ego car is the one closest to center - need to loop
    ego_car = None
    for kart in kart_obj:
        if kart['is_center_kart'] == True:
            ego_car = kart['kart_name']
            break

    q1 = {
        "question": "What kart is the ego car?",
        "answer": ego_car,
        "image_file": image_file
    }
    
    # 2. Total karts question
    # How many karts are there in the scenario?

    num_karts = 0
    for kart in kart_obj:
        num_karts += 1

    q2 = {
        "question": "How many karts are there in the scenario?",
        "answer": str(num_karts),
        "image_file": image_file
    }

    # 3. Track information questions
    # What track is this?

    q3 = {
        "question": "What track is this?",
        "answer": track_info,
        "image_file": image_file
    }

    # 4. Relative position questions for each kart

    ego_center = None
    for kart in kart_obj:
        if kart['is_center_kart'] == True:
            ego_center = kart['center']
            break

    #setup counters to use in q5
    count_left = 0
    count_right = 0
    count_front = 0
    count_back = 0


    #collect q4 questions:
    q4_questions = []

    # Loop through non-ego karts for question 4
    for kart in kart_obj:
        if kart['is_center_kart']:
            continue  # skip ego kart

        # determine left/right
        if kart['center'][0] < ego_center[0]:
            lr = "left"
            count_left += 1
        else:
            lr = "right"
            count_right += 1

        # determine front/back (lower y = front)
        if kart['center'][1] < ego_center[1]:
            fb = "front"
            count_front += 1
        else:
            fb = "back"
            count_back += 1


        # Is {kart_name} to the left or right of the ego car?

        q4_questions.append({
            "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
            "answer": lr,
            "image_file": image_file
            })

        # Is {kart_name} in front of or behind the ego car?
        q4_questions.append({
            "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
            "answer": fb,
            "image_file": image_file
            })

        # Where is {kart_name} relative to the ego car?
        q4_questions.append({
            "question": f"Where is {kart['kart_name']} relative to the ego car?",
            "answer": f"{fb} and {lr}",
            "image_file": image_file
            })

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    q5a = {
        "question": "How many karts are to the left of the ego car?",
        "answer": str(count_left),
        "image_file": image_file
        }
    # How many karts are to the right of the ego car?
    q5b = {
        "question": "How many karts are to the right of the ego car?",
        "answer": str(count_right),
        "image_file": image_file
        }
    # How many karts are in front of the ego car?
    q5c = {
        "question": "How many karts are in front of the ego car?",
        "answer": str(count_front),
        "image_file": image_file
        }
    # How many karts are behind the ego car?
    q5d = {
        "question": "How many karts are behind the ego car?",
        "answer": str(count_back),
        "image_file": image_file
        }
    
    #assemble the 10 questions into a list and return

    return [q1, q2, q3] + q4_questions + [q5a, q5b, q5c, q5d]

    raise NotImplementedError("Not implemented")


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""

def validate(gt_file: str = "data/valid_grader/balanced_qa_pairs.json", 
             info_dir: str = "data/valid"):
    """Compare generated QA pairs against validation ground truth."""

    with open(gt_file) as f:
        gt_pairs = json.load(f)

    correct = 0
    total = 0

    for gt in gt_pairs:
        parts = gt["image_file"].split("/")
        filename = parts[-1]
        pieces = filename.split("_")
        frame_id = pieces[0]
        view_idx = int(pieces[1])

        info_file = str(Path(info_dir) / f"{frame_id}_info.json")

        qa_pairs = generate_qa_pairs(info_file, view_idx)

        match = None
        for qa in qa_pairs:
            if qa["question"] == gt["question"]:
                match = qa
                break

        if match is None:
            print(f"MISSING: {gt['question']} for {gt['image_file']}")
            total += 1
            continue

        if match["answer"] == gt["answer"]:
            correct += 1
        else:
            print(f"WRONG: {gt['question']}")
            print(f"  Expected: {gt['answer']}")
            print(f"  Got:      {match['answer']}")

        total += 1

    print(f"\nAlignment: {correct}/{total} ({100*correct/total:.1f}%)")

def generate_all(data_dir: str = "data/train"):
    data_path = Path(data_dir)
    info_files = sorted(data_path.glob("*_info.json"))
    
    all_qa_pairs = []
    
    for info_file in info_files:
        for view_index in range(10):
            try:
                qa_pairs = generate_qa_pairs(str(info_file), view_index)
                all_qa_pairs.extend(qa_pairs)
            except Exception as e:
                print(f"Error on {info_file} view {view_index}: {e}")
    
    # Save to a file that data.py will pick up
    output_file = data_path / "generated_qa_pairs.json"
    with open(output_file, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    print(f"Generated {len(all_qa_pairs)} QA pairs -> {output_file}")

def main():
    fire.Fire({"check": check_qa_pairs, "generate_all": generate_all, "validate": validate})


if __name__ == "__main__":
    main()
