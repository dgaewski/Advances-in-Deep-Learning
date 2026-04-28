from pathlib import Path
import json

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
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

    #the ego car is the one closest to center - need to loop
    ego_car = None
    for kart in kart_obj:
        if kart['is_center_kart'] == True:
            ego_car = kart['kart_name']
            break

    # 1. Ego car
    # {kart_name} is the ego car.
    c1 = {
        "image_file": image_file,
        "caption": f"{ego_car} is the ego car."
        }

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    num_karts = 0
    for kart in kart_obj:
        num_karts += 1

    c2 = {
        "image_file": image_file,
        "caption": f"There are {num_karts} karts in the scenario."
        }

    # 3. Track name
    # The track is {track_name}.
    
    c3 = {
        "image_file": image_file,
        "caption": f"The track is {track_info}."
        }

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    ego_center = None
    for kart in kart_obj:
        if kart['is_center_kart'] == True:
            ego_center = kart['center']
            break

    count_left = 0
    count_right = 0
    count_front = 0
    count_back = 0

    #collect c4 captions:
    c4_captions = []

    # Loop through non-ego karts for question 4
    for kart in kart_obj:
        if kart['is_center_kart']:
            continue  # skip ego kart

        # determine left/right
        if kart['center'][0] < ego_center[0]:
            lr = "left"
        else:
            lr = "right"

        # determine front/back (lower y = front)
        if kart['center'][1] < ego_center[1]:
            fb = "front"
        else:
            fb = "back"

        kart_name = kart['kart_name']

        c4_captions.append({
            "image_file": image_file,
            "caption": f"{kart_name} is {fb} and {lr} of the ego car."
            })
        
    return [c1, c2, c3] + c4_captions

    raise NotImplementedError("Not implemented")


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""
def generate_all(data_dir: str = "data/train"):
    from .generate_qa import extract_kart_objects, extract_track_info
    
    data_path = Path(data_dir)
    info_files = sorted(data_path.glob("*_info.json"))

    all_captions = []

    for info_file in info_files:
        for view_index in range(10):
            try:
                captions = generate_caption(str(info_file), view_index)
                all_captions.extend(captions)
            except Exception as e:
                print(f"Error on {info_file} view {view_index}: {e}")

    output_file = data_path / "generated_captions.json"
    with open(output_file, "w") as f:
        json.dump(all_captions, f, indent=2)

    print(f"Generated {len(all_captions)} captions -> {output_file}")



def main():
    fire.Fire({"check": check_caption, "generate_all": generate_all})


if __name__ == "__main__":
    main()
