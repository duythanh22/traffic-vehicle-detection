import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image


def create_xml(image_path, label_path, output_path, class_names):
    # Read image dimensions
    img = Image.open(image_path)
    width, height = img.size

    # Parse YOLO label file
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, w, h = map(float, line.strip().split(','))

                # Convert YOLO coordinates to VOC format
                x_min = int((x_center - w / 2) * width)
                y_min = int((y_center - h / 2) * height)
                x_max = int((x_center + w / 2) * width)
                y_max = int((y_center + h / 2) * height)

                # Ensure coordinates are within image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(width, x_max)
                y_max = min(height, y_max)

                boxes.append({
                    'class': class_names[int(class_id)],
                    'xmin': x_min,
                    'ymin': y_min,
                    'xmax': x_max,
                    'ymax': y_max
                })

    # Create XML structure
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = os.path.basename(os.path.dirname(os.path.dirname(image_path)))

    filename = ET.SubElement(annotation, 'filename')
    filename.text = os.path.basename(image_path)

    path = ET.SubElement(annotation, 'path')
    path.text = image_path

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    width_elem = ET.SubElement(size, 'width')
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, 'height')
    height_elem.text = str(height)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(img.layers if hasattr(img, 'layers') else 3)

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    # Add object elements
    for box in boxes:
        obj = ET.SubElement(annotation, 'object')

        name = ET.SubElement(obj, 'name')
        name.text = box['class']

        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'

        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'

        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'

        bndbox = ET.SubElement(obj, 'bndbox')

        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(box['xmin'])

        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(box['ymin'])

        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(box['xmax'])

        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(box['ymax'])

    # Create pretty XML string
    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent='    ')

    # Save XML file
    with open(output_path, 'w') as f:
        f.write(xml_str)


def convert_dataset(base_dir, class_names):
    for split in ['train', 'dev']:
        images_dir = os.path.join(base_dir, split, 'images')
        labels_dir = os.path.join(base_dir, split, 'labels')
        annotations_dir = os.path.join(base_dir, split, 'annotations')

        # Create annotations directory if it doesn't exist
        os.makedirs(annotations_dir, exist_ok=True)

        # Process each image
        for image_file in os.listdir(images_dir):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(images_dir, image_file)
            label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')
            xml_path = os.path.join(annotations_dir, os.path.splitext(image_file)[0] + '.xml')

            create_xml(image_path, label_path, xml_path, class_names)
            print(f"Processed {image_file}")


# Example usage
if __name__ == "__main__":
    # Define your class names (in order corresponding to the class IDs in YOLO format)
    class_names = ['0', '1', '2', '3']  # Replace with your actual class names

    # Base directory containing train and val folders
    base_dir = r"D:\DUYTHANH\SoICT-traffic-detec\data_soict_2024"  # Replace with your dataset path

    convert_dataset(base_dir, class_names)