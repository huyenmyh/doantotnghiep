from ultralytics import YOLO
import socket
from flask import Flask, request, render_template, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
import subprocess  # Thêm thư viện subprocess để sử dụng ffmpeg

# Initialize Flask application
app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/Output/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set maximum upload size (50MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(model_path='models/yolov8n.pt'):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Global YOLO model
model = load_model()

def process_image(filepath, output_path):
    try:
        # Run YOLO model
        results = model(filepath)

        # Open original image
        image = Image.open(filepath).convert('RGB')
        draw = ImageDraw.Draw(image)

        # Process each detection
        for result in results[0].boxes.xywh.cpu().numpy():
            x_center, y_center, width, height = result[:4]
            x1 = int((x_center - width / 2) * image.width)
            y1 = int((y_center - height / 2) * image.height)
            x2 = int((x_center + width / 2) * image.width)
            y2 = int((y_center + height / 2) * image.height)

            # Apply effects for road damage (class 0)
            if result[-1] == 0:
                region = image.crop((x1, y1, x2, y2))
                blurred_region = region.filter(ImageFilter.GaussianBlur(5))
                image.paste(blurred_region, (x1, y1, x2, y2))

        # Process segmentation masks
        if results[0].masks is not None:
            for i, mask in enumerate(results[0].masks.data.cpu().numpy()):
                if results[0].boxes.cls[i] == 0:
                    mask_image = Image.fromarray((mask * 255).astype('uint8'))
                    mask_image = mask_image.resize((image.width, image.height))
                    mask_color = Image.new('RGB', image.size, (255, 0, 0))
                    mask_image = mask_image.filter(ImageFilter.GaussianBlur(5))
                    image = Image.composite(mask_color, image, mask_image.convert('L'))

        image.save(output_path)
        print(f"Saved output image at: {output_path}")
    except Exception as e:
        print(f"Error processing image: {e}")

def process_video(input_path, output_path):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return

        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec for better browser support
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO model
            results = model(frame)

            # Process bounding boxes and masks
            for box, conf, cls, mask in zip(results[0].boxes.xyxy.cpu().numpy(),
                                            results[0].boxes.conf.cpu().numpy(),
                                            results[0].boxes.cls.cpu().numpy(),
                                            (results[0].masks.data.cpu().numpy() if results[0].masks is not None else [])):
                x1, y1, x2, y2 = map(int, box)
                confidence = conf
                class_label = "Pothole" if int(cls) == 0 else f"Class {int(cls)}"

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Add label
                label = f"{class_label} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_bg = (x1, y1 - label_size[1] - 5, x1 + label_size[0] + 5, y1)
                cv2.rectangle(frame, (label_bg[0], label_bg[1]), (label_bg[2], label_bg[3]), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Apply mask for class 0 (Pothole)
                if int(cls) == 0 and mask is not None:
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)

                    # Create colored overlay for the mask
                    red_overlay = np.zeros_like(frame, dtype=np.uint8)
                    red_overlay[:, :, 2] = mask_binary * 255  # Set red channel

                    # Blend mask with frame
                    frame = cv2.addWeighted(frame, 1, red_overlay, 0.5, 0)

            # Write the frame to the output video
            out.write(frame)

        cap.release()
        out.release()
        print(f"Saved output video at: {output_path}")

        # Convert the processed video to ensure it's browser-compatible
        convert_video_to_h264(output_path, output_path)
    except Exception as e:
        print(f"Error processing video: {e}")


def convert_video_to_h264(input_path, output_path):
    try:
        command = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            output_path
        ]
        subprocess.run(command, check=True)
        print(f"Video converted to H.264 and saved at: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")

@app.route('/')
def index():
    return render_template('index1.html', results=None, uploaded_filename=None, 
                         image_with_boxes_path=None, video_with_segment_path=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Handle image files
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'gif')):
            output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_' + filename)
            process_image(filepath, output_image_path)
            image_url = url_for('static', filename='Output/output_' + filename)
            return render_template('index1.html', uploaded_filename=filename, 
                                image_with_boxes_path=image_url, video_with_segment_path=None)

        # Handle video files
        elif filename.lower().endswith(('mp4', 'avi', 'mov', 'mkv')):
            output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_' + filename)
            process_video(filepath, output_video_path)
            video_url = url_for('static', filename='Output/output_' + filename)
            return render_template('index1.html', uploaded_filename=filename, 
                                image_with_boxes_path=None, video_with_segment_path=video_url)

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == "__main__":
    app.run(port=5001, debug=True, use_reloader=False)

