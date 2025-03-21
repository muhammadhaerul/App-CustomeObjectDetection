import streamlit as st
import numpy as np
import tempfile
import os
from PIL import Image
from ultralytics import YOLOWorld

# Set page configuration
st.set_page_config(
    page_title="Custom Object Detection App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling - dark theme matching the screenshot
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Dark theme */
    .stApp {
        background-color: #121212;
        color: white;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: white !important;
    }
    
    .main-header {
        font-size: 1.8rem;
        text-align: center;
        margin-bottom: 1rem;
        color: white;
    }
    
    .step-header {
        font-size: 1.2rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    /* Sidebar styling - wider */
    [data-testid="stSidebar"] {
        min-width: 300px !important;
        max-width: 300px !important;
    }
    
    .css-1d391kg, .css-163ttbj, .css-1oe6o3n {
        background-color: #1E1E1E;
    }
    
    /* Input fields - dark style */
    .stTextInput > div > div, .stNumberInput > div > div {
        background-color: #2E2E2E;
        color: white;
        border: 1px solid #444;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #444;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #555;
    }
    
    /* File uploader styling */
    .stUploadButton > div {
        background-color: #2E2E2E !important;
        border: 1px dashed #555 !important;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 0.5rem;
        background-color: #121212;
        border-top: 1px solid #333;
        color: #888;
        font-size: 0.8rem;
    }
    
    /* Slider styling */
    .stSlider {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
    }
    
    /* File upload area */
    .upload-container {
        border: 1px dashed #555;
        border-radius: 5px;
        padding: 2rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .css-1adrfps {
        background-color: #2E2E2E;
    }
    
    /* Remove padding from expander */
    .st-emotion-cache-1erivf3 {
        padding-top: 0 !important;
    }
    
    /* Clean up numbered lists */
    ol {
        padding-left: 1.5rem;
    }
    
    ol li {
        margin-bottom: 0.5rem;
    }
    
    /* Results placeholder */
    .results-placeholder {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Instructions and info
with st.sidebar:
    st.markdown("### How to Use")
    
    st.markdown("""
    1. **Add Classes:** Enter the names of objects you want to detect
    2. **Upload Media:** Upload an image or video file
    3. **Adjust Settings:** Configure detection parameters
    4. **Run Detection:** Click the button to start detection
    5. **View Results:** See original vs detected media side by side
    """)
    
    st.markdown("### About YOLOWorld")
    st.markdown("""
    YOLOWorld is an advanced version of YOLO (You Only Look Once) that can detect multiple classes of objects in images and videos.
    
    It provides open-vocabulary object detection capabilities with high accuracy and performance.
    """)

# Main content
st.markdown('<h1 class="main-header">Custom Object Detection using Prompt</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; margin-bottom: 2rem;">Prompt-based detection powered by YOLOWorld</p>', unsafe_allow_html=True)

# Initialize session state
if 'num_classes' not in st.session_state:
    st.session_state.num_classes = 1
if 'detection_complete' not in st.session_state:
    st.session_state.detection_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# Step 1: Define Objects to Detect
st.markdown('<div class="step-header">Step 1: Define Objects to Detect</div>', unsafe_allow_html=True)

# First ask how many classes
num_classes = st.number_input(
    "Number of classes to detect", 
    min_value=1, 
    max_value=10, 
    value=st.session_state.num_classes,
    step=1
)

# Update session state if changed
if num_classes != st.session_state.num_classes:
    st.session_state.num_classes = num_classes
    st.rerun()

# Create class inputs
class_names = []
for i in range(st.session_state.num_classes):
    class_label = f"Class {i+1}"
    default_value = "person" if i == 0 else ""
    class_input = st.text_input(class_label, value=default_value)
    if class_input:
        class_names.append(class_input)

# Step 2: Upload Media
st.markdown('<div class="step-header">Step 2: Upload Media</div>', unsafe_allow_html=True)
st.markdown("Upload an image or video file")

# Create upload area
uploaded_file = st.file_uploader(
    "Drag and drop file here",
    type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
    label_visibility="collapsed"
)

# File size limits note
st.caption("Limit 200MB per file • JPG, JPEG, PNG, MP4, AVI, MPEG4")

# Step 3: Detection Settings
st.markdown('<div class="step-header">Step 3: Detection Settings</div>', unsafe_allow_html=True)

# Multiple configuration options
settings_cols = st.columns(2)

with settings_cols[0]:
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.25, 
        step=0.05,
        help="Minimum confidence score for detection"
    )

with settings_cols[1]:
    iou_threshold = st.slider(
        "IoU Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.45, 
        step=0.05,
        help="Intersection over Union threshold for NMS"
    )



# Step 4: Run Detection
st.markdown('<div class="step-header">Step 4: Run Detection</div>', unsafe_allow_html=True)

# Detection function
def run_detection(file, classes, conf_threshold, iou_threshold):
    # Create temp file to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    
    # Filter out empty classes
    valid_classes = [cls for cls in classes if cls.strip()]
    
    if not valid_classes:
        st.error("Please enter at least one class name.")
        return None, None, None, None
    
    # Initialize model
    with st.spinner("Load YOLOWorld model..."):
        model = YOLOWorld("yolov8s-worldv2.pt")
        
        # Set custom classes
        model.set_classes(valid_classes)
    
    # Check if it's an image or video
    file_extension = os.path.splitext(file.name)[1].lower()
    
    if file_extension in ['.jpg', '.jpeg', '.png']:
        # For image
        original_image = Image.open(temp_file_path)
        
        # Run object detection
        with st.spinner("Detecting objects..."):
            results = model.predict(
                temp_file_path, 
                conf=conf_threshold, 
                iou=iou_threshold
            )
            
            # Get the detection results
            detections = results[0]
            
            # Create a copy of the original image for drawing
            original_np = np.array(original_image)
            
            # Draw bounding boxes on the original image
            for box in detections.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class and confidence
                class_id = int(box.cls)
                class_name = detections.names[class_id]
                conf = float(box.conf)
                
                # Draw rectangle on original image
                cv2.rectangle(original_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add label with confidence
                label = f"{class_name} {conf:.2f}"
                cv2.putText(original_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(original_np)
            
            # Get detection summary
            detection_summary = get_detection_summary(results[0])
            
        return results, original_image, processed_image, detection_summary
    
    # Video processing placeholder
    if file_extension in ['.mp4', '.avi', '.mov']:
        # Save original image for display
        st.warning("Video processing is coming soon. Please upload an image for now.")
        return None, None, None, None
    
    else:
        st.error("Unsupported file format.")
        return None, None, None, None

# Function to get detection summary
def get_detection_summary(result):
    boxes = result.boxes
    class_counts = {}
    
    # If no detections
    if len(boxes) == 0:
        return {"No objects detected": 0}
    
    # Count objects per class
    for box in boxes:
        class_id = int(box.cls)
        class_name = result.names[class_id]
        confidence = float(box.conf)
        
        if class_name in class_counts:
            class_counts[class_name]["count"] += 1
            class_counts[class_name]["confidences"].append(confidence)
        else:
            class_counts[class_name] = {
                "count": 1, 
                "confidences": [confidence]
            }
    
    # Calculate average confidence per class
    for class_name in class_counts:
        confidences = class_counts[class_name]["confidences"]
        class_counts[class_name]["avg_confidence"] = sum(confidences) / len(confidences)
    
    return class_counts

# Run detection button
run_detection_btn = st.button("Run Detection", type="primary")

if run_detection_btn and uploaded_file is not None:
    # Run detection
    results, original_image, processed_image, detection_summary = run_detection(
        uploaded_file, 
        class_names,
        confidence_threshold,
        iou_threshold
    )
    
    if results is not None:
        st.session_state.detection_complete = True
        st.session_state.results = results
        st.session_state.original_image = original_image
        st.session_state.processed_image = processed_image
        st.session_state.detection_summary = detection_summary

# Step 5: Display Results
st.markdown('<div class="step-header">Step 5: View Results</div>', unsafe_allow_html=True)

if st.session_state.detection_complete:
    # Display images side by side
    result_cols = st.columns(2)
    
    with result_cols[0]:
        st.subheader("Original Image")
        st.image(st.session_state.original_image, use_column_width=True)
    
    with result_cols[1]:
        st.subheader("Detection Result")
        st.image(st.session_state.processed_image, use_column_width=True)
    
    # Display enhanced detection summary
    st.markdown('<h3 style="margin-top: 1.5rem;">Detection Summary</h3>', unsafe_allow_html=True)
    
    if hasattr(st.session_state, 'detection_summary') and st.session_state.detection_summary:
        summary = st.session_state.detection_summary
        
        # Create a table for class counts and confidence
        summary_table = []
        for class_name, data in summary.items():
            if class_name != "No objects detected":
                summary_table.append({
                    "Class": class_name,
                    "Count": data["count"],
                    "Avg. Confidence": f"{data['avg_confidence']:.2%}"
                })
        
        if summary_table:
            # Display the table with dark styling
            st.markdown(
                """
                <style>
                .summary-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 1rem;
                    background-color: #1E1E1E;
                }
                .summary-table th {
                    background-color: #333;
                    color: white;
                    text-align: left;
                    padding: 8px 12px;
                }
                .summary-table td {
                    padding: 8px 12px;
                    border-top: 1px solid #444;
                }
                .summary-table tr:nth-child(even) {
                    background-color: #2A2A2A;
                }
                .summary-info {
                    background-color: #1E1E1E;
                    padding: 1rem;
                    border-radius: 4px;
                    margin-top: 1rem;
                }
                .summary-info p {
                    margin-bottom: 0.5rem;
                }
                </style>
                """, 
                unsafe_allow_html=True
            )
            
            # Calculate total objects
            total_objects = sum(data["count"] for cls, data in summary.items() if cls != "No objects detected")
            
            # Generate HTML table
            table_html = '<table class="summary-table"><tr><th>Class</th><th>Count</th><th>Avg. Confidence</th></tr>'
            for row in summary_table:
                table_html += f'<tr><td>{row["Class"]}</td><td>{row["Count"]}</td><td>{row["Avg. Confidence"]}</td></tr>'
            table_html += '</table>'
            
            st.markdown(table_html, unsafe_allow_html=True)
            
            # Calculate additional metrics
            if len(summary.items()) > 0 and total_objects > 0:
                # Find most frequent class
                most_common_class = max([(cls, data["count"]) for cls, data in summary.items() 
                                         if cls != "No objects detected"], key=lambda x: x[1])
                
                # Find highest confidence class
                highest_conf_class = max([(cls, data["avg_confidence"]) for cls, data in summary.items() 
                                          if cls != "No objects detected"], key=lambda x: x[1])
                
                # Generate timestamp
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Enhanced info panel
                st.markdown(
                    f"""
                    <div class="summary-info">
                        <p><strong>Total objects detected:</strong> {total_objects}</p>
                        <p><strong>Most frequent class:</strong> {most_common_class[0]} ({most_common_class[1]} instances)</p>
                        <p><strong>Highest confidence class:</strong> {highest_conf_class[0]} ({highest_conf_class[1]:.2%})</p>
                        <p><strong>Classes detected:</strong> {", ".join([cls for cls in summary.keys() if cls != "No objects detected"])}</p>
                        <p><strong>Detection settings:</strong> Confidence threshold: {confidence_threshold}, IoU threshold: {iou_threshold}</p>
                        <p><strong>Detection time:</strong> {timestamp}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("No objects were detected. Try adjusting your class names or confidence threshold.")
        else:
            st.info("No objects were detected. Try adjusting your class names or confidence threshold.")
else:
    st.markdown(
        '<div class="results-placeholder">Run detection to see results here</div>',
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    '<div class="footer">Developed by @muhammadhaerul</div>',
    unsafe_allow_html=True
)
