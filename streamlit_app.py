import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import mediapipe as mp
import tempfile
import os
from pathlib import Path
from datetime import datetime

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create videos directory if it doesn't exist
VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(exist_ok=True)

def get_timestamp():
    """Return current timestamp in a clean format"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_and_save_video(video_path, prefix, max_frames=None):
    """Process video and save annotated version with timestamp"""
    timestamp = get_timestamp()
    output_path = VIDEO_DIR / f"{prefix}_{timestamp}.mp4"
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    keypoints = []

    while cap.isOpened() and (max_frames is None or frame_count < max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            kps = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            keypoints.append(kps)

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame_rgb,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Convert back to BGR for video writing
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        frame_count += 1

    cap.release()
    out.release()
    pose.close()
    
    return np.array(keypoints), frame_count, str(output_path)

def get_latest_videos():
    """Get the most recent reference and student videos"""
    ref_videos = sorted(VIDEO_DIR.glob("reference_*.mp4"), key=os.path.getmtime, reverse=True)
    stu_videos = sorted(VIDEO_DIR.glob("student_*.mp4"), key=os.path.getmtime, reverse=True)
    
    latest_ref = str(ref_videos[0]) if ref_videos else None
    latest_stu = str(stu_videos[0]) if stu_videos else None
    
    return latest_ref, latest_stu

def compare_videos(ref_path, stu_path, max_frames=100):
    """Compare two videos and return results"""
    # Process reference video
    ref_kps, ref_count, ref_output = extract_and_save_video(
        ref_path, 
        "reference", 
        max_frames
    )
    
    # Process student video
    stu_kps, stu_count, stu_output = extract_and_save_video(
        stu_path, 
        "student", 
        max_frames
    )
    
    # Calculate DTW distance if we have keypoints
    if len(ref_kps) > 0 and len(stu_kps) > 0:
        distance, path = fastdtw(
            ref_kps.reshape(len(ref_kps), -1),
            stu_kps.reshape(len(stu_kps), -1),
            dist=euclidean
        )
        normalized_distance = distance / max(len(ref_kps), len(stu_kps))
        
        return {
            'dtw_distance': distance,
            'normalized_distance': normalized_distance,
            'ref_frames': ref_count,
            'stu_frames': stu_count,
            'keypoints_detected': (len(ref_kps), len(stu_kps)),
            'ref_output': ref_output,
            'stu_output': stu_output
        }
    return None

def main():
    st.title("Pose Comparison App")
    st.write("Upload videos to compare poses side-by-side")
    
    # File uploaders
    ref_file = st.file_uploader("Reference Video", type=["mp4", "mov", "avi"])
    stu_file = st.file_uploader("Student Video", type=["mp4", "mov", "avi"])
    
    max_frames = st.slider("Max frames to process", 10, 500, 100)
    
    if ref_file and stu_file:
        # Save uploaded files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as ref_temp:
            ref_temp.write(ref_file.read())
            ref_path = ref_temp.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as stu_temp:
            stu_temp.write(stu_file.read())
            stu_path = stu_temp.name
        
        # Process and compare
        with st.spinner('Processing videos...'):
            results = compare_videos(ref_path, stu_path, max_frames)
            
            # Get the latest videos (should be the ones we just processed)
            latest_ref, latest_stu = get_latest_videos()
            
            if latest_ref and latest_stu:
                # Display side-by-side comparison
                st.write("## Side-by-Side Comparison")
                st.write(f"Processed at {get_timestamp()}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Reference Video")
                    st.video(latest_ref)
                    if results:
                        st.write(f"Poses detected: {results['keypoints_detected'][0]}/{results['ref_frames']} frames")
                
                with col2:
                    st.write("### Student Video")
                    st.video(latest_stu)
                    if results:
                        st.write(f"Poses detected: {results['keypoints_detected'][1]}/{results['stu_frames']} frames")
            
                # Show comparison results if available
                if results:
                    st.write("## Comparison Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("DTW Distance", f"{results['dtw_distance']:.2f}")
                    with col2:
                        st.metric("Normalized Distance", f"{results['normalized_distance']:.4f}")
        
        # Clean up
        os.unlink(ref_path)
        os.unlink(stu_path)

if __name__ == "__main__":
    main()