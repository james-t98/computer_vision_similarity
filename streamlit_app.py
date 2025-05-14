# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# from scipy.spatial.distance import cosine
# import tempfile
# import os
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="Athlete Similarity", layout="wide")

# # --- Helper Functions ---
# def extract_frames(video_path, max_frames=100):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     count = 0
#     while cap.isOpened() and count < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, (512, 512))
#         frames.append(frame)
#         count += 1
#     cap.release()
#     return frames

# def extract_pose_keypoints(frames):
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose()
#     keypoints_list = []

#     for frame in frames:
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(rgb_frame)

#         if results.pose_landmarks:
#             keypoints = []
#             for landmark in results.pose_landmarks.landmark:
#                 keypoints.extend([landmark.x, landmark.y, landmark.z])
#             keypoints_list.append(keypoints)
#         else:
#             keypoints_list.append([0] * (33 * 3))
    
#     pose.close()
#     return np.array(keypoints_list)

# def generate_embedding(keypoints_array):
#     return np.mean(keypoints_array, axis=0)

# def compute_similarity(embedding1, embedding2):
#     return 1 - cosine(embedding1, embedding2)

# def visualize_similarity(pose1, pose2):
#     min_len = min(len(pose1), len(pose2))
#     distances = [np.linalg.norm(p1 - p2) for p1, p2 in zip(pose1[:min_len], pose2[:min_len])]
#     fig, ax = plt.subplots()
#     ax.plot(distances, color='orange')
#     ax.set_title("Pose Distance Over Time")
#     ax.set_xlabel("Frame")
#     ax.set_ylabel("L2 Distance")
#     ax.grid(True)
#     st.pyplot(fig)

# # --- Streamlit UI ---
# st.title("🏃 Athlete Similarity Comparison (MediaPipe + AI)")

# col1, col2 = st.columns(2)

# with col1:
#     video1 = st.file_uploader("Upload First Athlete Video", type=["mp4", "mov"], key="video1")

# with col2:
#     video2 = st.file_uploader("Upload Second Athlete Video", type=["mp4", "mov"], key="video2")

# if video1 and video2:
#     with st.spinner("🔍 Processing videos and extracting poses..."):
#         temp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         temp1.write(video1.read())
#         temp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         temp2.write(video2.read())

#         frames1 = extract_frames(temp1.name)
#         frames2 = extract_frames(temp2.name)

#         pose1 = extract_pose_keypoints(frames1)
#         pose2 = extract_pose_keypoints(frames2)

#         emb1 = generate_embedding(pose1)
#         emb2 = generate_embedding(pose2)

#         score = compute_similarity(emb1, emb2)

#         os.remove(temp1.name)
#         os.remove(temp2.name)

#     st.success(f"✅ Similarity Score: **{score:.4f}** (1.0 = identical, 0 = different)")

#     st.markdown("---")
#     st.subheader("Pose Comparison Details")
#     st.write("Pose embeddings were calculated using 3D keypoints from MediaPipe Pose, averaged across frames from each video.")

#     visualize_similarity(pose1, pose2)

# else:
#     st.info("Please upload two video files to begin the comparison.")

import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Athlete Pose Comparison", layout="wide")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Moved to global scope

def extract_frames(video_path, max_frames=150, step=2):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.resize(frame, (512, 512))
            frames.append(frame)
        count += 1
    cap.release()
    return frames

def extract_keypoints(frames):
    pose = mp_pose.Pose()
    keypoints_list = []
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            keypoints_list.append(keypoints)
        else:
            keypoints_list.append([0] * (33 * 3))
    pose.close()
    return np.array(keypoints_list)

def draw_pose_on_frame(frame, pose_landmarks):
    annotated_frame = frame.copy()
    mp_drawing.draw_landmarks(
        annotated_frame,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )
    return annotated_frame

def compute_cosine_similarity(pose1, pose2):
    emb1 = np.mean(pose1, axis=0)
    emb2 = np.mean(pose2, axis=0)
    return 1 - cosine(emb1, emb2)

def compute_dtw_similarity(pose1, pose2):
    distance, _ = fastdtw(pose1, pose2, dist=lambda x, y: cosine(x, y))
    return distance

def plot_trajectory(pose_sequence, label):
    pose_sequence = np.array(pose_sequence)
    if pose_sequence.ndim == 2 and pose_sequence.shape[1] >= 3:
        x = pose_sequence[:, 0]
        y = pose_sequence[:, 1]
        plt.plot(x, y, label=label)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Trajectory of Nose Keypoint")
        plt.legend()

def create_annotated_video(frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, fourcc, 15, (width, height))
    pose = mp_pose.Pose()
    
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            annotated = draw_pose_on_frame(frame, results.pose_landmarks)
        else:
            annotated = frame
        
        # Convert back to BGR for video writing
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        out.write(annotated)
    
    out.release()
    pose.close()

st.title("🏃 Athlete Pose Comparison with Real-Time Overlays")

col1, col2 = st.columns(2)
with col1:
    video1 = st.file_uploader("Upload Reference Video (No Overlay)", type=["mp4"], key="v1")
with col2:
    video2 = st.file_uploader("Upload Comparison Video (Overlayed)", type=["mp4"], key="v2")

if video1 and video2:
    slow_motion = st.slider("Playback Speed (1 = normal, higher = slower)", 1, 5, 2)
    with st.spinner("Processing videos..."):
        # Save uploaded videos to temporary files
        t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t1.write(video1.read())
        t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t2.write(video2.read())
        t1.close()
        t2.close()

        frames1 = extract_frames(t1.name, step=slow_motion)
        frames2 = extract_frames(t2.name, step=slow_motion)

        keypoints1 = extract_keypoints(frames1)
        keypoints2 = extract_keypoints(frames2)

        cosine_sim = compute_cosine_similarity(keypoints1, keypoints2)
        dtw_dist = compute_dtw_similarity(keypoints1, keypoints2)

        # Save annotated video2
        annotated_path = "annotated_output.mp4"  # Using a fixed name
        create_annotated_video(frames2, annotated_path)

    st.success(f"Cosine Similarity Score: {cosine_sim:.4f}")
    st.success(f"DTW Distance (lower is more similar): {dtw_dist:.4f}")

    st.subheader("🎥 Video Comparison (Side-by-Side Playback)")

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Original Reference Video**")
        st.video(t1.name)
    with col6:
        st.markdown("**Overlayed Comparison Video**")
        st.video(annotated_path)

    st.subheader("📈 Trajectory Comparison of Nose Keypoint")
    fig, ax = plt.subplots()
    nose1 = [kp[:3] for kp in keypoints1]
    nose2 = [kp[:3] for kp in keypoints2]
    plot_trajectory(nose1, "Video 1")
    plot_trajectory(nose2, "Video 2")
    st.pyplot(fig)

    # Clean up
    os.remove(t1.name)
    os.remove(t2.name)
    if os.path.exists(annotated_path):
        os.remove(annotated_path)
else:
    st.info("Please upload two videos to begin.")