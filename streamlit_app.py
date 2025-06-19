import streamlit as st
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import mediapipe as mp
from matplotlib.animation import FuncAnimation
from io import BytesIO
import base64
import datetime

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# === Pose Extraction and Visualization ===
def extract_keypoints_with_visualization(video_path, max_frames=None):
    """
    Extracts pose keypoints and draws annotated landmarks on each frame

    Args:
        video_path: Path to video file
        max_frames: Optional max number of frames to process

    Returns:
        tuple: (keypoints as ndarray, list of annotated RGB frames)
    """
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(video_path)
    keypoints = []
    annotated_frames = []

    while cap.isOpened() and (max_frames is None or len(keypoints) < max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        annotated_frame = frame_rgb.copy()
        if results.pose_landmarks:
            kps = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            keypoints.append(kps)

            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        annotated_frames.append(annotated_frame)

    cap.release()
    pose.close()

    return np.array(keypoints), annotated_frames

# === Fallback HTML video animation using image frames ===
def frames_to_html_video(frames, frame_rate=30):
    """
    Converts a list of RGB frames to HTML5 video for inline display

    Args:
        frames: List of frames
        frame_rate: Frame playback rate

    Returns:
        HTML string of video player
    """
    if not frames:
        return "<p>No frames to display</p>"

    height, width, _ = frames[0].shape
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    plt.axis('off')
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / frame_rate, blit=True)

    # Workaround for environments without ffmpeg
    try:
        video_html = ani.to_html5_video()
    except RuntimeError:
        video_html = "<p><b>Video rendering not supported (missing ffmpeg).</b></p>"

    plt.close()
    return video_html

# === Core Comparison Logic ===
def compare_videos_with_visualization(ref_path, stu_path, max_frames=100):
    """
    Compares two videos using pose keypoints and DTW

    Args:
        ref_path: Path to reference video
        stu_path: Path to student video
        max_frames: Max frames to compare

    Returns:
        dict: Comparison results and plots
    """
    ref_kps, ref_frames = extract_keypoints_with_visualization(ref_path, max_frames)
    stu_kps, stu_frames = extract_keypoints_with_visualization(stu_path, max_frames)

    results = {
        'ref_frames': ref_frames,
        'stu_frames': stu_frames
    }

    if len(ref_kps) > 0 and len(stu_kps) > 0:
        distance, path = fastdtw(
            ref_kps.reshape(len(ref_kps), -1),
            stu_kps.reshape(len(stu_kps), -1),
            dist=euclidean
        )
        normalized_distance = distance / max(len(ref_kps), len(stu_kps))

        # Create summary comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        path = np.array(path)
        ax1.plot(path[:, 0], path[:, 1], 'b-', alpha=0.5)
        ax1.set(xlabel='Reference Frames', ylabel='Student Frames', title='DTW Alignment Path')
        ax1.grid(True)

        kp_diff = np.mean(np.abs(ref_kps.mean(0) - stu_kps.mean(0)), axis=0)
        ax2.bar(['X', 'Y', 'Z'], kp_diff)
        ax2.set(title='Average Keypoint Differences', ylabel='Mean Absolute Difference')

        plt.tight_layout()
        st.pyplot(fig)

        results.update({
            'dtw_distance': distance,
            'normalized_distance': normalized_distance,
            'keypoint_diff': kp_diff
        })

    return results

# === Streamlit App Layout ===
st.set_page_config(layout="wide")
st.title("📊 Pose Comparison Tool")

with st.sidebar:
    st.header("Upload Videos")
    ref_file = st.file_uploader("Upload Reference Video", type=["mp4", "mov"])
    stu_file = st.file_uploader("Upload Student Video", type=["mp4", "mov"])
    max_frames = st.slider("Max Frames to Process", min_value=10, max_value=500, value=200, step=10)

if ref_file and stu_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as ref_temp:
        ref_temp.write(ref_file.read())
        ref_path = ref_temp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as stu_temp:
        stu_temp.write(stu_file.read())
        stu_path = stu_temp.name

    st.success("Videos uploaded successfully! Starting comparison...")

    results = compare_videos_with_visualization(ref_path, stu_path, max_frames=max_frames)

    st.subheader("🎥 Annotated Pose Videos")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Reference Video**")
        ref_html = frames_to_html_video(results['ref_frames'])
        st.components.v1.html(ref_html, height=360, width=320)

    with col2:
        st.markdown("**Student Video**")
        stu_html = frames_to_html_video(results['stu_frames'])
        st.components.v1.html(stu_html, height=360, width=320)

    st.success("Comparison Completed Successfully!")
    st.markdown(f"**DTW Distance:** {results['dtw_distance']:.2f}")
    st.markdown(f"**Normalized DTW Distance:** {results['normalized_distance']:.4f}")
    st.markdown(f"**Avg Keypoint Diff (XYZ):** {np.mean(results['keypoint_diff']):.4f}")

    # Display comparison report
    st.subheader("📋 Comparison Report")
    report_text = f"""
    ## Pose Comparison Summary Report

    - **Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
    - **Max Frames Processed**: {max_frames}
    - **DTW Distance**: {results['dtw_distance']:.2f}
    - **Normalized DTW Distance**: {results['normalized_distance']:.4f}
    - **Average Keypoint Difference (XYZ)**: {np.mean(results['keypoint_diff']):.4f}
    """
    st.markdown(report_text)

    # Report download button (currently disabled)
    # Uncomment to enable actual download functionality in future
    # buffer = BytesIO()
    # buffer.write(report_text.encode())
    # buffer.seek(0)
    # st.download_button("Download Report", buffer, file_name="pose_comparison_report.txt")

    if st.button("Download Report"):
        st.warning("🔒 Download functionality will be available soon.")

else:
    st.info("Please upload both a reference and student video to begin.")