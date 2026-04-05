import streamlit as st
import pandas as pd

st.title("Multi-Object Detection and Persistent ID Tracking")
st.write("Demo of sports/event multi-object tracking project.")

st.subheader("Project Links")
st.markdown("[GitHub Repository](https://github.com/CHIDARISAIKRISHNA/Predusk-Assignment)")
st.subheader("Output Video")
video_file = open("outputs/Output_Video.mp4", "rb")
video_bytes = video_file.read()
st.video(video_bytes)

st.subheader("Tracking Statistics")
df = pd.read_csv("outputs/track_stats.csv")
st.dataframe(df)

st.subheader("Sample Screenshots")
for img in [
    "outputs/screenshots/frame_000_idx0.jpg",
    "outputs/screenshots/frame_001_idx125.jpg",
    "outputs/screenshots/frame_002_idx251.jpg",
    "outputs/screenshots/frame_003_idx376.jpg",
    "outputs/screenshots/frame_004_idx502.jpg",
    "outputs/screenshots/frame_005_idx628.jpg",
]:
    st.image(img, use_container_width=True)