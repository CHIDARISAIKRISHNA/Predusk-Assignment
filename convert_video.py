from moviepy import VideoFileClip

input_path = "outputs/Output_Video.mp4"
output_path = "outputs/Output_Video_web.mp4"

clip = VideoFileClip(input_path)
clip.write_videofile(output_path, codec="libx264", audio=False)
clip.close()

print("Converted successfully:", output_path)