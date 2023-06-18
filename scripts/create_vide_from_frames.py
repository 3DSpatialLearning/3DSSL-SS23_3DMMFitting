import cv2
import os

def convert_images_to_video(image_folder, video_name, frame_rate):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

# Provide the folder path containing the RGB images
image_folder = "./output/rendered"

# Provide the name of the output video file
video_name = "output_rendered.mp4"

# Set the frame rate (e.g., 5 frames per second)
frame_rate = 5

# Call the conversion function
convert_images_to_video(image_folder, video_name, frame_rate)