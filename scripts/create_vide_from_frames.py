import fire
import cv2
import os

def convert_images_to_video(
        image_folder: str = "data/toy_task/multi_frame_rgbd_fitting/222200037/images",
        video_name: str = "output.mp4",
        frame_rate: int = 5
):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    fire.Fire(convert_images_to_video)
