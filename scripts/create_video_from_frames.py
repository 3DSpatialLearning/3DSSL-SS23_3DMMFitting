import fire
import cv2
import os


def create_video_from_frames(
        image_folder: str = "../output/",
        video_name: str = "../output/output.mp4",
        frame_rate: int = 15
):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

    print("Video created saved in", video_name)


if __name__ == '__main__':
    fire.Fire(create_video_from_frames)
