import fire
import cv2
import os


def convert_images_to_video(
        image_folder: str = "output",
        video_name: str = "combined.mp4",
        frame_rate: int = 24
):
    blended_images = [img for img in os.listdir(image_folder) if img.startswith("blended")]
    input_images = [img for img in os.listdir(image_folder) if img.startswith("input")]
    rendered_images = [img for img in os.listdir(image_folder) if img.startswith("rendered")]

    blended_images = sorted(blended_images, key=lambda x: int(x.split("_")[1].split(".")[0]))
    input_images = sorted(input_images, key=lambda x: int(x.split("_")[1].split(".")[0]))
    rendered_images = sorted(rendered_images, key=lambda x: int(x.split("_")[1].split(".")[0]))

    frame = cv2.imread(os.path.join(image_folder, input_images[0]))
    height, width, _ = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (3 * width, height))

    for blended_img, input_img, rendered_img in zip(blended_images, input_images, rendered_images):
        bl_img = cv2.imread(os.path.join(image_folder, blended_img))
        in_img = cv2.imread(os.path.join(image_folder, input_img))
        re_img = cv2.imread(os.path.join(image_folder, rendered_img))

        combined_img = cv2.hconcat([bl_img, in_img, re_img])
        video.write(combined_img)

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    fire.Fire(convert_images_to_video)
