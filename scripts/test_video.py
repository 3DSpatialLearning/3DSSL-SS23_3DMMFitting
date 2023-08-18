import cv2

from pathlib import Path

deca_output_folder_path = './data/deca/subject_0'
deca_output_folder = Path(deca_output_folder_path)
deca_output_images = sorted(deca_output_folder.glob('*.png'))

custom_render_folder_path = 'output/rendered_images/2023-07-19_01:47:09'
custom_render_blended_images = sorted(Path(custom_render_folder_path).glob('blended_*.png'), key=lambda x: int(x.stem.split("_")[1].split(".")[0]))
custom_render_input_images = sorted(Path(custom_render_folder_path).glob('input_*.png'), key=lambda x: int(x.stem.split("_")[2].split(".")[0]))

# get first image to get dimensions
first_image_custom = cv2.imread(str(custom_render_input_images[0]))
height, width, _ = first_image_custom.shape

# create video writer
video = cv2.VideoWriter("compare_deca.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 12, (3 * width, height))

for deca_output, custom_blended, input_color in zip(deca_output_images, custom_render_blended_images, custom_render_input_images):
    print(deca_output.stem)
    print(custom_blended.stem)
    print(input_color.stem)
    deca_output = cv2.imread(str(deca_output))
    deca_output = cv2.resize(deca_output, (width, height))
    custom_blended = cv2.imread(str(custom_blended))
    input_color = cv2.imread(str(input_color))

    combined_img = cv2.hconcat([input_color, deca_output, custom_blended])
    video.write(combined_img)

cv2.destroyAllWindows()
video.release()


