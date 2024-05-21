from PIL import Image

def merge_images(image1_path, image2_path, image3_path):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    image3 = Image.open(image3_path)

    # Ensure the images have the same height
    max_height = max(image1.height, image2.height, image3.height)
    image1 = image1.resize((image1.width, max_height))
    image2 = image2.resize((image2.width, max_height))
    image3 = image3.resize((image3.width, max_height))

    # Merge the images
    merged_image = Image.new('RGB', (image1.width + image2.width + image3.width, max_height))
    merged_image.paste(image1, (0, 0))
    merged_image.paste(image2, (image1.width, 0))
    merged_image.paste(image3, (image1.width + image2.width, 0))

    # Change all (0,0,0) and (255,255,255) pixels to (0,0,0)
    # But only change the pixels when all 3 channels are 0 or 255

    for x in range(merged_image.width):
        for y in range(merged_image.height):
            r, g, b = merged_image.getpixel((x, y))
            if (r, g, b) == (255, 255, 255):
                merged_image.putpixel((x, y), (0, 0, 0))

    return merged_image

if __name__ == '__main__':
    # report/images/my_model_5/0403.png

    merged_image = merge_images('report/images/my_model_5/0403.png', 'report/images/my_model_5/0407.png', 'report/images/my_model_5/0409.png')
    merged_image.save('report/images/my_model_5/merged_image.png')

    # report/images/my_model_20/0403.png

    merged_image = merge_images('report/images/my_model_20/0403.png', 'report/images/my_model_20/0407.png', 'report/images/my_model_20/0409.png')
    merged_image.save('report/images/my_model_20/merged_image.png')

    # report/images/my_model_50/0403.png

    merged_image = merge_images('report/images/my_model_50/0403.png', 'report/images/my_model_50/0407.png', 'report/images/my_model_50/0409.png')
    merged_image.save('report/images/my_model_50/merged_image.png')

    # report/images/pretrained_5/0403.png

    merged_image = merge_images('report/images/pretrained_5/0403.png', 'report/images/pretrained_5/0407.png', 'report/images/pretrained_5/0409.png')
    merged_image.save('report/images/pretrained_5/merged_image.png')

    # report/images/pretrained_20/0403.png

    merged_image = merge_images('report/images/pretrained_20/0403.png', 'report/images/pretrained_20/0407.png', 'report/images/pretrained_20/0409.png')
    merged_image.save('report/images/pretrained_20/merged_image.png')

    # report/images/pretrained_50/0403.png

    merged_image = merge_images('report/images/pretrained_50/0403.png', 'report/images/pretrained_50/0407.png', 'report/images/pretrained_50/0409.png')
    merged_image.save('report/images/pretrained_50/merged_image.png')

    # report/images/scratch_5/0403.png

    merged_image = merge_images('report/images/scratch_5/0403.png', 'report/images/scratch_5/0407.png', 'report/images/scratch_5/0409.png')
    merged_image.save('report/images/scratch_5/merged_image.png')

    # report/images/scratch_20/0403.png

    merged_image = merge_images('report/images/scratch_20/0403.png', 'report/images/scratch_20/0407.png', 'report/images/scratch_20/0409.png')
    merged_image.save('report/images/scratch_20/merged_image.png')

    # report/images/scratch_50/0403.png

    merged_image = merge_images('report/images/scratch_50/0403.png', 'report/images/scratch_50/0407.png', 'report/images/scratch_50/0409.png')
    merged_image.save('report/images/scratch_50/merged_image.png')
