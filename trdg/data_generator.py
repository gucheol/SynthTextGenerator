import os
import random as rnd
from urllib import parse

from PIL import Image, ImageFilter

from trdg import computer_text_generator, background_generator, distorsion_generator

try:
    from trdg import handwritten_text_generator
except ImportError as e:
    print("Missing modules for handwritten text generation.")


def get_font_name(path):
    head, tail = os.path.split(path)
    return tail[:-3]


def encode2_safe_file_name(file_name):
    conversion = {'*': '__Star__',
                  '\\': '__BackSlash__',
                  '/': '__Slash__',
                  ':': '__Colon__',
                  ';': '__SemiColon__',
                  '<': '__LeftBrace__',
                  '>': '__RightBrace__'

    }
    tmp_file_name = file_name[:-4]
    ext = file_name[-4:]
    new_file_name = ''
    for ch in tmp_file_name:
        if conversion.get(ch) is not None:
            ch = conversion[ch]
        new_file_name += ch

    return new_file_name + ext


def decode_safe_file_name(file_name):
    conversion = {'*': '__Star__',
                  '\\': '__BackSlash__',
                  '/': '__Slash__',
                  ':': '__Colon__',
                  ';': '__SemiColon__',
                  '<': '__LeftBrace__',
                  '>': '__RightBrace__'
                  }
    new_file_name = file_name[:-4]
    ext = file_name[-4:]
    for key, value in conversion.items():
        new_file_name = new_file_name.replace(value, key)
    return new_file_name + ext


class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
            Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    @classmethod
    def generate(
        cls,
        index,
        text,
        font,
        out_dir,
        size,
        extension,
        skewing_angle,
        random_skew,
        blur,
        random_blur,
        background_type,
        distorsion_type,
        distorsion_orientation,
        is_handwritten,
        name_format,
        width,
        alignment,
        text_color,
        orientation,
        space_width,
        character_spacing,
        margins,
        fit,
        output_mask,
        word_split,
    ):
        image = None

        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        ##########################
        # Create picture of text #
        ##########################
        if is_handwritten:
            if orientation == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            image, mask = handwritten_text_generator.generate(text, text_color)
        else:
            image, mask = computer_text_generator.generate(
                text,
                font,
                text_color,
                size,
                orientation,
                space_width,
                character_spacing,
                fit,
                word_split,
            )
        random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        rotated_img = image.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        rotated_mask = mask.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        #############################
        # Apply distorsion to image #
        #############################
        if distorsion_type == 0:
            distorted_img = rotated_img  # Mind = blown
            distorted_mask = rotated_mask
        elif distorsion_type == 1:
            distorted_img, distorted_mask = distorsion_generator.sin(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            distorted_img, distorted_mask = distorsion_generator.cos(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            distorted_img, distorted_mask = distorsion_generator.random(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            new_width = int(
                distorted_img.size[0]
                * (float(size - vertical_margin) / float(distorted_img.size[1]))
            )
            resized_img = distorted_img.resize(
                (new_width, size - vertical_margin), Image.ANTIALIAS
            )
            resized_mask = distorted_mask.resize((new_width, size - vertical_margin))
            background_width = width if width > 0 else new_width + horizontal_margin
            background_height = size
        # Vertical text
        elif orientation == 1:
            new_height = int(
                float(distorted_img.size[1])
                * (float(size - horizontal_margin) / float(distorted_img.size[0]))
            )
            resized_img = distorted_img.resize(
                (size - horizontal_margin, new_height), Image.ANTIALIAS
            )
            resized_mask = distorted_mask.resize(
                (size - horizontal_margin, new_height), Image.ANTIALIAS
            )
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background_img = background_generator.gaussian_noise(
                background_height, background_width
            )
        elif background_type == 1:
            background_img = background_generator.plain_white(
                background_height, background_width
            )
        elif background_type == 2:
            background_img = background_generator.quasicrystal(
                background_height, background_width
            )
        else:
            background_img = background_generator.picture(
                background_height, background_width
            )
        background_mask = Image.new(
            "RGB", (background_width, background_height), (0, 0, 0)
        )

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = resized_img.size

        if alignment == 0 or width == -1:
            background_img.paste(resized_img, (margin_left, margin_top), resized_img)
            background_mask.paste(resized_mask, (margin_left, margin_top))
        elif alignment == 1:
            background_img.paste(
                resized_img,
                (int(background_width / 2 - new_text_width / 2), margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (int(background_width / 2 - new_text_width / 2), margin_top),
            )
        else:
            background_img.paste(
                resized_img,
                (background_width - new_text_width - margin_right, margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (background_width - new_text_width - margin_right, margin_top),
            )

        ##################################
        # Apply gaussian blur #
        ##################################

        gaussian_filter = ImageFilter.GaussianBlur(
            radius=blur if not random_blur else rnd.randint(0, blur)
        )
        final_image = background_img.filter(gaussian_filter)
        final_mask = background_mask.filter(gaussian_filter)

        #####################################
        # Generate name for resulting image #
        #####################################
        if name_format == 0:
            image_name = "{}_{}.{}".format(text, str(index), extension)
            mask_name = "{}_{}_mask.png".format(text, str(index))
        elif name_format == 1:
            image_name = "{}_{}.{}".format(str(index), text, extension)
            mask_name = "{}_{}_mask.png".format(str(index), text)
        elif name_format == 2:
            image_name = "{}.{}".format(str(index), extension)
            mask_name = "{}_mask.png".format(str(index))
        else:
            print("{} is not a valid name format. Using default.".format(name_format))
            image_name = "{}_{}.{}".format(text, str(index), extension)
            mask_name = "{}_{}_mask.png".format(text, str(index))

        # Save the image
        font_name = get_font_name(font)
        safe_file_name = encode2_safe_file_name (image_name)
        if out_dir is not None:
            final_image.convert("RGB").save(os.path.join(out_dir, font_name + '-' + safe_file_name))
            if output_mask == 1:
                final_mask.convert("RGB").save(os.path.join(out_dir, mask_name))
        else:
            if output_mask == 1:
                return final_image.convert("RGB"), final_mask.convert("RGB")
            return final_image.convert("RGB")
