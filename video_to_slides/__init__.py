from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Deque
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image
from PIL import ImageChops
from PIL import ImageFilter
from rapidfuzz import fuzz

try:
    from tqdm import tqdm

    DEFAULT_SHOW_PROGRESS_BAR: bool = True
    has_tqdm = True
except ImportError:
    DEFAULT_SHOW_PROGRESS_BAR = False
    has_tqdm = False

DEFAULT_VISUAL_DIFF_THRESHOLD: float = 0.05
DEFAULT_VISUAL_DIFF_CHECK_TEXT_THRESHOLD: float = 0.01
DEFAULT_TEXT_DIFF_THRESHOLD: float = 0.3
DEFAULT_VISUAL_DIFF_BLUR_RADIUS: int = 2
DEFAULT_TEXT_LANG: str = "eng"
DEFAULT_EVERY_NTH_FRAME: int = 30


PathOrStr = Union[Path, str]


try:
    import click


    @click.command()
    @click.option(
        "--visual-threshold",
        "visual_diff_threshold",
        type=float,
        default=DEFAULT_VISUAL_DIFF_THRESHOLD,
        help=(
                "2 images with an average grayscale pixel difference under this threshold "
                "will be considered to possibly belong to the same slide."
                "(See '--visual-check-text-threshold' and for more details)"
        ),
    )
    @click.option(
        "--visual-check-text-threshold",
        "visual_diff_check_text_threshold",
        type=float,
        default=DEFAULT_VISUAL_DIFF_CHECK_TEXT_THRESHOLD,
        help=(
                "2 images with an average grayscale pixel difference greater or equal to this threshold "
                "and under '--visual-threshold' will compare their text's similarity to determine "
                "whether they belong to the same slide."
        )
    )
    @click.option(
        "--text-threshold",
        "text_diff_threshold",
        type=float,
        default=DEFAULT_TEXT_DIFF_THRESHOLD,
        help=(
                "2 images with an average grayscale pixel difference less than '--visual-threshold' "
                "and greater or equal to 'visual-threshold' will be considered to belong to the same slide "
                "if the text difference is below this threshold. "
                "The text difference is: '1 - normalized Levenshtein distance'."
        ),
    )
    @click.option(
        "--visual-diff-blur-radius",
        type=int,
        default=DEFAULT_VISUAL_DIFF_BLUR_RADIUS,
        help=(
                "Blur radius of the gaussian blur, applied before "
                "the average grayscale pixel difference is calculated."
        ),
    )
    @click.option(
        "--text-lang",
        "text_diff_lang",
        type=str,
        default=DEFAULT_TEXT_LANG,
        help="Language(s) used for tesseract OCR.",
    )
    @click.option(
        "--every-nth-frame",
        type=int,
        default=DEFAULT_EVERY_NTH_FRAME,
        help="Determine whether a new slide has appeared comparing every nth video frame."
    )
    @click.option(
        "--progress/--hide-progress",
        "show_progress_bar",
        default=DEFAULT_SHOW_PROGRESS_BAR,
        help="Show or hide progress bar (requires 'tqdm' python package)."
    )
    @click.option(
        "--display-video/--dont-display-video",
        default=False,
        help="Display every nth frame while video is being processed.",
    )
    @click.argument(
        "video_path",
        type=click.Path(),
    )
    @click.argument(
        "pdf_dst_path",
        type=click.Path(),
        required=False,
    )
    def cli(
            visual_diff_threshold: float,
            visual_diff_check_text_threshold: float,
            text_diff_threshold: float,
            visual_diff_blur_radius: int,
            text_diff_lang: str,
            every_nth_frame: int,
            show_progress_bar: bool,
            display_video: bool,
            video_path: PathOrStr,
            pdf_dst_path: PathOrStr,
    ) -> None:
        """Convert a video of a presentation to a pdf of the slides."""
        convert_video_to_pdf_slides(
            video_path,
            pdf_dst_path=pdf_dst_path,
            visual_diff_threshold=visual_diff_threshold,
            visual_diff_check_text_threshold=visual_diff_check_text_threshold,
            text_diff_threshold=text_diff_threshold,
            visual_diff_blur_radius=visual_diff_blur_radius,
            text_diff_lang=text_diff_lang,
            every_nth_frame=every_nth_frame,
            show_progress_bar=show_progress_bar,
            display_video=display_video,
        )

except ImportError as click_import_err:

    def cli():
        raise ImportError("Package 'click' must be installed to use command line interface!")


def convert_video_to_pdf_slides(
        video_path: PathOrStr,
        pdf_dst_path: Optional[PathOrStr] = None,
        show_progress_bar: bool = DEFAULT_SHOW_PROGRESS_BAR, **kwargs
) -> None:
    """Convert a video of a presentation to a pdf of the slides."""
    if pdf_dst_path is None:
        pdf_dst_path = get_pdf_dst_path_from_video_path(video_path)

    progress_bar = None

    if show_progress_bar:
        if not has_tqdm:
            raise ValueError("Package 'tqdm' must be installed to use progress bar!")
        try:
            progress_bar = tqdm()
        except:
            raise Warning("Could not create progress bar!")

    with tmp_dir(f".tmp_slides_dir_{Path(pdf_dst_path).name[:-len('.pdf')]}") as tmp_dir_path:
        pdf_paths = []

        for i, image in enumerate(get_slide_images_from_video(
                video_path,
                progress_bar=progress_bar,
                **kwargs
        )):
            if progress_bar is not None:
                progress_bar.set_description(f"Slide {i + 1}")
            pdf_path = tmp_dir_path / f"slide{i + 1}.pdf"
            convert_image_to_ocr_pdf(image, pdf_path)

            pdf_paths.append(pdf_path)

        merge_pdfs(pdf_paths, pdf_dst_path)

    if progress_bar is not None:
        progress_bar.close()


def get_pdf_dst_path_from_video_path(video_path: PathOrStr) -> str:
    video_path = Path(video_path)
    return f"{video_path.stem}.pdf"


def get_slide_images_from_video(
        video_path: PathOrStr,
        compare_max_last_slides_n: int = 4,
        progress_bar: Optional[tqdm] = None, **kwargs
) -> Iterator[Image]:
    old_slide_images: Deque[Image] = deque(maxlen=compare_max_last_slides_n)
    for image in get_images_from_video(video_path, progress_bar=progress_bar, **kwargs):
        if is_image_new_slide(old_slide_images, image, **kwargs):
            yield image
            old_slide_images.append(image)


def is_image_new_slide(
        old_slide_images: Deque[Image],
        new_image: Image,
        visual_diff_threshold: float = DEFAULT_VISUAL_DIFF_THRESHOLD,
        visual_diff_check_text_threshold: float = DEFAULT_VISUAL_DIFF_CHECK_TEXT_THRESHOLD,
        text_diff_threshold: float = DEFAULT_TEXT_DIFF_THRESHOLD,
        **kwargs
) -> bool:
    for old_slide_image in old_slide_images:
        visual_delta = get_images_avg_pixel_diff(old_slide_image, new_image, **kwargs)
        if visual_delta >= visual_diff_threshold:
            continue

        # If images are very visually similar, do not check if text is also similar
        if visual_delta < visual_diff_check_text_threshold:
            return False

        if get_images_text_diff(old_slide_image, new_image, **kwargs) < text_diff_threshold:
            return False

    return True


def get_images_text_diff(image1: Image, image2: Image, **kwargs) -> float:
    image_string1 = get_normalized_image_string(image1, **kwargs)
    image_string2 = get_normalized_image_string(image2, **kwargs)

    return 1 - fuzz.ratio(image_string1, image_string2) / 100


def get_normalized_image_string(image: Image, text_diff_lang: str = DEFAULT_TEXT_LANG, **kwargs) -> str:
    image_string = pytesseract.image_to_string(image, lang=text_diff_lang)

    return normalize_image_string(image_string)


def normalize_image_string(image_string: str) -> str:
    return "".join(char for char in image_string.lower() if char.isalpha())


def get_images_avg_pixel_diff(
        image1: Image,
        image2: Image,
        visual_diff_blur_radius: int = DEFAULT_VISUAL_DIFF_BLUR_RADIUS,
        **kwargs
) -> float:
    # Blur images so noise from video has less of an effect
    blurred_image1 = blur_image(image1, radius=visual_diff_blur_radius)
    blurred_image2 = blur_image(image2, radius=visual_diff_blur_radius)

    diff = ImageChops.difference(blurred_image1, blurred_image2)
    gray_diff = diff.convert("LA")
    return abs(127 - np.array(gray_diff).mean()) / 128


def blur_image(image: Image, radius: int) -> Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def get_images_from_video(
        video_path: PathOrStr,
        every_nth_frame: int = DEFAULT_EVERY_NTH_FRAME,
        display_video: bool = False,
        progress_bar: Optional[tqdm] = None,
        **kwargs
) -> Iterator[Image]:
    cap = cv2.VideoCapture(video_path)

    if progress_bar is not None:
        progress_bar.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if cap.isOpened() is False:
        raise RuntimeError("Error opening video  file")

    frame_n = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            if display_video:
                cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            yield convert_frame_to_image(frame)

            frame_n += every_nth_frame
            cap.set(1, frame_n)

            if progress_bar is not None:
                try:
                    progress_bar.update(every_nth_frame)
                except:
                    raise Warning("Could not update progress bar!")
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def convert_frame_to_image(frame) -> Image:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def convert_image_to_ocr_pdf(image: Image, name: PathOrStr) -> None:
    pdf = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')
    with open(name, 'w+b') as f:
        f.write(pdf)


@contextmanager
def tmp_dir(path: PathOrStr) -> Iterator[Path]:
    try:
        yield create_dir(path)
    finally:
        remove_dir(path)


def create_dir(path: PathOrStr) -> Path:
    path = Path(path)
    path.mkdir(parents=True)
    return path


def remove_dir(path: PathOrStr) -> None:
    path = Path(path)
    for child in path.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            remove_dir(child)
    path.rmdir()


def merge_pdfs(pdf_paths: List[Path], dst_path: PathOrStr) -> None:
    merge_pdf = None
    try:
        merged_pdf = fitz.open()

        for pdf in pdf_paths:
            with fitz.open(pdf) as pdf_file:
                merged_pdf.insert_pdf(pdf_file)

        merged_pdf.save(dst_path)
    finally:
        if merge_pdf is not None:
            merged_pdf.close()
