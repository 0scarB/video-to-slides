# video-to-slides

Convert a video of a presentation to a pdf of slides.

## Installation

1. Install `tesseract-ocr`
   - Docs https://tesseract-ocr.github.io/tessdoc/Home.html
   - Ubuntu PPA: https://launchpad.net/~alex-p/+archive/ubuntu/tesseract-ocr-devel
   - Debian: https://notesalexp.org/tesseract-ocr/#tesseract_5.x
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
3. `pip install video-to-slides`

## Basic Usage

### Command line

`video-to-slides path/to/video.mp4 path/to/pdf.pdf` will create a pdf with slides from the video.

#### Full help message:
```
Usage: video-to-slides [OPTIONS] VIDEO_PATH [PDF_DST_PATH]

  Convert a video of a presentation to a pdf of the slides.

Options:
  --visual-threshold FLOAT        2 images with an average grayscale pixel
                                  difference under this threshold will be
                                  considered to possibly belong to the same
                                  slide.(See '--visual-check-text-threshold'
                                  and for more details)
  --visual-check-text-threshold FLOAT
                                  2 images with an average grayscale pixel
                                  difference greater or equal to this
                                  threshold and under '--visual-threshold'
                                  will compare their text's similarity to
                                  determine whether they belong to the same
                                  slide.
  --text-threshold FLOAT          2 images with an average grayscale pixel
                                  difference less than '--visual-threshold'
                                  and greater or equal to 'visual-threshold'
                                  will be considered to belong to the same
                                  slide if the text difference is below this
                                  threshold. The text difference is: '1 -
                                  normalized Levenshtein distance'.
  --visual-diff-blur-radius INTEGER
                                  Blur radius of the gaussian blur, applied
                                  before the average grayscale pixel
                                  difference is calculated.
  --text-lang TEXT                Language(s) used for tesseract OCR.
  --every-nth-frame INTEGER       Determine whether a new slide has appeared
                                  comparing every nth video frame.
  --progress / --hide-progress    Show or hide progress bar (requires 'tqdm'
                                  python package).
  --display-video / --dont-display-video
                                  Display every nth frame while video is being
                                  processed.
  --help                          Show this message and exit.

```
