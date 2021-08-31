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

Type `video-to-slides --help` for full list of options.