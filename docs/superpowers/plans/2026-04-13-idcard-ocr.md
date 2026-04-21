# 身份证 OCR 识别系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a command-line Python script that takes an ID card photo, preprocesses it with OpenCV, crops field regions, and uses PaddleOCR to recognize each field individually.

**Architecture:** Single-file script (`idcard_ocr.py`) with 4 sequential stages: image preprocessing → ID card detection & perspective correction → region cropping → per-region OCR. OpenCV handles all image manipulation; PaddleOCR handles text recognition. A `requirements.txt` lists dependencies.

**Tech Stack:** Python 3.10, PaddleOCR, OpenCV, NumPy

---

## File Structure

```
face/
├── idcard.jpg              # Existing test image
├── idcard_ocr.py           # Main script (single file, all logic)
└── requirements.txt        # pip dependencies
```

`idcard_ocr.py` contains all functions:
- `preprocess(image)` — grayscale, blur, edge detection
- `find_and_warp_card(image, edged)` — contour detection, perspective transform, resize to 640x400
- `crop_regions(card)` — crop 6 field regions from the normalized card image
- `ocr_regions(regions, ocr_engine)` — run PaddleOCR on each region, return dict of results
- `main()` — CLI entry point, orchestrates the pipeline and prints results

---

### Task 1: Create requirements.txt and install dependencies

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```txt
paddlepaddle
paddleocr
opencv-python
numpy
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully. PaddleOCR will download its detection/recognition models on first use (~100MB).

- [ ] **Step 3: Verify installation**

Run: `python -c "import cv2; import paddleocr; import numpy; print('OK')"`
Expected: Prints `OK` with no errors.

---

### Task 2: Implement image preprocessing

**Files:**
- Create: `idcard_ocr.py`

- [ ] **Step 1: Create idcard_ocr.py with preprocess function and a temporary main for visual verification**

```python
import sys
import cv2
import numpy as np


def preprocess(image):
    """Convert image to edged binary for contour detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    return edged


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python idcard_ocr.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot read image '{image_path}'")
        sys.exit(1)

    edged = preprocess(image)
    cv2.imwrite("debug_edged.jpg", edged)
    print("Saved debug_edged.jpg - check that card edges are visible")
```

- [ ] **Step 2: Run and verify preprocessing**

Run: `python idcard_ocr.py idcard.jpg`
Expected: Prints message about debug_edged.jpg. Open the file to confirm the card edges are clearly visible as white lines on black background. The rectangular outline of the ID card should be the dominant shape.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt idcard_ocr.py
git commit -m "feat: add image preprocessing with edge detection"
```

---

### Task 3: Implement ID card detection and perspective correction

**Files:**
- Modify: `idcard_ocr.py`

- [ ] **Step 1: Add order_points and find_and_warp_card functions after preprocess**

```python
def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


CARD_WIDTH = 640
CARD_HEIGHT = 400


def find_and_warp_card(image, edged):
    """Find the ID card contour and warp to a standard 640x400 rectangle.

    If no quadrilateral contour is found, assume the image is already
    a cropped card and just resize it.
    """
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            card_contour = approx
            break

    if card_contour is not None:
        pts = card_contour.reshape(4, 2).astype("float32")
        rect = order_points(pts)
        dst = np.array(
            [[0, 0], [CARD_WIDTH, 0], [CARD_WIDTH, CARD_HEIGHT], [0, CARD_HEIGHT]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (CARD_WIDTH, CARD_HEIGHT))
    else:
        print("Warning: card contour not found, resizing original image")
        warped = cv2.resize(image, (CARD_WIDTH, CARD_HEIGHT))

    return warped
```

- [ ] **Step 2: Update the main block to use the new function**

Replace the existing `if __name__ == "__main__":` block with:

```python
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python idcard_ocr.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot read image '{image_path}'")
        sys.exit(1)

    edged = preprocess(image)
    card = find_and_warp_card(image, edged)
    cv2.imwrite("debug_card.jpg", card)
    print("Saved debug_card.jpg - check that card is properly rectified to 640x400")
```

- [ ] **Step 3: Run and verify**

Run: `python idcard_ocr.py idcard.jpg`
Expected: Prints message about debug_card.jpg. Open it to verify the ID card is rectified into a clean 640x400 rectangle with the text horizontal and readable.

- [ ] **Step 4: Commit**

```bash
git add idcard_ocr.py
git commit -m "feat: add ID card contour detection and perspective correction"
```

---

### Task 4: Implement region cropping

**Files:**
- Modify: `idcard_ocr.py`

- [ ] **Step 1: Add REGIONS dict and crop_regions function after find_and_warp_card**

```python
# Each region is defined as (x_ratio, y_ratio, w_ratio, h_ratio) relative to card size.
# These values are tuned for a standard Chinese ID card front side.
REGIONS = {
    "姓名": (0.15, 0.06, 0.35, 0.13),
    "性别": (0.15, 0.21, 0.10, 0.10),
    "民族": (0.35, 0.21, 0.15, 0.10),
    "出生日期": (0.15, 0.33, 0.45, 0.11),
    "住址": (0.15, 0.47, 0.45, 0.22),
    "身份证号": (0.08, 0.77, 0.85, 0.15),
}


def crop_regions(card):
    """Crop field regions from the normalized card image."""
    h, w = card.shape[:2]
    crops = {}
    for name, (rx, ry, rw, rh) in REGIONS.items():
        x = int(rx * w)
        y = int(ry * h)
        cw = int(rw * w)
        ch = int(rh * h)
        crops[name] = card[y : y + ch, x : x + cw]
    return crops
```

- [ ] **Step 2: Update main block to save cropped regions for visual verification**

Replace the `if __name__ == "__main__":` block with:

```python
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python idcard_ocr.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot read image '{image_path}'")
        sys.exit(1)

    edged = preprocess(image)
    card = find_and_warp_card(image, edged)
    regions = crop_regions(card)

    for name, crop in regions.items():
        filename = f"debug_region_{name}.jpg"
        cv2.imwrite(filename, crop)
        print(f"Saved {filename}")

    print("Check that each region image contains the correct field content")
```

- [ ] **Step 3: Run and verify all 6 regions**

Run: `python idcard_ocr.py idcard.jpg`
Expected: 6 debug images saved. Open each one and verify:
- `debug_region_姓名.jpg` — contains the name text
- `debug_region_性别.jpg` — contains gender text
- `debug_region_民族.jpg` — contains ethnicity text
- `debug_region_出生日期.jpg` — contains birth date text
- `debug_region_住址.jpg` — contains full address text (possibly 2 lines)
- `debug_region_身份证号.jpg` — contains the 18-digit ID number

If any region is misaligned, adjust the corresponding ratio values in the `REGIONS` dict and re-run until all 6 are correct.

- [ ] **Step 4: Commit**

```bash
git add idcard_ocr.py
git commit -m "feat: add region cropping for 6 ID card fields"
```

---

### Task 5: Implement per-region OCR and final output

**Files:**
- Modify: `idcard_ocr.py`

- [ ] **Step 1: Add PaddleOCR import at top of file**

Add after the existing imports:

```python
from paddleocr import PaddleOCR
```

- [ ] **Step 2: Add ocr_regions function after crop_regions**

```python
def ocr_regions(regions, ocr_engine):
    """Run OCR on each cropped region and return a dict of field name -> text."""
    results = {}
    for name, crop in regions.items():
        # For ID number region, apply binary thresholding to improve recognition
        if name == "身份证号":
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        ocr_result = ocr_engine.ocr(crop, cls=True)
        texts = []
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                texts.append(line[1][0])
        results[name] = "".join(texts)
    return results
```

- [ ] **Step 3: Replace the main block with the final version**

Replace the `if __name__ == "__main__":` block with:

```python
def main():
    if len(sys.argv) < 2:
        print("Usage: python idcard_ocr.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot read image '{image_path}'")
        sys.exit(1)

    # Stage 1: Preprocess
    edged = preprocess(image)

    # Stage 2: Detect and warp card
    card = find_and_warp_card(image, edged)

    # Stage 3: Crop regions
    regions = crop_regions(card)

    # Stage 4: OCR each region
    ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    results = ocr_regions(regions, ocr_engine)

    # Output
    print("========== 身份证识别结果 ==========")
    print(f"姓名：{results.get('姓名', '')}")
    print(f"性别：{results.get('性别', '')}")
    print(f"民族：{results.get('民族', '')}")
    print(f"出生日期：{results.get('出生日期', '')}")
    print(f"住址：{results.get('住址', '')}")
    print(f"身份证号：{results.get('身份证号', '')}")
    print("=" * 36)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run end-to-end test**

Run: `python idcard_ocr.py idcard.jpg`
Expected output (approximately):
```
========== 身份证识别结果 ==========
姓名：朱礼涛
性别：男
民族：汉
出生日期：1987年2月8日
住址：湖北省蕲春县赤东镇田围村4组
身份证号：421126198702080813
====================================
```

If any field is wrong or empty, check the corresponding debug region image from Task 4. If the region is correct but OCR fails, try adjusting the crop slightly to give more padding around the text.

- [ ] **Step 5: Clean up debug files**

Delete all debug images generated during development:
```bash
rm -f debug_edged.jpg debug_card.jpg debug_region_*.jpg
```

- [ ] **Step 6: Commit**

```bash
git add idcard_ocr.py
git commit -m "feat: add per-region OCR and structured output"
```

---

### Task 6: Final verification and cleanup

**Files:**
- Review: `idcard_ocr.py`

- [ ] **Step 1: Run a clean end-to-end test from scratch**

Run: `python idcard_ocr.py idcard.jpg`
Expected: Clean structured output with all 6 fields recognized correctly. No debug files created, no warnings (except the contour warning if applicable).

- [ ] **Step 2: Test error handling**

Run with no arguments:
```bash
python idcard_ocr.py
```
Expected: `Usage: python idcard_ocr.py <image_path>` then exit.

Run with non-existent file:
```bash
python idcard_ocr.py nonexistent.jpg
```
Expected: `Error: cannot read image 'nonexistent.jpg'` then exit.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete ID card OCR recognition system"
```
