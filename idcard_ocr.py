import os
import re
import sys

import cv2
import numpy as np
from paddleocr import PaddleOCR


def _verify_id_checksum(id_str):
    """Verify Chinese ID card number using the standard checksum algorithm."""
    if len(id_str) != 18:
        return False
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_codes = "10X98765432"
    try:
        total = sum(int(id_str[i]) * weights[i] for i in range(17))
        return check_codes[total % 11] == id_str[17].upper()
    except (ValueError, IndexError):
        return False


def _order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_card(image):
    """Try to detect and crop the ID card from the image.

    Returns the cropped card image if a rectangle is found,
    otherwise returns the original image.
    """
    # Resize large images for more reliable contour detection
    h_orig, w_orig = image.shape[:2]
    scale = 1.0
    if max(h_orig, w_orig) > 1000:
        scale = 1000.0 / max(h_orig, w_orig)
        resized = cv2.resize(image, None, fx=scale, fy=scale)
    else:
        resized = image.copy()

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try multiple edge detection approaches
    candidates = []

    # Approach 1: Canny edge detection
    for low, high in [(30, 100), (50, 150), (20, 80)]:
        edged = cv2.Canny(blurred, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged = cv2.dilate(edged, kernel, iterations=2)
        edged = cv2.erode(edged, kernel, iterations=1)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(contours)

    # Approach 2: Adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(contours)

    # Approach 3: Saturation-based segmentation — removes colored backgrounds (red tables,
    # green mats, etc.) and isolates the white/light card area even when card edges are
    # partially occluded by adjacent light-colored objects.
    hsv_img = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    sat = hsv_img[:, :, 1]
    val = hsv_img[:, :, 2]
    # Card: low saturation (<70) AND reasonably bright (>80)
    fg_mask = cv2.bitwise_and(
        cv2.threshold(sat, 70, 255, cv2.THRESH_BINARY_INV)[1],
        cv2.threshold(val, 80, 255, cv2.THRESH_BINARY)[1],
    )
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k3, iterations=4)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k3, iterations=2)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(contours)

    # Sort all candidates by area (largest first)
    candidates = sorted(candidates, key=cv2.contourArea, reverse=True)

    img_area = resized.shape[0] * resized.shape[1]

    for c in candidates:
        area = cv2.contourArea(c)
        if area < img_area * 0.05 or area > img_area * 0.7:
            continue

        # Use minAreaRect for robustness (handles rounded corners)
        min_rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(min_rect)
        box = box.astype("float32")

        w = min_rect[1][0]
        h = min_rect[1][1]
        if w == 0 or h == 0:
            continue

        ratio = max(w, h) / min(w, h)
        if ratio < 1.2 or ratio > 2.2:
            continue

        # Also try approxPolyDP with different epsilon values
        peri = cv2.arcLength(c, True)
        found_quad = False
        for eps in [0.02, 0.03, 0.05, 0.08, 0.12]:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                box = approx.reshape(4, 2).astype("float32")
                found_quad = True
                break

        if not found_quad:
            # Use minAreaRect box points
            box = cv2.boxPoints(min_rect).astype("float32")

        # Scale points back to original image size
        box = box / scale
        rect = _order_points(box)

        rw = np.linalg.norm(rect[1] - rect[0])
        rh = np.linalg.norm(rect[3] - rect[0])

        # Ensure landscape orientation
        card_w = int(max(rw, rh))
        card_h = int(min(rw, rh))

        if card_w < 100 or card_h < 60:
            continue

        dst = np.array(
            [[0, 0], [card_w, 0], [card_w, card_h], [0, card_h]],
            dtype="float32",
        )

        # If detected as portrait, rotate points
        if rh > rw:
            rect = np.array([rect[3], rect[0], rect[1], rect[2]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (card_w, card_h))
        return warped

    # No card contour found, return original image
    return image


def enhance_image(image):
    """Enhance image quality for better OCR: sharpen, contrast, denoise."""
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Sharpen
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Convert back to BGR for PaddleOCR
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


ETHNICITIES = set(
    "汉蒙回藏维苗彝壮满侗瑶白土"
    "哈黎傣畲佤水怒京羌珞布朝纳"
    "拉傈僳德柯锡仡塔独鄂俄"
)
PROVINCES = [
    "北京", "天津", "上海", "重庆",
    "河北", "山西", "辽宁", "吉林",
    "黑龙江", "江苏", "浙江", "安徽",
    "福建", "江西", "山东", "河南",
    "湖北", "湖南", "广东", "海南",
    "四川", "贵州", "云南", "陕西",
    "甘肃", "青海", "台湾", "内蒙古",
    "广西", "西藏", "宁夏", "新疆",
    "香港", "澳门",
]
ADDR_KEYWORDS = set(
    "省市县区镇乡村路街道弄号"
    "室楼幢栋组屯坪社巷"
)
LABEL_KEYWORDS = (
    "姓名", "性别", "民族", "民旅",
    "出生", "号码", "公民", "身份",
)
LABEL_FRAGMENTS = set(
    "生别名住址族旅性码民公身"
)


def _classify_line(text):
    """Tag an OCR line with a semantic category. Returns (tag, clean, info)."""
    clean = text.strip().replace(" ", "")
    if not clean:
        return ("empty", "", {})

    m = re.search(r"(\d{17}[\dXx])", clean)
    if m:
        return ("id_number", clean, {"value": m.group(1).upper()})

    if re.fullmatch(r"[A-Za-z#\W_]{1,5}", clean):
        return ("junk", clean, {})

    if "姓名" in clean or re.search(r"姓\s+名", text):
        val = re.sub(r"^.*?姓\s*名", "", clean).strip()
        return ("name_label", clean, {"value": val})

    if re.search(r"[住佳往][址址]", clean) or "住 址" in text:
        val = re.sub(r"^.*?[住佳往][址址]", "", clean).strip()
        return ("addr_label", clean, {"value": val})

    m = re.search(r"(\d{2,4})年(\d{1,2})月(\d{1,2})日", clean)
    if m:
        return ("date_full", clean, {"y": m.group(1), "m": m.group(2), "d": m.group(3)})

    m = re.fullmatch(r"(?:出生)?(\d{2,4})年", clean)
    if m:
        return ("date_year", clean, {"value": m.group(1)})

    m = re.fullmatch(r"(\d{1,2})月(\d{1,2})日?", clean)
    if m:
        return ("date_md", clean, {"m": m.group(1), "d": m.group(2)})

    if re.search(r"[男女]", clean) and (
        "民" in clean or "族" in clean or "旅" in clean
    ):
        g = re.search(r"[男女]", clean).group(0)
        e = None
        after = re.search(r"民.*$", clean)
        if after:
            for ch in after.group(0)[1:]:
                if ch in ETHNICITIES:
                    e = ch
                    break
        if not e:
            for ch in clean:
                if ch in ETHNICITIES:
                    e = ch
                    break
        return ("gender_ethnicity", clean, {"gender": g, "ethnicity": e})

    if re.search(r"民[族旅]", clean):
        val = re.sub(r"^.*?民[族旅]", "", clean).strip()
        e = val[0] if val and val[0] in ETHNICITIES else None
        return ("ethnicity_label", clean, {"value": e})

    if clean in ("男", "女"):
        return ("gender", clean, {"value": clean})

    if "号码" in clean or "公民" in clean or "身份" in clean:
        return ("id_label", clean, {})

    if any(c in clean for c in ADDR_KEYWORDS) and len(clean) >= 2:
        return ("address_part", clean, {})

    if "出生" in clean:
        return ("birth_label", clean, {})

    if len(clean) == 1 and clean in LABEL_FRAGMENTS:
        return ("label_fragment", clean, {})

    if 1 <= len(clean) <= 3 and all(c in ETHNICITIES for c in clean):
        return ("ethnicity_value", clean, {"value": clean[0]})

    if 2 <= len(clean) <= 4 and re.fullmatch(r"[一-鿿]+", clean):
        if not any(kw in clean for kw in LABEL_KEYWORDS):
            return ("name_candidate", clean, {})

    return ("other", clean, {})


def _strip_addr_prefix(addr):
    """Remove OCR noise before the first province marker."""
    for prov in PROVINCES:
        idx = addr.find(prov)
        if idx == 0:
            return addr
        if 0 < idx <= 4:
            return addr[idx:]
    return addr


def parse_id_card(lines):
    """Parse ID card fields using a classify-then-bind approach.

    Resilient to scrambled OCR line order: each line is tagged semantically,
    then each field picks the best candidate from the tag pool instead of
    relying on positional state.
    """
    name_idx = -1
    for i, text in enumerate(lines):
        if "姓名" in text or "姓 名" in text:
            name_idx = i
            break
    if name_idx >= 0 and name_idx > len(lines) // 2:
        lines = list(reversed(lines))

    result = {
        "姓名": "",
        "性别": "",
        "民族": "",
        "出生日期": "",
        "住址": "",
        "身份证号": "",
    }

    tagged = [_classify_line(line) for line in lines]

    # ID number: prefer checksum-valid line match, then window scan, then any match
    for tag, _, info in tagged:
        if tag == "id_number" and _verify_id_checksum(info["value"]):
            result["身份证号"] = info["value"]
            break
    if not result["身份证号"]:
        all_digits = ""
        for tag, clean, _ in tagged:
            if tag in ("id_number", "id_label") or re.search(r"\d{10,}", clean):
                all_digits += re.sub(r"[^\dXx]", "", clean)
        for i in range(max(0, len(all_digits) - 17)):
            cand = all_digits[i:i + 18]
            if re.fullmatch(r"\d{17}[\dXx]", cand) and _verify_id_checksum(cand):
                result["身份证号"] = cand.upper()
                break
        if not result["身份证号"]:
            for tag, _, info in tagged:
                if tag == "id_number":
                    result["身份证号"] = info["value"]
                    break

    # Gender and ethnicity
    for tag, _, info in tagged:
        if tag == "gender_ethnicity":
            if not result["性别"]:
                result["性别"] = info["gender"]
            if not result["民族"] and info.get("ethnicity"):
                result["民族"] = info["ethnicity"]
        elif tag == "gender" and not result["性别"]:
            result["性别"] = info["value"]
        elif tag == "ethnicity_label" and not result["民族"] and info.get("value"):
            result["民族"] = info["value"]
    if not result["民族"]:
        for tag, _, info in tagged:
            if tag == "ethnicity_value":
                result["民族"] = info["value"]
                break

    # Birth date: full match first, then piece from year + month/day fragments
    for tag, _, info in tagged:
        if tag == "date_full":
            result["出生日期"] = (
                f"{info['y']}年{int(info['m'])}月{int(info['d'])}日"
            )
            break
    if not result["出生日期"]:
        year = md = None
        for tag, _, info in tagged:
            if tag == "date_year" and year is None:
                year = info["value"]
            elif tag == "date_md" and md is None:
                md = (info["m"], info["d"])
        if year and md:
            result["出生日期"] = (
                f"{year}年{int(md[0])}月{int(md[1])}日"
            )

    # Name: nearest name_candidate to a name_label / lone 名 fragment
    label_positions = []
    for i, (tag, clean, info) in enumerate(tagged):
        if tag == "name_label":
            val = info.get("value", "")
            if val and 2 <= len(val) <= 4 and re.fullmatch(r"[一-鿿]+", val):
                result["姓名"] = val
                break
            label_positions.append(i)
        elif tag == "label_fragment" and clean == "名":
            label_positions.append(i)
    if not result["姓名"]:
        candidates = [(i, clean) for i, (tag, clean, _) in enumerate(tagged)
                      if tag == "name_candidate"]
        if label_positions and candidates:
            best = min(
                candidates,
                key=lambda p: min(abs(p[0] - lp) for lp in label_positions),
            )
            result["姓名"] = best[1]
        elif candidates:
            result["姓名"] = candidates[0][1]

    # Address: addr_label value (if any) + all address_parts, strip junk prefix
    addr_head = ""
    addr_parts = []
    for tag, clean, info in tagged:
        if tag == "addr_label":
            val = info.get("value", "")
            if val:
                addr_head = val
        elif tag == "address_part":
            if result["姓名"] and clean == result["姓名"]:
                continue
            addr_parts.append(clean)
    full_addr = addr_head + "".join(addr_parts)
    if full_addr:
        result["住址"] = _strip_addr_prefix(full_addr)

    # Fallback from ID number
    if result["身份证号"] and len(result["身份证号"]) == 18:
        idn = result["身份证号"]
        if not result["出生日期"]:
            y, m, d = idn[6:10], idn[10:12], idn[12:14]
            result["出生日期"] = (
                f"{y}年{int(m)}月{int(d)}日"
            )
        if not result["性别"]:
            result["性别"] = "男" if int(idn[16]) % 2 == 1 else "女"

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python idcard_ocr.py <image_path>")
        sys.exit(1)

    image_path = os.path.abspath(sys.argv[1])
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot read image '{image_path}'")
        sys.exit(1)

    # Stage 1: Try to detect and crop the card from background
    card = extract_card(image)

    # Stage 2: Enhance image for better OCR
    card = enhance_image(card)

    # Upscale small cards so OCR has enough pixels to read fine characters
    h, w = card.shape[:2]
    if w < 900 or h < 550:
        scale = max(900 / w, 550 / h)
        card = cv2.resize(card, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Stage 3: OCR on the enhanced card image
    ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    ocr_result = ocr_engine.ocr(card, cls=True)

    if not ocr_result or not ocr_result[0]:
        print("Error: OCR failed, no text detected")
        sys.exit(1)

    # Collect all recognized text lines
    lines = [line[1][0] for line in ocr_result[0]]
    print(f"[debug] card size: {card.shape[1]}x{card.shape[0]}")
    print("[debug] OCR lines:")
    for i, l in enumerate(lines):
        print(f"  [{i}] {l}")

    # Stage 3: Parse fields
    result = parse_id_card(lines)

    # Output
    print("========== 身份证识别结果 ==========")
    print(f"姓名：{result['姓名']}")
    print(f"性别：{result['性别']}")
    print(f"民族：{result['民族']}")
    print(f"出生日期：{result['出生日期']}")
    print(f"住址：{result['住址']}")
    print(f"身份证号：{result['身份证号']}")
    print("=" * 36)


if __name__ == "__main__":
    main()
