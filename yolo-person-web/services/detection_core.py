from config import BOX_EXPAND_X, BOX_EXPAND_Y, NMS_IOU_THRESHOLD, TILE_GRID, TILE_OVERLAP


def first_present(data: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def extract_box_values(box: dict | list) -> list | None:
    if isinstance(box, dict):
        nested_box = (
            box.get("bbox")
            or box.get("box")
            or box.get("box_2d")
            or box.get("bbox_2d")
            or box.get("coordinates")
            or box.get("position")
        )
        if isinstance(nested_box, list) and len(nested_box) >= 4:
            return nested_box[:4]
        if isinstance(box.get("x1"), list) and len(box["x1"]) >= 4:
            return box["x1"][:4]
        if isinstance(nested_box, dict):
            return [
                first_present(nested_box, ("x1", "left")),
                first_present(nested_box, ("y1", "top")),
                first_present(nested_box, ("x2", "right")),
                first_present(nested_box, ("y2", "bottom")),
            ]
        return [
            first_present(box, ("x1", "left")),
            first_present(box, ("y1", "top")),
            first_present(box, ("x2", "right")),
            first_present(box, ("y2", "bottom")),
        ]
    if isinstance(box, list) and len(box) >= 4:
        return box[:4]
    return None


def normalize_raw_box(raw_box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = raw_box
    try:
        max_value = max(x1, y1, x2, y2)
    except (TypeError, ValueError):
        return None

    if max_value <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif max_value <= 1000 and (x2 > width or y2 > height):
        x1, x2 = x1 / 1000 * width, x2 / 1000 * width
        y1, y2 = y1 / 1000 * height, y2 / 1000 * height

    left = max(0, min(width - 1, round(min(x1, x2))))
    top = max(0, min(height - 1, round(min(y1, y2))))
    right = max(0, min(width - 1, round(max(x1, x2))))
    bottom = max(0, min(height - 1, round(max(y1, y2))))

    if right - left < 4 or bottom - top < 4:
        return None
    return left, top, right, bottom


def normalize_box(box: dict | list, width: int, height: int) -> tuple[int, int, int, int] | None:
    raw_values = extract_box_values(box)
    if raw_values is None:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in raw_values]
    except (TypeError, ValueError):
        return None
    return normalize_raw_box((x1, y1, x2, y2), width, height)


def infer_box_coordinate_size(boxes: list, default_width: int, default_height: int) -> tuple[int, int]:
    raw_boxes = []
    for box in boxes:
        raw_values = extract_box_values(box)
        if raw_values is None:
            continue
        try:
            raw_boxes.append(tuple(float(value) for value in raw_values[:4]))
        except (TypeError, ValueError):
            continue
    if not raw_boxes:
        return default_width, default_height

    max_x = max(max(box[0], box[2]) for box in raw_boxes)
    max_y = max(max(box[1], box[3]) for box in raw_boxes)
    max_value = max(max_x, max_y)
    if max_value <= 1.0:
        return 1, 1
    if max_value <= 1000 and (default_width > 1200 or default_height > 1200):
        return 1000, 1000
    return default_width, default_height


def expand_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    expand_x: float = BOX_EXPAND_X,
    expand_y: float = BOX_EXPAND_Y,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    pad_x = round((right - left) * expand_x)
    pad_y = round((bottom - top) * expand_y)
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width - 1, right + pad_x),
        min(height - 1, bottom + pad_y),
    )


def box_iou(first: dict, second: dict) -> float:
    left = max(first["x1"], second["x1"])
    top = max(first["y1"], second["y1"])
    right = min(first["x2"], second["x2"])
    bottom = min(first["y2"], second["y2"])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = (first["x2"] - first["x1"]) * (first["y2"] - first["y1"])
    second_area = (second["x2"] - second["x1"]) * (second["y2"] - second["y1"])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def merge_boxes(boxes: list[dict]) -> list[dict]:
    valid_boxes = [box for box in boxes if box["x2"] > box["x1"] and box["y2"] > box["y1"]]
    valid_boxes.sort(key=lambda box: (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]), reverse=True)
    merged = []
    for box in valid_boxes:
        if all(box_iou(box, kept_box) < NMS_IOU_THRESHOLD for kept_box in merged):
            merged.append(box)
    return merged


def limit_boxes(boxes: list[dict], max_count: int) -> list[dict]:
    return boxes if len(boxes) <= max_count else boxes[:max_count]


def convert_boxes_to_model_coords(
    boxes: list,
    source_width: int,
    source_height: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[dict]:
    coord_width, coord_height = infer_box_coordinate_size(boxes, source_width, source_height)
    scale_x = source_width / coord_width
    scale_y = source_height / coord_height
    converted = []
    for box in boxes:
        normalized_box = normalize_box(box, coord_width, coord_height)
        if normalized_box is None:
            continue
        left, top, right, bottom = normalized_box
        converted.append(
            {
                "x1": round(left * scale_x + offset_x),
                "y1": round(top * scale_y + offset_y),
                "x2": round(right * scale_x + offset_x),
                "y2": round(bottom * scale_y + offset_y),
            }
        )
    return converted


def generate_tiles(width: int, height: int) -> list[tuple[int, int, int, int]]:
    if TILE_GRID <= 1:
        return []
    step_x = width / TILE_GRID
    step_y = height / TILE_GRID
    overlap_x = round(step_x * TILE_OVERLAP)
    overlap_y = round(step_y * TILE_OVERLAP)
    tiles = []
    for row in range(TILE_GRID):
        for col in range(TILE_GRID):
            left = max(0, round(col * step_x) - overlap_x)
            top = max(0, round(row * step_y) - overlap_y)
            right = min(width, round((col + 1) * step_x) + overlap_x)
            bottom = min(height, round((row + 1) * step_y) + overlap_y)
            if right - left >= 120 and bottom - top >= 120:
                tiles.append((left, top, right, bottom))
    return tiles
