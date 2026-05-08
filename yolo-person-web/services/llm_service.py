import json
import re
from pathlib import Path

import requests

from config import ENABLE_TILE_DETECTION, LLM_TIMEOUT, LOG_DIR
from services.detection_core import convert_boxes_to_model_coords, generate_tiles, limit_boxes, merge_boxes
from services.image_service import pil_image_to_data_url, prepare_model_image_object, save_model_response_log


def repair_json_text(text: str) -> str:
    repaired = text.strip()
    repaired = repaired.replace("，", ",").replace("：", ":")
    repaired = repaired.replace("“", '"').replace("”", '"').replace("‘", '"').replace("’", '"')
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', repaired)
    return repaired


def extract_detection_from_text(text: str) -> dict | None:
    count_match = re.search(r'"?person_count"?\s*[:：]\s*(\d+)', text)
    if not count_match:
        count_match = re.search(r"(?:人数|检测到|person_count)[^\d]{0,12}(\d+)", text)

    boxes = []
    for match in re.finditer(
        r'"?x\d+"?\s*[:：]\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
        text,
    ):
        boxes.append({"x1": float(match.group(1)), "y1": float(match.group(2)), "x2": float(match.group(3)), "y2": float(match.group(4))})

    for match in re.finditer(
        r'"?x1"?\s*[:：]\s*(-?\d+(?:\.\d+)?).*?'
        r'"?y1"?\s*[:：]\s*(-?\d+(?:\.\d+)?).*?'
        r'"?x2"?\s*[:：]\s*(-?\d+(?:\.\d+)?).*?'
        r'"?y2"?\s*[:：]\s*(-?\d+(?:\.\d+)?)',
        text,
        flags=re.DOTALL,
    ):
        boxes.append({"x1": float(match.group(1)), "y1": float(match.group(2)), "x2": float(match.group(3)), "y2": float(match.group(4))})

    if not count_match and not boxes:
        return None

    person_count = int(count_match.group(1)) if count_match else len(boxes)
    return {"person_count": person_count, "boxes": boxes, "analysis": "模型返回的 JSON 不完整，系统已尽量提取人数和坐标。"}


def parse_model_json(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)

    last_error = None
    for candidate in (cleaned, repair_json_text(cleaned)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    fallback = extract_detection_from_text(cleaned)
    if fallback is not None:
        return fallback

    preview = cleaned[:1000]
    raise ValueError(f"模型没有返回严格 JSON，解析失败：{last_error}。返回内容片段：{preview}")


def build_detection_prompt(width: int, height: int, scope: str = "整张图片") -> str:
    return (
        f"请检测{scope}中可见的人数，并给出每个真实人物的外接矩形框。图片尺寸为 {width}x{height} 像素。"
        "只统计真实可见的人，不统计照片、海报、雕像或屏幕里的人。"
        "每一个计入 person_count 的人都必须在 boxes 中有且只有一个外接矩形框，boxes 数量必须等于 person_count。"
        "不要把同一个人拆成多个框，不要为车辆、影子、反光、标牌或身体部位单独建框。"
        "不要漏掉远处、小尺寸、遮挡或边缘位置的真实人物。"
        "坐标必须使用当前图片像素坐标，左上角为 (0,0)，字段为 x1,y1,x2,y2。"
        "必须只返回一行紧凑 JSON，不要 Markdown，不要解释，不要换行，不要尾逗号。格式为："
        '{"person_count": 0, "boxes": [{"x1": 0, "y1": 0, "x2": 0, "y2": 0}], "analysis": "简短中文说明"}'
    )


def request_detection(api_config: dict, image_url: str, width: int, height: int, scope: str) -> tuple[dict, dict]:
    payload = {
        "model": api_config["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_detection_prompt(width, height, scope)},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_config['api_key']}", "Content-Type": "application/json"}

    try:
        response = requests.post(api_config["api_url"], headers=headers, json=payload, timeout=LLM_TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise ValueError(f"接口响应超时：已等待 {LLM_TIMEOUT} 秒，请重试或调大 LLM_TIMEOUT。") from exc
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text.strip()
        if len(detail) > 800:
            detail = f"{detail[:800]}..."
        raise ValueError(f"接口请求失败：HTTP {response.status_code}，地址：{api_config['api_url']}，响应：{detail}") from exc

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return parse_model_json(content), {"response": data, "content": content}


def call_vision_model(image_path: Path, api_config: dict, draw_person_boxes) -> tuple[int, str, str]:
    if not api_config["api_key"]:
        raise ValueError("未配置 API Key，请先设置大模型 API Key。")

    model_image, model_image_info = prepare_model_image_object(image_path)
    model_image_url, jpeg_bytes = pil_image_to_data_url(model_image)
    model_image_info = {**model_image_info, "jpeg_bytes": jpeg_bytes}
    model_image_width, model_image_height = model_image.size

    result, raw_log = request_detection(api_config, model_image_url, model_image_width, model_image_height, "整张图片")
    log_payload = {"global": raw_log, "tiles": []}
    log_path = Path(save_model_response_log(image_path, api_config, log_payload, raw_log["content"], model_image_info))

    try:
        person_count = int(result["person_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"模型返回结果缺少有效 person_count：{result}") from exc

    analysis = str(result.get("analysis") or "模型未返回说明。")
    boxes = result.get("boxes") or result.get("persons") or result.get("people") or result.get("detections") or []
    if not isinstance(boxes, list):
        boxes = []
    if len(boxes) > person_count:
        boxes = boxes[:person_count]
    merged_boxes = convert_boxes_to_model_coords(boxes, model_image_width, model_image_height)

    tile_added = 0
    tile_errors = 0
    if ENABLE_TILE_DETECTION:
        for tile_index, (left, top, right, bottom) in enumerate(generate_tiles(model_image_width, model_image_height), start=1):
            crop = model_image.crop((left, top, right, bottom))
            tile_url, _ = pil_image_to_data_url(crop)
            try:
                tile_result, tile_log = request_detection(api_config, tile_url, crop.width, crop.height, f"第 {tile_index} 个局部区域")
            except Exception as exc:
                tile_errors += 1
                log_payload["tiles"].append({"tile": [left, top, right, bottom], "error": str(exc)})
                continue

            tile_boxes = tile_result.get("boxes") or tile_result.get("persons") or tile_result.get("people") or tile_result.get("detections") or []
            if isinstance(tile_boxes, list) and len(merged_boxes) < person_count:
                before_count = len(merged_boxes)
                merged_boxes = merge_boxes(merged_boxes + convert_boxes_to_model_coords(tile_boxes, crop.width, crop.height, left, top))
                merged_boxes = limit_boxes(merged_boxes, person_count)
                tile_added += max(0, len(merged_boxes) - before_count)
            log_payload["tiles"].append({"tile": [left, top, right, bottom], "result": tile_result, "raw_content": tile_log["content"], "raw_response": tile_log["response"]})

        saved_log = json.loads(log_path.read_text(encoding="utf-8"))
        saved_log["raw_response"] = log_payload
        log_path.write_text(json.dumps(saved_log, ensure_ascii=False, indent=2), encoding="utf-8")
        if tile_added:
            analysis = f"{analysis}（分块补检新增 {tile_added} 个候选框。）"
        if tile_errors:
            analysis = f"{analysis}（有 {tile_errors} 个分块补检失败。）"

    result_name, drawn_count = draw_person_boxes(image_path, merged_boxes, (model_image_width, model_image_height))
    if drawn_count != person_count:
        analysis = f"{analysis}（注意：模型识别人数为 {person_count}，合并后成功绘制的框为 {drawn_count} 个。）"
    analysis = f"{analysis}（原始返回已保存：{log_path.relative_to(LOG_DIR.parent).as_posix()}）"
    return person_count, analysis, result_name
