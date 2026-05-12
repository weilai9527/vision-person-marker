import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import requests

from config import ENABLE_TILE_DETECTION, LLM_TIMEOUT, LOG_DIR
from models import DetectionResult, db
from services.detection_core import convert_boxes_to_model_coords, generate_tiles, limit_boxes, merge_boxes
from services.image_service import pil_image_to_data_url, prepare_model_image_object, save_model_response_log
from services.yolo_service import run_yolo_detection


def repair_json_text(text: str) -> str:
    repaired = text.strip()
    repaired = repaired.replace("，", ",").replace("：", ":")
    repaired = repaired.replace("“", '"').replace("”", '"').replace("‘", '"').replace("’", '"')
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', repaired)
    # 修复大模型简写格式：{"x1":83,752,324,926} → {"x1":83,"y1":752,"x2":324,"y2":926}
    repaired = re.sub(
        r'\{\s*"x1"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\}',
        r'{"x1":\1,"y1":\2,"x2":\3,"y2":\4}',
        repaired,
    )
    return repaired


def extract_detection_from_text(text: str) -> dict | None:
    count_match = re.search(r'"?(?:person_count|vehicle_count)"?\s*[:：]\s*(\d+)', text)
    if not count_match:
        count_match = re.search(r"(?:人数|车辆数|检测到|person_count|vehicle_count)[^\d]{0,12}(\d+)", text)

    boxes = []
    # 标准数组格式：[x1,y1,x2,y2]
    for match in re.finditer(
        r'"?x\d+"?\s*[:：]\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
        text,
    ):
        boxes.append({"x1": float(match.group(1)), "y1": float(match.group(2)), "x2": float(match.group(3)), "y2": float(match.group(4))})

    # 标准键值对格式：{"x1":v1,"y1":v2,"x2":v3,"y2":v4}
    for match in re.finditer(
        r'"?x1"?\s*[:：]\s*(-?\d+(?:\.\d+)?).*?'
        r'"?y1"?\s*[:：]\s*(-?\d+(?:\.\d+)?).*?'
        r'"?x2"?\s*[:：]\s*(-?\d+(?:\.\d+)?).*?'
        r'"?y2"?\s*[:：]\s*(-?\d+(?:\.\d+)?)',
        text,
        flags=re.DOTALL,
    ):
        boxes.append({"x1": float(match.group(1)), "y1": float(match.group(2)), "x2": float(match.group(3)), "y2": float(match.group(4))})

    # 容错：处理大模型省略键名或错误使用括号的格式
    # 例如：{"x1":334,589,556,867} 或 {"x1":458,546),(601,745)} 或 {"x1":84,753,324,926}
    for match in re.finditer(
        r'\{\s*"?x1"?\s*[:\uFF1A]\s*(-?\d+(?:\.\d+)?)[,;:\s)\]]*(-?\d+(?:\.\d+)?)[,;:\s)\]]*(-?\d+(?:\.\d+)?)[,;:\s)\]]*(-?\d+(?:\.\d+)?)',
        text,
    ):
        boxes.append({"x1": float(match.group(1)), "y1": float(match.group(2)), "x2": float(match.group(3)), "y2": float(match.group(4))})

    if not count_match and not boxes:
        return None

    count = int(count_match.group(1)) if count_match else len(boxes)

    # 如果声称有目标但一个框都没提取到，说明 JSON 解析失败，不应静默 fallback
    if count > 0 and not boxes:
        return None

    return {"person_count": count, "vehicle_count": count, "boxes": boxes, "analysis": "模型返回的 JSON 不完整，系统已尽量提取数量和坐标。"}


def parse_model_json(content: str) -> dict:
    cleaned = content.strip()
    if not cleaned:
        raise ValueError("模型返回内容为空，无法解析 JSON。")
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
        f"任务：检测{scope}中可见的人数，并给出每个真实人物的外接矩形框。图片尺寸为 {width}x{height} 像素。"
        "只统计真实可见的人，不统计照片、海报、雕像或屏幕里的人。"
        "每一个计入 person_count 的人都必须在 boxes 中有且只有一个外接矩形框，boxes 数量必须等于 person_count。"
        "不要把同一个人拆成多个框，不要为车辆、影子、反光、标牌或身体部位单独建框。"
        "不要漏掉远处、小尺寸、遮挡或边缘位置的真实人物。"
        "坐标必须使用当前图片像素坐标，左上角为 (0,0)。\n"
        "\n"
        "【输出格式要求 - 严格遵守】\n"
        "1. 必须输出严格合法的 JSON 对象，必须通过标准 JSON 解析器验证。\n"
        f'2. 每个框必须使用完整键名，格式为：{{"x1":整数,"y1":整数,"x2":整数,"y2":整数}}\n'
        "3. 禁止省略 y1、x2、y2 键名，禁止使用圆括号，禁止把四个数字直接跟在 x1 后面。\n"
        "4. analysis 必须少于 20 个汉字。\n"
        "5. 只返回一行紧凑 JSON，不要 Markdown，不要解释，不要换行，不要尾逗号。\n"
        "\n"
        "正确示例（合法 JSON）：\n"
        f'{{"person_count":2,"boxes":[{{"x1":100,"y1":200,"x2":300,"y2":400}},{{"x1":500,"y1":600,"x2":700,"y2":800}}],"analysis":"两人站立交谈"}}\n'
        "\n"
        "错误示例（非法 JSON，严禁输出）：\n"
        f'{{"x1":100,200,300,400}}  ← 省略了 y1/x2/y2 键名，不是合法 JSON\n'
        f'{{"x1":100,"y1":200,"x2":300}}  ← 缺少 y2 键名，不是合法 JSON\n'
    )


def build_yolo_judge_prompt(width: int, height: int, yolo_boxes: list[dict]) -> str:
    candidates = [
        {
            "id": index,
            "x1": box.get("x1"),
            "y1": box.get("y1"),
            "x2": box.get("x2"),
            "y2": box.get("y2"),
            "conf": box.get("conf"),
        }
        for index, box in enumerate(yolo_boxes, start=1)
    ]
    return (
        "你是 YOLO 人员检测流程的行为裁判，不要替代 YOLO 做检测，不要重新输出人物框。"
        f"图片尺寸为 {width}x{height} 像素。"
        f"YOLO 候选框如下：{json.dumps(candidates, ensure_ascii=False, separators=(',', ':'))}。"
        "请观察原图，只给 YOLO 后处理建议。"
        "只统计真实可见的人，不统计照片、海报、雕像或屏幕里的人。"
        "action 只能是 keep_yolo、filter_false_positive、rerun_low_conf、rerun_high_conf、rerun_high_res。"
        "如果明显漏检，选 rerun_low_conf；如果小目标密集，选 rerun_high_res；如果明显误检多，选 rerun_high_conf；如果只有少数明确误检，选 filter_false_positive。"
        "false_positive_ids 只填写明显不是人的 YOLO 框编号，不确定就留空。"
        "missed_hint 只写是否疑似漏检，用短中文描述，不要给坐标。"
        "analysis 必须少于 30 个汉字。最终人数和画框由 YOLO 决定。"
        "必须只返回一行紧凑 JSON，不要 Markdown，不要解释，不要尾逗号。格式为："
        '{"action":"keep_yolo","false_positive_ids":[],"missed_hint":"","analysis":"简述"}'
    )


def build_vehicle_judge_prompt(width: int, height: int, yolo_boxes: list[dict]) -> str:
    candidates = [
        {
            "id": index,
            "x1": box.get("x1"),
            "y1": box.get("y1"),
            "x2": box.get("x2"),
            "y2": box.get("y2"),
            "conf": box.get("conf"),
        }
        for index, box in enumerate(yolo_boxes, start=1)
    ]
    return (
        "你是 YOLO 车辆检测流程的行为裁判，不要替代 YOLO 做检测，不要重新输出车辆框。"
        f"图片尺寸为 {width}x{height} 像素。"
        f"YOLO 候选框如下：{json.dumps(candidates, ensure_ascii=False, separators=(',', ':'))}。"
        "请观察原图，只给 YOLO 后处理建议。"
        "只统计真实可见的车辆，包括自行车、小汽车、摩托车、公交车、卡车。不统计照片、海报、模型或屏幕里的车辆。"
        "action 只能是 keep_yolo、filter_false_positive、rerun_low_conf、rerun_high_conf、rerun_high_res。"
        "如果明显漏检，选 rerun_low_conf；如果小目标密集，选 rerun_high_res；如果明显误检多，选 rerun_high_conf；如果只有少数明确误检，选 filter_false_positive。"
        "false_positive_ids 只填写明显不是车辆的 YOLO 框编号（如阴影、广告牌、建筑物被误检为车），不确定就留空。"
        "missed_hint 只写是否疑似漏检，用短中文描述，不要给坐标。"
        "analysis 必须少于 30 个汉字。最终车辆数和画框由 YOLO 决定。"
        "必须只返回一行紧凑 JSON，不要 Markdown，不要解释，不要换行，不要尾逗号。格式为："
        '{"action":"keep_yolo","false_positive_ids":[],"missed_hint":"","analysis":"简述"}'
    )


def get_request_temperature(api_config: dict) -> int:
    provider = (api_config.get("provider") or "").lower()
    api_url = (api_config.get("api_url") or "").lower()
    model = (api_config.get("model") or "").lower()
    if provider == "kimi" or "moonshot" in api_url or model.startswith("kimi-"):
        return 1
    return 0


def supports_response_format(api_config: dict) -> bool:
    api_url = (api_config.get("api_url") or "").lower()
    if "moonshot" in api_url:
        return True
    if "api.openai.com" in api_url:
        return True
    return False


def save_failed_response_log(api_config: dict, payload: dict, response_data: dict | str, content: str) -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}-failed.json"
    log_path = LOG_DIR / log_name
    log_payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "provider": api_config.get("provider"),
        "api_url": api_config.get("api_url"),
        "model": api_config.get("model"),
        "payload_without_image": {
            **payload,
            "messages": [
                {
                    **message,
                    "content": [
                        item if item.get("type") != "image_url" else {"type": "image_url", "image_url": {"url": "[base64 image omitted]"}}
                        for item in message.get("content", [])
                    ],
                }
                for message in payload.get("messages", [])
            ],
        },
        "raw_content": content,
        "raw_response": response_data,
    }
    log_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(log_path)


def post_vision_json(api_config: dict, payload: dict) -> tuple[dict, dict, str]:
    headers = {"Authorization": f"Bearer {api_config['api_key']}", "Content-Type": "application/json"}

    try:
        response = requests.post(api_config["api_url"], headers=headers, json=payload, timeout=LLM_TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise ValueError(f"接口响应超时：已等待 {LLM_TIMEOUT} 秒，请重试或调大 LLM_TIMEOUT。") from exc
    except requests.exceptions.RequestException as exc:
        log_path = save_failed_response_log(api_config, payload, str(exc), "")
        raise ValueError(
            "接口请求失败：网络或代理连接异常，未能连接到大模型服务。"
            f"失败请求已保存：{Path(log_path).relative_to(LOG_DIR.parent).as_posix()}。"
            f"详情：{exc}"
        ) from exc
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text.strip()
        if len(detail) > 800:
            detail = f"{detail[:800]}..."
        raise ValueError(f"接口请求失败：HTTP {response.status_code}，地址：{api_config['api_url']}，响应：{detail}") from exc

    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
    try:
        parsed = parse_model_json(content)
    except ValueError as exc:
        log_path = save_failed_response_log(api_config, payload, data, content)
        raise ValueError(f"{exc} 失败响应已保存：{Path(log_path).relative_to(LOG_DIR.parent).as_posix()}") from exc
    return parsed, data, content


SYSTEM_PROMPT = (
    "你是一个严格遵循 JSON 格式规范的视觉检测助手。"
    "你的唯一任务是检测图片中的目标并输出严格合法的 JSON。"
    "任何格式错误（如省略键名、使用圆括号、缺少引号）都会导致结果无法使用。"
    "你必须确保输出的 JSON 可以通过标准 JSON 解析器直接验证。"
)


def build_payload(api_config: dict, prompt: str, image_url: str, max_tokens: int) -> dict:
    payload = {
        "model": api_config["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": get_request_temperature(api_config),
        "max_tokens": max_tokens,
    }
    if supports_response_format(api_config):
        payload["response_format"] = {"type": "json_object"}
    return payload


def request_detection(api_config: dict, image_url: str, width: int, height: int, scope: str) -> tuple[dict, dict]:
    payload = build_payload(api_config, build_detection_prompt(width, height, scope), image_url, 4096)
    parsed, data, content = post_vision_json(api_config, payload)
    return parsed, {"response": data, "content": content}


def request_yolo_judgement(api_config: dict, image_url: str, width: int, height: int, yolo_boxes: list[dict]) -> tuple[dict, dict]:
    payload = build_payload(api_config, build_yolo_judge_prompt(width, height, yolo_boxes), image_url, 1024)
    parsed, data, content = post_vision_json(api_config, payload)
    return parsed, {"response": data, "content": content}


def request_vehicle_judgement(api_config: dict, image_url: str, width: int, height: int, yolo_boxes: list[dict]) -> tuple[dict, dict]:
    payload = build_payload(api_config, build_vehicle_judge_prompt(width, height, yolo_boxes), image_url, 1024)
    parsed, data, content = post_vision_json(api_config, payload)
    return parsed, {"response": data, "content": content}


def normalize_judgement_ids(value, max_id: int) -> list[int]:
    if not isinstance(value, list):
        return []
    ids = []
    for item in value:
        try:
            candidate = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= candidate <= max_id and candidate not in ids:
            ids.append(candidate)
    return ids


def scale_boxes_for_model(boxes: list[dict], source_width: int, source_height: int, target_width: int, target_height: int) -> list[dict]:
    if source_width <= 0 or source_height <= 0:
        return boxes
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    scaled_boxes = []
    for box in boxes:
        scaled = dict(box)
        scaled["x1"] = round(float(box["x1"]) * scale_x)
        scaled["y1"] = round(float(box["y1"]) * scale_y)
        scaled["x2"] = round(float(box["x2"]) * scale_x)
        scaled["y2"] = round(float(box["y2"]) * scale_y)
        scaled_boxes.append(scaled)
    return scaled_boxes


def choose_yolo_action(action: str) -> tuple[str, dict, str]:
    action = action if action in {"keep_yolo", "filter_false_positive", "rerun_low_conf", "rerun_high_conf", "rerun_high_res"} else "keep_yolo"
    if action == "rerun_low_conf":
        return action, {"conf": 0.12, "iou": 0.55, "imgsz": 1536, "min_area": 35}, "低阈值补漏"
    if action == "rerun_high_conf":
        return action, {"conf": 0.45, "iou": 0.45, "imgsz": 1280, "min_area": 80}, "高阈值去误检"
    if action == "rerun_high_res":
        return action, {"conf": 0.20, "iou": 0.55, "imgsz": 1600, "min_area": 35}, "高分辨率小目标"
    if action == "filter_false_positive":
        return action, {}, "剔除明确误检"
    return "keep_yolo", {}, "保留 YOLO"


def call_yolo_judge_model(
    image_path: Path,
    image_record_id: int,
    api_config: dict,
    draw_person_boxes,
    yolo_boxes: list[dict],
    yolo_width: int,
    yolo_height: int,
    analysis_prefix: str = "",
) -> tuple[int, str, str, int, int]:
    if not api_config["api_key"]:
        raise ValueError("未配置 API Key，请先设置大模型 API Key。")

    model_image, model_image_info = prepare_model_image_object(image_path)
    model_image_url, jpeg_bytes = pil_image_to_data_url(model_image)
    model_image_info = {**model_image_info, "jpeg_bytes": jpeg_bytes}
    model_image_width, model_image_height = model_image.size
    judge_boxes = scale_boxes_for_model(yolo_boxes, yolo_width, yolo_height, model_image_width, model_image_height)

    judgement, raw_log = request_yolo_judgement(
        api_config,
        model_image_url,
        model_image_width,
        model_image_height,
        judge_boxes,
    )
    log_payload = {
        "mode": "yolo_judge",
        "yolo_box_count": len(yolo_boxes),
        "judgement": raw_log,
    }
    log_path = Path(save_model_response_log(image_path, api_config, log_payload, raw_log["content"], model_image_info))

    action, yolo_overrides, action_label = choose_yolo_action(str(judgement.get("action") or "keep_yolo"))
    false_positive_ids = normalize_judgement_ids(judgement.get("false_positive_ids"), len(yolo_boxes))

    if yolo_overrides:
        reran_boxes, reran_width, reran_height = run_yolo_detection(image_path, **yolo_overrides)
        # 大模型不能剔除 YOLO 标注：rerun 后如果框变少了，回退到原框保证不减少
        if len(reran_boxes) >= len(yolo_boxes):
            final_boxes, final_width, final_height = reran_boxes, reran_width, reran_height
            false_positive_ids = []
        else:
            final_boxes, final_width, final_height = yolo_boxes, yolo_width, yolo_height
            false_positive_ids = []
            action_label = f"{action_label}（ rerun 后框减少，已回退保留原框）"
    else:
        final_boxes = yolo_boxes
        final_width = yolo_width
        final_height = yolo_height
        # 大模型不能剔除 YOLO 标注：禁用 filter_false_positive
        if action == "filter_false_positive":
            action_label = f"{action_label}（已禁用，保留全部 {len(yolo_boxes)} 个框）"
        false_positive_ids = []

    result_name, drawn_count = draw_person_boxes(image_path, final_boxes, (final_width, final_height))

    missed_hint = str(judgement.get("missed_hint") or "").strip()
    judge_analysis = str(judgement.get("analysis") or "大模型已复核 YOLO 结果。")
    analysis = (
        f"{analysis_prefix}"
        f"大模型只给 YOLO 后处理建议：{judge_analysis}"
        f"行为：{action_label}。YOLO 原始框 {len(yolo_boxes)} 个，剔除明显误检 {len(false_positive_ids)} 个，最终绘制 {drawn_count} 个框。"
    )
    if missed_hint:
        analysis = f"{analysis}提示：{missed_hint}。"
    analysis = f"{analysis}（裁判原始返回已保存：{log_path.relative_to(LOG_DIR.parent).as_posix()}）"

    detection_result = DetectionResult(
        image_id=image_record_id,
        person_count=drawn_count,
        bounding_boxes_json=json.dumps(final_boxes, ensure_ascii=False),
        llm_analysis_text=analysis,
        result_image_path=f"static/results/{result_name}",
        llm_api_provider=api_config.get("provider", "vision_model_judge"),
        llm_model_name=api_config.get("model", ""),
        raw_llm_response_log_path=str(log_path),
    )
    db.session.add(detection_result)
    db.session.commit()

    return drawn_count, analysis, result_name, final_width, final_height


def call_vehicle_yolo_judge_model(
    image_path: Path,
    image_record_id: int,
    api_config: dict,
    draw_vehicle_boxes,
    yolo_boxes: list[dict],
    yolo_width: int,
    yolo_height: int,
    analysis_prefix: str = "",
) -> tuple[int, str, str, int, int]:
    if not api_config["api_key"]:
        raise ValueError("未配置 API Key，请先设置大模型 API Key。")

    model_image, model_image_info = prepare_model_image_object(image_path)
    model_image_url, jpeg_bytes = pil_image_to_data_url(model_image)
    model_image_info = {**model_image_info, "jpeg_bytes": jpeg_bytes}
    model_image_width, model_image_height = model_image.size
    judge_boxes = scale_boxes_for_model(yolo_boxes, yolo_width, yolo_height, model_image_width, model_image_height)

    judgement, raw_log = request_vehicle_judgement(
        api_config,
        model_image_url,
        model_image_width,
        model_image_height,
        judge_boxes,
    )
    log_payload = {
        "mode": "vehicle_yolo_judge",
        "yolo_box_count": len(yolo_boxes),
        "judgement": raw_log,
    }
    log_path = Path(save_model_response_log(image_path, api_config, log_payload, raw_log["content"], model_image_info))

    action, yolo_overrides, action_label = choose_yolo_action(str(judgement.get("action") or "keep_yolo"))
    false_positive_ids = normalize_judgement_ids(judgement.get("false_positive_ids"), len(yolo_boxes))

    if yolo_overrides:
        reran_boxes, reran_width, reran_height = run_yolo_detection(
            image_path,
            class_ids=VEHICLE_CLASS_IDS,
            target_label="vehicle",
            **yolo_overrides,
        )
        # 大模型不能剔除 YOLO 标注：rerun 后如果框变少了，回退到原框保证不减少
        if len(reran_boxes) >= len(yolo_boxes):
            final_boxes, final_width, final_height = reran_boxes, reran_width, reran_height
            false_positive_ids = []
        else:
            final_boxes, final_width, final_height = yolo_boxes, yolo_width, yolo_height
            false_positive_ids = []
            action_label = f"{action_label}（ rerun 后框减少，已回退保留原框）"
    else:
        final_boxes = yolo_boxes
        final_width = yolo_width
        final_height = yolo_height
        # 大模型不能剔除 YOLO 标注：禁用 filter_false_positive
        if action == "filter_false_positive":
            action_label = f"{action_label}（已禁用，保留全部 {len(yolo_boxes)} 个框）"
        false_positive_ids = []

    result_name, drawn_count = draw_vehicle_boxes(image_path, final_boxes, (final_width, final_height))

    missed_hint = str(judgement.get("missed_hint") or "").strip()
    judge_analysis = str(judgement.get("analysis") or "大模型已复核 YOLO 结果。")
    analysis = (
        f"{analysis_prefix}"
        f"大模型只给 YOLO 后处理建议：{judge_analysis}"
        f"行为：{action_label}。YOLO 原始框 {len(yolo_boxes)} 个，剔除明显误检 {len(false_positive_ids)} 个，最终绘制 {drawn_count} 个框。"
    )
    if missed_hint:
        analysis = f"{analysis}提示：{missed_hint}。"
    analysis = f"{analysis}（裁判原始返回已保存：{log_path.relative_to(LOG_DIR.parent).as_posix()}）"

    detection_result = DetectionResult(
        image_id=image_record_id,
        person_count=drawn_count,
        bounding_boxes_json=json.dumps(final_boxes, ensure_ascii=False),
        llm_analysis_text=analysis,
        result_image_path=f"static/results/{result_name}",
        llm_api_provider=api_config.get("provider", "vision_model_vehicle_judge"),
        llm_model_name=api_config.get("model", ""),
        raw_llm_response_log_path=str(log_path),
    )
    db.session.add(detection_result)
    db.session.commit()

    return drawn_count, analysis, result_name, final_width, final_height


def call_vision_model(
    image_path: Path,
    image_record_id: int,
    api_config: dict,
    draw_person_boxes,
    analysis_prefix: str = "",
) -> tuple[int, str, str, int, int]:
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
    if analysis_prefix:
        analysis = f"{analysis_prefix}{analysis}"
    analysis = f"{analysis}（原始返回已保存：{log_path.relative_to(LOG_DIR.parent).as_posix()}）"

    detection_result = DetectionResult(
        image_id=image_record_id,
        person_count=person_count,
        bounding_boxes_json=json.dumps(merged_boxes, ensure_ascii=False),
        llm_analysis_text=analysis,
        result_image_path=f"static/results/{result_name}",
        llm_api_provider=api_config.get("provider", "vision_model"),
        llm_model_name=api_config.get("model", ""),
        raw_llm_response_log_path=str(log_path),
    )
    db.session.add(detection_result)
    db.session.commit()

    return person_count, analysis, result_name, model_image_width, model_image_height


def build_vehicle_detection_prompt(width: int, height: int, scope: str = "整张图片") -> str:
    return (
        f"任务：检测{scope}中可见的车辆，并给出每个真实车辆的外接矩形框。图片尺寸为 {width}x{height} 像素。"
        "只统计真实可见的车辆，包括自行车、小汽车、摩托车、公交车、卡车。"
        "不统计照片、海报、模型或屏幕里的车辆。"
        "每一个计入 vehicle_count 的车辆都必须在 boxes 中有且只有一个外接矩形框，boxes 数量必须等于 vehicle_count。"
        "不要把同一辆车拆成多个框，不要为阴影、反光、广告牌或建筑物单独建框。"
        "不要漏掉远处、小尺寸、遮挡或边缘位置的真实车辆。"
        "坐标必须使用当前图片像素坐标，左上角为 (0,0)。\n"
        "\n"
        "【输出格式要求 - 严格遵守】\n"
        "1. 必须输出严格合法的 JSON 对象，必须通过标准 JSON 解析器验证。\n"
        f'2. 每个框必须使用完整键名，格式为：{{"x1":整数,"y1":整数,"x2":整数,"y2":整数}}\n'
        "3. 禁止省略 y1、x2、y2 键名，禁止使用圆括号，禁止把四个数字直接跟在 x1 后面。\n"
        "4. analysis 必须少于 20 个汉字。\n"
        "5. 只返回一行紧凑 JSON，不要 Markdown，不要解释，不要换行，不要尾逗号。\n"
        "\n"
        "正确示例（合法 JSON）：\n"
        f'{{"vehicle_count":2,"boxes":[{{"x1":100,"y1":200,"x2":300,"y2":400}},{{"x1":500,"y1":600,"x2":700,"y2":800}}],"analysis":"多辆汽车与摩托车"}}\n'
        "\n"
        "错误示例（非法 JSON，严禁输出）：\n"
        f'{{"x1":100,200,300,400}}  ← 省略了 y1/x2/y2 键名，不是合法 JSON\n'
        f'{{"x1":100,"y1":200,"x2":300}}  ← 缺少 y2 键名，不是合法 JSON\n'
    )


def request_vehicle_detection(api_config: dict, image_url: str, width: int, height: int, scope: str) -> tuple[dict, dict]:
    payload = build_payload(api_config, build_vehicle_detection_prompt(width, height, scope), image_url, 4096)
    parsed, data, content = post_vision_json(api_config, payload)
    return parsed, {"response": data, "content": content}


def call_vehicle_vision_model(
    image_path: Path,
    image_record_id: int,
    api_config: dict,
    draw_vehicle_boxes,
    analysis_prefix: str = "",
) -> tuple[int, str, str, int, int]:
    if not api_config["api_key"]:
        raise ValueError("未配置 API Key，请先设置大模型 API Key。")

    model_image, model_image_info = prepare_model_image_object(image_path)
    model_image_url, jpeg_bytes = pil_image_to_data_url(model_image)
    model_image_info = {**model_image_info, "jpeg_bytes": jpeg_bytes}
    model_image_width, model_image_height = model_image.size

    result, raw_log = request_vehicle_detection(api_config, model_image_url, model_image_width, model_image_height, "整张图片")
    log_payload = {"global": raw_log, "tiles": []}
    log_path = Path(save_model_response_log(image_path, api_config, log_payload, raw_log["content"], model_image_info))

    try:
        vehicle_count = int(result["vehicle_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"模型返回结果缺少有效 vehicle_count：{result}") from exc

    analysis = str(result.get("analysis") or "模型未返回说明。")
    boxes = result.get("boxes") or result.get("vehicles") or result.get("detections") or []
    if not isinstance(boxes, list):
        boxes = []
    if len(boxes) > vehicle_count:
        boxes = boxes[:vehicle_count]
    merged_boxes = convert_boxes_to_model_coords(boxes, model_image_width, model_image_height)

    tile_added = 0
    tile_errors = 0
    if ENABLE_TILE_DETECTION:
        for tile_index, (left, top, right, bottom) in enumerate(generate_tiles(model_image_width, model_image_height), start=1):
            crop = model_image.crop((left, top, right, bottom))
            tile_url, _ = pil_image_to_data_url(crop)
            try:
                tile_result, tile_log = request_vehicle_detection(api_config, tile_url, crop.width, crop.height, f"第 {tile_index} 个局部区域")
            except Exception as exc:
                tile_errors += 1
                log_payload["tiles"].append({"tile": [left, top, right, bottom], "error": str(exc)})
                continue

            tile_boxes = tile_result.get("boxes") or tile_result.get("vehicles") or tile_result.get("detections") or []
            if isinstance(tile_boxes, list) and len(merged_boxes) < vehicle_count:
                before_count = len(merged_boxes)
                merged_boxes = merge_boxes(merged_boxes + convert_boxes_to_model_coords(tile_boxes, crop.width, crop.height, left, top))
                merged_boxes = limit_boxes(merged_boxes, vehicle_count)
                tile_added += max(0, len(merged_boxes) - before_count)
            log_payload["tiles"].append({"tile": [left, top, right, bottom], "result": tile_result, "raw_content": tile_log["content"], "raw_response": tile_log["response"]})

        saved_log = json.loads(log_path.read_text(encoding="utf-8"))
        saved_log["raw_response"] = log_payload
        log_path.write_text(json.dumps(saved_log, ensure_ascii=False, indent=2), encoding="utf-8")
        if tile_added:
            analysis = f"{analysis}（分块补检新增 {tile_added} 个候选框。）"
        if tile_errors:
            analysis = f"{analysis}（有 {tile_errors} 个分块补检失败。）"

    result_name, drawn_count = draw_vehicle_boxes(image_path, merged_boxes, (model_image_width, model_image_height))
    if drawn_count != vehicle_count:
        analysis = f"{analysis}（注意：模型识别车辆数为 {vehicle_count}，合并后成功绘制的框为 {drawn_count} 个。）"
    if analysis_prefix:
        analysis = f"{analysis_prefix}{analysis}"
    analysis = f"{analysis}（原始返回已保存：{log_path.relative_to(LOG_DIR.parent).as_posix()}）"

    detection_result = DetectionResult(
        image_id=image_record_id,
        person_count=vehicle_count,
        bounding_boxes_json=json.dumps(merged_boxes, ensure_ascii=False),
        llm_analysis_text=analysis,
        result_image_path=f"static/results/{result_name}",
        llm_api_provider=api_config.get("provider", "vision_model_vehicle"),
        llm_model_name=api_config.get("model", ""),
        raw_llm_response_log_path=str(log_path),
    )
    db.session.add(detection_result)
    db.session.commit()

    return vehicle_count, analysis, result_name, model_image_width, model_image_height
