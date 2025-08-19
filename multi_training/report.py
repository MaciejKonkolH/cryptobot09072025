from datetime import datetime
from pathlib import Path
import json


def save_json_report(evaluation_results: dict, model_params: dict, data_info: dict, out_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = out_dir / f"results_MULTI_{ts}.json"
    payload = {
        "symbols": data_info.get("symbols"),
        "generated_at": datetime.now().isoformat(),
        "model_params": model_params,
        "data_info": data_info,
        "evaluation_results": evaluation_results,
    }
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out

