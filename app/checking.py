from app.schemas import TermHit

def normalize(text: str) -> str:
    return "".join(text.split())

def find_missing_terms(translation: str, must_use: list[TermHit]) -> list[TermHit]:
    normalized_translation = normalize(translation)
    missing: list[TermHit] = []
    for term in must_use:
        if normalize(term.target) not in normalized_translation:
            missing.append(term)

    return missing

def adherence_rate(must_use: list[TermHit], missing: list[TermHit]) -> float:
    """术语遵循率，用上的必须使用术语占全部的比例。"""
    if not must_use:
        return 1.0
    used = len(must_use) - len(missing)
    return used / len(must_use)

if __name__ == "__main__":
    must_use = [
        TermHit(source="power button", target="开关键", match_type="exact"),
        TermHit(source="control hub", target="多功能网关", match_type="exact"),
        TermHit(source="usb cable", target="USB数据线", match_type="exact"),
    ]

    examples = [
        "长按开关键以重启多功能网关，并检查 USB 数据线。",
        "长按电源按钮以重启多功能网关，并检查 usb数据线。",
        "长按电源按钮以重启控制中枢。",
    ]

    for translation in examples:
        missing = find_missing_terms(translation=translation, must_use=must_use)
        rate = adherence_rate(missing=missing, must_use=must_use)
        print(translation)
        print("  漏掉：", [term.target for term in missing])
        print(f"  遵循率：{rate:.0%}")