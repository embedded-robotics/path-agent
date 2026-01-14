ALIASES = {
    "head-neck": "head_neck",
    "gyn": "gynecologic",
    "gi": "gastrointestinal",
    "hem": "hematolymphoid",
}

def normalize(key: str) -> str:
    return ALIASES.get(key, key)
