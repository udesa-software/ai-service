"""
Validación y limpieza del texto crudo devuelto por el modelo.

El modelo puede devolver markdown, comillas, texto en inglés,
outputs demasiado largos o demasiado cortos. Este módulo normaliza
todo eso antes de persistir la bio.
"""

import re
from typing import Optional

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

MIN_CHARS = 15
MAX_CHARS = 150
# Si truncamos y perdemos más del 40% del texto, preferimos re-generar
MAX_TRUNCATION_RATIO = 0.40


def clean(raw: str) -> Optional[str]:
    """
    Limpia y valida el output crudo del modelo.
    Devuelve la bio normalizada o None si debe descartarse/re-generarse.
    """
    text = raw.strip()

    # Rechazar formato pipe-separado antes de cualquier limpieza
    # (p.ej. "Mateo Díaz | 5to Año UdeSA | Rosario | rugby y piano")
    if " | " in text or text.count("|") >= 2:
        return None

    # Quitar comillas envolventes
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Quitar markdown: negrita, cursiva, encabezados, bullets
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-•*]\s+", "", text, flags=re.MULTILINE)

    # Quitar prefijos que el modelo a veces agrega
    prefixes = [
        "bio:", "perfil:", "sobre mí:", "presentación:",
        "bio de perfil:", "texto:", "aquí está", "aquí va",
        "claro,", "por supuesto,",
    ]
    text_lower = text.lower()
    for prefix in prefixes:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            text_lower = text.lower()
            break

    # Si tiene saltos de línea, quedarse con la primera línea no vacía
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        text = lines[0]

    # Truncar si excede MAX_CHARS
    if len(text) > MAX_CHARS:
        original_len = len(text)
        # Buscar el último punto/coma/! antes del límite
        cut_pos = MAX_CHARS
        for sep in (".", ",", "!", "?", ";"):
            pos = text.rfind(sep, 0, MAX_CHARS)
            if pos > MAX_CHARS * 0.6:  # al menos el 60% del texto
                cut_pos = pos + 1
                break
        truncated = text[:cut_pos].strip()
        lost_ratio = (original_len - len(truncated)) / original_len
        if lost_ratio > MAX_TRUNCATION_RATIO:
            return None  # se perdió demasiado, mejor re-generar
        text = truncated

    # Validar longitud mínima
    if len(text) < MIN_CHARS:
        return None

    # Validar idioma (requiere langdetect)
    if LANGDETECT_AVAILABLE:
        try:
            lang = detect(text)
            if lang not in ("es", "ca", "pt"):  # acepta español, catalán y portugués (falsos positivos comunes)
                return None
        except LangDetectException:
            pass  # texto muy corto para detectar — lo dejamos pasar

    return text


def is_valid(bio: str) -> bool:
    return clean(bio) is not None
