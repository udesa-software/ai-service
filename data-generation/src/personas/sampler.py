"""
Muestreo estratificado de personas ficticias.

Garantiza cobertura uniforme de carreras y estilos de escritura
antes de muestrear aleatoriamente el resto de los atributos.
"""

import random
import json
import unicodedata
from dataclasses import dataclass, asdict
from typing import List

from faker import Faker

from .taxonomy import (
    CAREERS,
    YEARS,
    PERSONALITIES,
    ORIGINS,
    WRITING_STYLES,
    INTERESTS_BY_CATEGORY,
    AGE_BY_YEAR,
)

fake = Faker("es_AR")


def _normalize_username(raw: str) -> str:
    """
    Convierte una cadena cruda en un username válido para el microservicio users:
    - Solo caracteres alfanuméricos (sin espacios, guiones, diéresis, apóstrofes, etc.)
    - Máximo 15 caracteres
    - Minúsculas

    Estrategia: descompone Unicode (NFD) para separar los caracteres base de sus
    diacríticos (e.g. é → e + combining accent), descarta todo lo que no sea ASCII
    y luego filtra para quedarse solo con alfanuméricos.
    """
    nfd = unicodedata.normalize("NFD", raw)
    ascii_only = nfd.encode("ascii", "ignore").decode("ascii")
    alnum = "".join(c for c in ascii_only if c.isalnum())
    return alnum.lower()[:15]


@dataclass
class Persona:
    name: str
    username: str
    email: str
    age: int
    career: str
    year: int
    personality: str
    origin: str
    writing_style: str
    interests: List[str]  # 2–4 items de distintas categorías
    template_id: int
    temperature: float

    def to_dict(self) -> dict:
        return asdict(self)

    def interests_str(self) -> str:
        return ", ".join(self.interests)


def _sample_interests(n: int = 3) -> List[str]:
    """
    Toma n intereses de distintas categorías para maximizar variedad.
    """
    categories = list(INTERESTS_BY_CATEGORY.keys())
    chosen_cats = random.sample(categories, min(n, len(categories)))
    interests = [random.choice(INTERESTS_BY_CATEGORY[cat]) for cat in chosen_cats]
    # Si piden más que las categorías disponibles, completa al azar
    all_interests = [i for cat in INTERESTS_BY_CATEGORY.values() for i in cat]
    while len(interests) < n:
        candidate = random.choice(all_interests)
        if candidate not in interests:
            interests.append(candidate)
    return interests


def _make_persona(
    career: str,
    writing_style: str,
    template_id: int,
    temperature: float,
) -> Persona:
    year = random.choice(YEARS)
    age_min, age_max = AGE_BY_YEAR[year]
    age = random.randint(age_min, age_max)

    first = fake.first_name()
    last = fake.last_name()
    name = f"{first} {last}"
    username = _normalize_username(f"{first}{last}{random.randint(10, 99)}")
    email = fake.email()

    n_interests = random.randint(2, 4)

    return Persona(
        name=name,
        username=username,
        email=email,
        age=age,
        career=career,
        year=year,
        personality=random.choice(PERSONALITIES),
        origin=random.choice(ORIGINS),
        writing_style=writing_style,
        interests=_sample_interests(n_interests),
        template_id=template_id,
        temperature=temperature,
    )


def generate_personas(
    count: int,
    n_templates: int,
    temperatures: List[float],
) -> List[Persona]:
    """
    Genera `count` personas con muestreo estratificado sobre
    carreras y estilos de escritura.

    Garantiza que cada carrera y cada estilo aparezca al menos
    floor(count / n_values) veces antes de rellenar al azar.
    """
    personas: List[Persona] = []

    # 1. Muestreo estratificado: asegura al menos 1 rep. de cada carrera × estilo
    guaranteed: List[tuple] = []
    for career in CAREERS:
        for style in WRITING_STYLES:
            guaranteed.append((career, style))

    # Cuántas veces queremos garantizar cada combinación carrera×estilo
    reps_per_combo = max(1, count // len(guaranteed))
    base_list = guaranteed * reps_per_combo

    # Completar hasta `count` con muestras aleatorias
    while len(base_list) < count:
        base_list.append(
            (random.choice(CAREERS), random.choice(WRITING_STYLES))
        )

    random.shuffle(base_list)
    base_list = base_list[:count]

    for career, style in base_list:
        template_id = random.randint(1, n_templates)
        temperature = random.choice(temperatures)
        personas.append(_make_persona(career, style, template_id, temperature))

    return personas
