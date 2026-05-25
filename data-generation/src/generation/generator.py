"""
Generador de bios: combina OllamaClient + templates + validator.

Para cada Persona:
  1. Renderiza el prompt con el template asignado.
  2. Llama a Ollama con la temperatura asignada.
  3. Limpia y valida el output.
  4. Reintenta hasta MAX_RETRIES veces si el output no es válido,
     rotando el template y la temperatura en cada intento.
"""

import random
from typing import Optional, Tuple

from ..personas.sampler import Persona
from ..personas.templates import SYSTEM_PROMPT, TEMPLATES, render
from .ollama_client import OllamaClient
from .validator import clean

MAX_RETRIES = 3
# En cada retry alternamos la temperatura para salir del mínimo local
RETRY_TEMPERATURE_BUMP = 0.05


async def generate_bio(
    client: OllamaClient,
    persona: Persona,
    model: str,
) -> Tuple[Optional[str], int]:
    """
    Genera la bio para una persona.
    Devuelve (bio_limpia, intentos_usados) o (None, MAX_RETRIES) si falló todo.
    """
    template_id = persona.template_id
    temperature = persona.temperature

    for attempt in range(1, MAX_RETRIES + 1):
        prompt = render(template_id, persona)
        raw = await client.generate(
            model=model,
            system=SYSTEM_PROMPT,
            prompt=prompt,
            temperature=min(temperature, 1.05),
        )
        bio = clean(raw)
        if bio is not None:
            return bio, attempt

        # Rotar template y ajustar temperatura para el próximo intento
        template_id = random.choice(
            [t for t in TEMPLATES.keys() if t != template_id]
        )
        temperature = min(temperature + RETRY_TEMPERATURE_BUMP, 1.05)

    return None, MAX_RETRIES
