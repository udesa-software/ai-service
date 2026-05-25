"""
Cliente async para la API de Ollama.

Ollama expone un servidor HTTP en localhost:11434.
Usamos aiohttp para hacer llamadas concurrentes sin bloquear el event loop.
"""

import asyncio
import json
from typing import Optional

import aiohttp

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=300)  # 5 min para modelos grandes (32b ~2-3 min/bio)


class OllamaError(Exception):
    pass


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT)
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    async def generate(
        self,
        model: str,
        system: str,
        prompt: str,
        temperature: float = 0.8,
        seed: Optional[int] = None,
    ) -> str:
        """
        Llama a /api/chat con roles system/user y devuelve el contenido
        del mensaje assistant como string.
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 80,   # ~140 chars en español suelen ser <80 tokens
                "top_p": 0.95,
            },
        }
        if seed is not None:
            payload["options"]["seed"] = seed

        try:
            async with self._session.post(
                f"{self.base_url}/api/chat", json=payload
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise OllamaError(f"HTTP {resp.status}: {body}")
                data = await resp.json()
                return data["message"]["content"].strip()
        except aiohttp.ClientConnectorError as e:
            raise OllamaError(
                f"No se pudo conectar a Ollama en {self.base_url}. "
                "¿Está corriendo `ollama serve`?"
            ) from e

    async def list_models(self) -> list[str]:
        """Devuelve los nombres de los modelos descargados."""
        async with self._session.get(f"{self.base_url}/api/tags") as resp:
            data = await resp.json()
            return [m["name"] for m in data.get("models", [])]

    @staticmethod
    async def check_model(model: str, base_url: str = OLLAMA_BASE_URL) -> None:
        """
        Verifica que Ollama esté corriendo y el modelo esté disponible.
        Lanza OllamaError con instrucciones claras si algo falla.
        """
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            try:
                async with s.get(f"{base_url}/api/tags") as resp:
                    data = await resp.json()
                    available = [m["name"] for m in data.get("models", [])]
            except aiohttp.ClientConnectorError:
                raise OllamaError(
                    "Ollama no está corriendo. Ejecutá: ollama serve"
                )

        # Ollama puede guardar el modelo como "qwen2.5:14b" o "qwen2.5:14b-..."
        if not any(model in name for name in available):
            raise OllamaError(
                f"Modelo '{model}' no encontrado. "
                f"Modelos disponibles: {available}\n"
                f"Para descargarlo: ollama pull {model}"
            )
