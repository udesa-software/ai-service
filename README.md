# ai-service — UdeSA-migos

Repositorio del microservicio de Inteligencia Artificial para UdeSA-migos.

## Feature: Bio Matcher

Recomienda amigos basándose en similitud semántica de biografías (al estilo Tinder), como portal adicional al discovery por cercanía ya implementado.

## Estructura

```
ai-service/
├── data-generation/   # Generación de corpus sintético de bios (fase actual)
├── embeddings/        # Fine-tuning y evaluación de modelos de embeddings
└── api/               # FastAPI: matching en tiempo real para la app
```

## Estado actual

- [x] `data-generation`: pipeline completo para generar ~5K-15K bios ficticias con Ollama (local, gratuito)
- [ ] `embeddings`: entrenamiento del modelo de embeddings con el corpus
- [ ] `api`: endpoint de matching que conecta con el `users service` de AWS

## Setup rápido (data-generation)

Ver [data-generation/README.md](data-generation/README.md).

## Setup rápido (api)

El servicio `api` es Python. Usar un virtualenv propio del repo evita mezclar dependencias con otros servicios.

```bash
cd ai-service/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para correr tests:

```bash
pytest
pytest tests/integration
pytest --cov=src --cov-report=xml
```

Si el prompt muestra un `.venv` de otro repo, por ejemplo `notifications/.venv`, desactivalo antes:

```bash
deactivate
cd ai-service/api
source .venv/bin/activate
pytest
```
