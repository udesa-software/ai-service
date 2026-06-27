from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkUser:
    id: str
    username: str
    biography: str
    category: str
    expected_top_ids: tuple[str, ...]


BENCHMARK_USERS: tuple[BenchmarkUser, ...] = (
    BenchmarkUser(
        id="user-ai-1",
        username="sofia",
        biography="Estudio inteligencia artificial, NLP y machine learning. Me encanta investigar modelos de lenguaje y programar en Python.",
        category="AI",
        expected_top_ids=("user-ai-2", "user-ai-3", "user-startup-1"),
    ),
    BenchmarkUser(
        id="user-ai-2",
        username="tomas",
        biography="Trabajo con ciencia de datos, deep learning y procesamiento de lenguaje natural. Participo en proyectos de IA aplicada.",
        category="AI",
        expected_top_ids=("user-ai-1", "user-ai-3", "user-startup-1"),
    ),
    BenchmarkUser(
        id="user-ai-3",
        username="valen",
        biography="Me apasiona el análisis de datos, el aprendizaje automático y construir productos con modelos de IA para educación.",
        category="AI",
        expected_top_ids=("user-ai-1", "user-ai-2", "user-startup-1"),
    ),
    BenchmarkUser(
        id="user-art-1",
        username="clara",
        biography="Escribo cuentos, voy al cine independiente y dibujo. Me interesan la literatura, el arte visual y la fotografía.",
        category="Art",
        expected_top_ids=("user-art-2", "user-art-3", "user-music-1"),
    ),
    BenchmarkUser(
        id="user-art-2",
        username="juan",
        biography="Amo la escritura creativa, la poesía, las novelas y las muestras de arte. También saco fotos analógicas.",
        category="Art",
        expected_top_ids=("user-art-1", "user-art-3", "user-music-1"),
    ),
    BenchmarkUser(
        id="user-art-3",
        username="lola",
        biography="Paso horas entre museos, fotografía callejera y talleres de dibujo. Me gustan el cine de autor y las historias visuales.",
        category="Art",
        expected_top_ids=("user-art-1", "user-art-2", "user-music-1"),
    ),
    BenchmarkUser(
        id="user-sports-1",
        username="mati",
        biography="Juego al fútbol, entreno running y voy al gimnasio. Me copan los deportes de equipo y la vida activa.",
        category="Sports",
        expected_top_ids=("user-sports-2", "user-sports-3", "user-social-1"),
    ),
    BenchmarkUser(
        id="user-sports-2",
        username="agus",
        biography="Hago natación, trail running y entrenamiento funcional. Siempre me sumo a planes deportivos y al aire libre.",
        category="Sports",
        expected_top_ids=("user-sports-1", "user-sports-3", "user-social-1"),
    ),
    BenchmarkUser(
        id="user-sports-3",
        username="nacho",
        biography="Me gusta el gimnasio, el tenis y organizar partidos con amigos. Soy fan de entrenar y competir.",
        category="Sports",
        expected_top_ids=("user-sports-1", "user-sports-2", "user-social-1"),
    ),
    BenchmarkUser(
        id="user-music-1",
        username="mora",
        biography="Toco guitarra, voy a recitales y produzco canciones. Me interesan la música indie, el audio y la composición.",
        category="Music",
        expected_top_ids=("user-music-2", "user-art-1", "user-art-2"),
    ),
    BenchmarkUser(
        id="user-music-2",
        username="fran",
        biography="Compongo, ensayo con mi banda y escucho discos todo el día. Me gusta producir música y descubrir artistas nuevos.",
        category="Music",
        expected_top_ids=("user-music-1", "user-art-1", "user-art-2"),
    ),
    BenchmarkUser(
        id="user-startup-1",
        username="bruno",
        biography="Me interesan los startups, producto, negocios y tecnología. Disfruto programar, validar ideas y armar proyectos con impacto.",
        category="Startup",
        expected_top_ids=("user-ai-1", "user-ai-2", "user-ai-3"),
    ),
    BenchmarkUser(
        id="user-social-1",
        username="cata",
        biography="Me encantan los planes con gente, organizar salidas, viajar y conocer personas nuevas. Siempre estoy para actividades grupales.",
        category="Social",
        expected_top_ids=("user-sports-1", "user-sports-2", "user-sports-3"),
    ),
)
