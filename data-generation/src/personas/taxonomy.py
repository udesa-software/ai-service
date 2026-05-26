"""
Taxonomía de atributos para generar personas ficticias.

Cada dimensión tiene valores suficientemente variados para que el
producto cruzado de combinaciones produzca bios semánticamente distintas.
"""

CAREERS = [
    "Ingeniería Industrial",
    "Ciencias de la Computación",
    "Economía",
    "Derecho",
    "Administración de Empresas",
    "Psicología",
    "Marketing",
    "Comunicación Social",
    "Relaciones Internacionales",
    "Arquitectura",
    "Filosofía",
    "Biología",
    "Finanzas",
    "Diseño Gráfico",
]

YEARS = [1, 2, 3, 4, 5]

PERSONALITIES = [
    "sociable",
    "introvertido y reflexivo",
    "ambicioso y enfocado",
    "relajado y espontáneo",
    "creativo y curioso",
    "analítico y metódico",
    "aventurero y arriesgado",
    "empático y colaborativo",
    "competitivo y apasionado",
    "tranquilo y observador",
]

ORIGINS = [
    "CABA",
    "GBA Norte (San Isidro / Vicente López)",
    "GBA Sur (Lomas / Quilmes)",
    "Córdoba capital",
    "Rosario",
    "Mendoza",
    "Tucumán",
    "Mar del Plata",
    "intercambio desde Brasil",
    "intercambio desde España",
    "intercambio desde México",
]

WRITING_STYLES = [
    # Reemplaza "formal y conciso" — "2am" evita registro corporativo rígido
    "directo y sin vueltas, como un mensaje de LinkedIn escrito a las 2am",

    # Reemplaza "informal con lunfardo porteño" — da vocabulario seguro, evita alucinaciones
    "muy coloquial porteño: usá 're', 'igual', 'piola', 'laburo', pero sin inventar lunfardo",

    # Reemplaza "con algunos emojis" — whitelist explícita, elimina caracteres CJK
    "coloquial con 1 o 2 emojis comunes (corazón, fuego, estrella, libro, pelota)",

    # Sin cambios — ya produce buenos resultados
    "storytelling en primera persona",

    # Reemplaza "listado directo de hechos" — elimina la semántica de pipes
    "descripción densa y compacta, como si tuvieras 2 líneas en una tarjeta de visita",

    # Sin cambios — ya produce buenos resultados
    "humorístico y autodescriptivo",

    # Sin cambios — ya produce buenos resultados
    "filosófico e introspectivo",

    # Sin cambios — ya produce buenos resultados
    "muy coloquial, estilo texto de WhatsApp",
]

# Intereses organizados por categoría para facilitar el muestreo diverso
INTERESTS_BY_CATEGORY = {
    "deporte": [
        "fútbol",
        "básquet",
        "tenis",
        "natación",
        "running",
        "crossfit",
        "rugby",
        "vóley",
        "paddle",
        "ciclismo",
        "surf",
        "escalada",
    ],
    "cultura_arte": [
        "fotografía",
        "cine independiente",
        "pintura",
        "teatro",
        "lectura",
        "escritura creativa",
        "museos",
        "cerámica",
    ],
    "musica": [
        "tocar guitarra",
        "producción musical",
        "cantar",
        "ir a recitales",
        "DJ",
        "piano",
        "cumbia",
        "jazz",
    ],
    "tech_gaming": [
        "videojuegos",
        "programar proyectos propios",
        "inteligencia artificial",
        "robótica",
        "3D printing",
        "hackathons",
    ],
    "social_lifestyle": [
        "cocinar",
        "viajar",
        "voluntariado",
        "política estudiantil",
        "emprendimiento",
        "meditación",
        "yoga",
        "senderismo",
        "café de especialidad",
        "astronomía",
        "podcasts",
    ],
}

# Apodo de la UdeSA que puede aparecer en el contexto de los prompts
UNIVERSITY_CONTEXT = "UdeSA (Universidad de San Andrés), Buenos Aires"

# Rangos de edad típicos por año
AGE_BY_YEAR = {
    1: (18, 19),
    2: (19, 20),
    3: (20, 21),
    4: (21, 22),
    5: (22, 26),
}

# Clichés a evitar explícitamente en los prompts (para reducir sesgos)
CLICHES_TO_AVOID = [
    # Originales
    "amante del mate",
    "fanático del asado",
    "vivir el momento",
    "apasionado/a de la vida",
    # Nuevos — frecuentes en bios genéricas de estudiantes
    "disfrutar cada instante",
    "busco mi mejor versión",
    "amo los viajes",
    "amante de la música",
    "soy una persona muy",
    "me apasiona aprender",
    "siempre con una sonrisa",
    "soñador/a",
    "full time estudiante",
    "café en mano",
    "sin filtros",
]

# Anti-patrones de formato — referenciados en SYSTEM_PROMPT y usables en validación futura
FORMAT_ANTIPATTERNS_TO_AVOID = [
    "pipe-separated fields (Nombre | Carrera | Ciudad)",
    "barra forms (apasionado/a, estudiante/a)",
    "Chinese or Japanese characters as emoji substitutes",
    "invented lunfardo conjugations",
]
