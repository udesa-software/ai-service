"""
Prompt templates para la generación de biografías.

Cada template plantea la tarea de forma diferente Y usa un subconjunto
distinto de los campos disponibles (career, year, origin, interests,
personality). Esto fuerza al modelo a ser creativo en vez de enumerar
todos los atributos como si fuera un formulario.

Matriz de campos por template:
  T1: name, career, year, interests, style
  T2: name, origin, interests, style
  T3: name, year, interests, personality, style
  T4: name, career, interests, style
  T5: name, career, year, origin, style
  T6: name, interests, personality, style
  T7: name, career, origin, personality, style
  T8: name, interests, style

El campo {style} siempre está presente (es el principal lever de diversidad).
El campo {name} siempre está presente (da un anchor humano al modelo).
Ningún template usa los 7 campos opcionales a la vez.

La función render() pasa todos los campos via str.format(**kwargs);
Python ignora kwargs no usados en la cadena, por lo que no se necesita
ningún cambio en el llamador.
"""

from typing import Dict

# ─── System prompt ─────────────────────────────────────────────────────────────
#
# Técnicas de prompt engineering aplicadas (modelo objetivo: Ollama 32b / Qwen2.5):
#
# 1. Few-shot examples — los modelos 32B locales aprenden mejor de ejemplos
#    concretos que de instrucciones abstractas. 3 bios calibran registro y densidad.
#
# 2. Anti-patrones con nombre (ENUMERACIÓN, PIPE_LIST, CJK, LUNFARDO_INVENTADO) —
#    nombrar el anti-patrón crea una categoría que el modelo puede suprimir activamente.
#
# 3. "Menú, no checklist" — la instrucción explícita de elegir 1-2 rasgos rompe
#    el comportamiento de enumerar todos los campos del prompt.
#
# 4. Instrucción de truncación precisa — describe exactamente lo que hace el validator
#    (cortar en el último punto/coma antes de 140), evitando output que requiere
#    truncación agresiva.
#
# 5. Safe-harbor de lunfardo — "si no recordás la forma exacta, usá coloquial normal"
#    permite que el modelo se autoredireccione sin romper el registro informal.
#
# 6. Whitelist de emojis vs CJK — los modelos multilingüe codifican "piano" como
#    钢琴. La instrucción "nunca CJK" + "si dudás, no uses ninguno" es una constraint
#    de safety absoluta.
#
# 7. Mini-ejemplo MAL/BIEN inline — ver el contraste transformado es más efectivo
#    que solo describir la regla en abstracto.
#
# 8. Género sin barra — el modelo recibe solo el nombre, sin género explícito.
#    Sin instrucción, produce "apasionado/a" como hedge. La regla "elegí uno y
#    mantené consistencia" elimina ese comportamiento.
#
# 9. Headers === — los modelos Qwen/Llama siguen mejor instrucciones estructuradas
#    en secciones que prosa continua; los compartimentos de atención reducen que
#    una regla quede "enterrada".

SYSTEM_PROMPT = (
    "Escribís biografías de perfil para UdeSA-migos, una app social de la Universidad "
    "de San Andrés (UdeSA), Buenos Aires. Escribís bios cortas y auténticas para "
    "estudiantes ficticios.\n\n"

    "=== REGLAS DE FORMATO (obligatorias) ===\n"
    "- Tu output ES el texto de la bio. Nada más: sin comillas, sin guiones, sin "
    "'Aquí va:', sin 'Bio:', sin explicaciones antes ni después.\n"
    "- Máximo 140 caracteres. Si el borrador supera ese límite, cortá en el último "
    "punto o coma antes de la posición 140, nunca a mitad de frase.\n"
    "- Siempre en español. Usá vos/sos/tenés. No uses tú/usted.\n"
    "- Elegí un género gramatical y mantené consistencia. Nunca escribas 'apasionado/a' "
    "ni formas con barra.\n\n"

    "=== INTEGRACIÓN DE INFORMACIÓN ===\n"
    "El prompt que recibís lista atributos de la persona. Esos atributos son un menú "
    "para elegir, NO una lista para completar. Tomá 1 o 2 rasgos y tejelos en la "
    "bio de forma natural. Ignorar algunos atributos es correcto y deseable.\n"
    "- MAL (enumeración): 'Estudia Arquitectura, 4to año, le gusta el 3D printing, "
    "la política estudiantil y escribir. Es creativo.'\n"
    "- BIEN (integrado): '4to año Arq. UdeSA. Obsesionado con el 3D printing; "
    "entre modelos y maquetas se me va la vida.'\n\n"

    "=== LUNFARDO Y COLOQUIALISMO ===\n"
    "Podés usar vocabulario coloquial porteño natural: 'laburo', 'piola', 're', "
    "'igual', 'fiaca', 'copado'. "
    "Evitá inventar lunfardo que no existe o conjugar incorrectamente. "
    "Si no recordás la forma exacta de una expresión, usá español coloquial "
    "argentino normal: es mejor claro que inventado.\n\n"

    "=== EMOJIS ===\n"
    "Si el estilo lo pide, usá emojis Unicode estándar. "
    "Nunca uses caracteres chinos, japoneses ni coreanos (CJK). "
    "Si no estás seguro del emoji correcto para algo, no uses ninguno.\n\n"

    "=== CLICHÉS PROHIBIDOS ===\n"
    "Nunca uses estas frases ni sus variaciones: 'amante del mate', 'fanático del "
    "asado', 'vivir el momento', 'apasionado/a de la vida', 'disfrutar cada instante', "
    "'busco mi mejor versión', 'amo los viajes', 'amante de la música', "
    "'me apasiona aprender', 'siempre con una sonrisa', 'soñador/a', "
    "'café en mano', 'sin filtros'.\n\n"

    "=== EJEMPLOS DE BIOS BUENAS ===\n"
    "Usá estos ejemplos para calibrar tono y densidad. No los copies:\n"
    "1. 'Catu acá, de Sanbo y Vilón. En UdeSA diseñando la vida. Pinto cuadros, "
    "medito, pego tenis y mato en los vidios. Siempre para adelante, sin gambetas.'\n"
    "2. 'En busca del equilibrio entre números y notas. Escalador de paredes y "
    "debates. Los museos me inspiran tanto como la política estudiantil.'\n"
    "3. '4to año Arq. UdeSA. De Lomas. Voy por la política estudiantil, el 3D "
    "printing me vuelve loco y escribo cuando el teclado calla.'\n\n"

    "=== BIOS MALAS (no hagas esto) ===\n"
    "A) ENUMERACIÓN: 'Estudiante de Cs de la Computación 5to año. Amante de los "
    "recitales, el 3D printing y los podcasts. Sociable y conectado.' "
    "→ Copia el prompt sin personalidad.\n"
    "B) PIPE_LIST: 'Mateo Díaz | 5to Año Cs Computación UdeSA | Rosario | rugby y piano' "
    "→ Formulario, no una bio.\n"
    "C) CARACTERES_CJK: '📸钢琴🎨 marketólogo en formación' "
    "→ Nunca caracteres chinos.\n"
    "D) LUNFARDO_INVENTADO: 'toco saxo cuando el alma me pedí' "
    "→ Conjugación incorrecta. Mejor: 'toco saxo cuando el ánimo me da'.\n"
    "E) TODO_LOS_CAMPOS: cualquier bio que menciona nombre + carrera + año + origen "
    "+ intereses + personalidad en la misma oración. Elegí 1-2 dimensiones, no todo.\n"
)

# ─── Templates de usuario ──────────────────────────────────────────────────────
#
# Principio: cada template usa un subconjunto diferente de campos (ver matriz arriba).
# Las instrucciones inline en cada template son refuerzos contextualizados: actúan
# en el momento exacto en que el modelo ve los datos específicos, lo que es más
# efectivo que solo las reglas generales del system prompt.

TEMPLATES: Dict[int, str] = {

    # T1: career + year + interests — académico con pivot hacia pasión
    # "elegí uno y desarrollalo" suprime la enumeración directamente en el template
    1: (
        "Escribí la bio de perfil de {name} en UdeSA-migos. "
        "Estudia {career}, va por {year}° año. "
        "Entre sus intereses: {interests}. "
        "Estilo de escritura: {style}. "
        "No menciones todos los intereses: elegí uno y desarrollalo."
    ),

    # T2: origin + interests — lugar y pasión, sin datos académicos
    # La instrucción negativa explícita es más fuerte que omitir el campo
    2: (
        "Escribí la presentación de {name} para UdeSA-migos. "
        "Viene de {origin}. Lo que más le mueve: {interests}. "
        "Estilo: {style}. "
        "No menciones carrera ni año: que la bio gire en torno a quién es, no qué estudia."
    ),

    # T3: year + interests + personality — año como contexto, personalidad como textura
    # Pide que la personalidad se note en el *cómo*, no solo mencionándola
    3: (
        "Creá la bio de {name} para una app de amigos universitarios en UdeSA. "
        "Va por {year}° año. Es {personality}. Entre sus pasiones: {interests}. "
        "Tono: {style}. "
        "Que la personalidad se note en cómo está escrita, no solo describiéndola."
    ),

    # T4: career + interests — tensión o vínculo entre carrera y hobbies
    # La constraint creativa fuerza síntesis en vez de listado paralelo
    4: (
        "Redactá la bio de {name} para UdeSA-migos. "
        "Carrera: {career}. Sus intereses fuera de la facu: {interests}. "
        "Estilo: {style}. "
        "Mostrá la tensión o el vínculo entre lo que estudia y lo que hace afuera."
    ),

    # T5: career + year + origin — puramente contextual, sin hobbies listados
    # Fuerza al modelo a construir identidad desde lugar + carrera
    5: (
        "Generá una bio corta para {name} en UdeSA-migos. "
        "Estudia {career}, {year}° año. Es de {origin}. "
        "Estilo: {style}. "
        "No listes hobbies: que el lugar de origen y la carrera cuenten la historia."
    ),

    # T6: interests + personality — mínimo de datos, máxima libertad creativa
    # "Ignorá todo lo que no sepas" da permiso explícito para inventar contexto
    6: (
        "Escribí en {style} la bio de {name} para UdeSA-migos. "
        "Información disponible: le interesan {interests} y es {personality}. "
        "Ignorá todo lo que no sepas. Que suene a persona real, no a ficha."
    ),

    # T7: career + origin + personality — académico + lugar + carácter, sin hobbies
    # Identidad desde origen y temperamento, sin depender de intereses específicos
    7: (
        "Armá la bio de {name} ({career}) para UdeSA-migos. "
        "Viene de {origin}. En pocas palabras, es {personality}. "
        "Estilo: {style}. "
        "No menciones hobbies: que la combinación de origen y carácter genere la bio."
    ),

    # T8: interests only + invención explícita — máxima creatividad del modelo
    # "Inventá el resto" es el frame más generativo; rinde mejor con temperature alta
    8: (
        "Solo sabés esto de {name}: le apasiona {interests}. "
        "Escribí su bio para UdeSA-migos con un tono {style}. "
        "Inventá el resto: que suene a estudiante real de Buenos Aires."
    ),
}

N_TEMPLATES = len(TEMPLATES)


def render(template_id: int, persona) -> str:
    """Renderiza el prompt de usuario para una persona dada."""
    template = TEMPLATES[template_id]
    return template.format(
        name=persona.name,
        career=persona.career,
        year=persona.year,
        origin=persona.origin,
        interests=persona.interests_str(),
        personality=persona.personality,
        style=persona.writing_style,
    )
