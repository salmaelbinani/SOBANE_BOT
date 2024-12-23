import os
import io
from PIL import Image
from pyrogram import Client, filters
from pyrogram.enums import ParseMode
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from collections import defaultdict
import json
import pandas as pd
import re

# Configuration
API_ID = "21788039"
API_HASH = "7a23778bd249ba47c6649136dc4fe5cb"
BOT_TOKEN = "7794252893:AAEnfZCoggvQ4RYJT9FMwGowW0glY2yq02o"
GENAI_API_KEY = "AIzaSyDhE2dZc3lFIeVgmArba2zhDkM8EC7kOZY"
VECTOR_DB_PATH = "VectorDB"
SOURCE_DATA_FOLDER = "SobaneData"

# Initialize folders
os.makedirs(SOURCE_DATA_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Initialize bot
app = Client("sobane_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

# Configure Gemini AI
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# SOBANE facettes structure
SOBANE_FACETTES = {
    1: "Locaux et zones de travail",
    2: "Organisation du travail",
    3: "Accidents de travail",
    4: "Risques électriques",
    5: "Risques d'incendie ou d'explosion",
    6: "Commandes et signaux",
    7: "Matériel de travail",
    8: "Positions de travail",
    9: "Efforts et manutentions",
    10: "Éclairage",
    11: "Bruit",
    12: "Ambiances thermiques",
    13: "Risques chimiques et biologiques",
    14: "Vibrations",
    15: "Relations de travail",
    16: "Environnement psychosocial",
    17: "Contenu du travail",
    18: "Contraintes de temps"
}

# User state tracking
user_states = defaultdict(lambda: {"current_facette": 1, "responses": {}, "current_question": 1, "facette_analyses": {}, "facette_global_points": {}})

# Prompt template for image analysis
IMAGE_ANALYSIS_TEMPLATE = """
Analyser cette image dans le contexte de la question: {question}

Veuillez fournir:
1. Une description détaillée de l'image
2. Réponse à la question basée sur l'analyse de l'image
3. Observations pertinentes pour l'évaluation SOBANE

Réponse structurée:
- Description de l'image: 
- Réponse à la question: 
- Observations SOBANE: 
"""

# Updated Facette Analysis Template with Severity Assessment
FACETTE_ANALYSIS_TEMPLATE = """
Vous êtes un expert en évaluation SOBANE pour la facette {facette_name} ({facette_number}/18).

Contexte des réponses utilisateur:
{user_responses}

Analysez ces réponses et générez:
1. Une évaluation détaillée des points positifs observés
2. Une évaluation détaillée des points negatifs observés
3. Les aspects nécessitant une étude plus approfondie
4. Des recommandations immédiates si nécessaire
5. IMPORTANT: Évaluez la gravité globale de la situation et proposez un emoji approprié selon ces critères:
   - 😀 (Situation tout à fait satisfaisante): Aucun risque significatif, conditions optimales
   - 😐 (Situation moyenne, à améliorer): Quelques points mineurs à améliorer, pas de danger immédiat
   - 😡 (Situation insatisfaisante, potentiellement dangereuse): Problèmes critiques nécessitant une intervention urgente

Votre évaluation globale doit être basée sur une analyse rigoureuse et objective, en pesant l'importance relative des points positifs et négatifs.

Format de sortie souhaité:

## Évaluation des points positifs
[Votre évaluation détaillée]

## Évaluation des points negatifs
[Votre évaluation détaillée]

## Points à approfondir
- Point 1
- Point 2
...

## Recommandations
- [Première recommandation]
- [Deuxième recommandation]
...

## Évaluation de la gravité globale et emoji recommandé
[Justification détaillée de l'emoji choisi]
"""

# Questions for each facette
FACETTE_QUESTIONS = {
    1: [
        "Les locaux et zones de travail sont-ils adaptés aux tâches à réaliser et à la sécurité des employés?"
    ],
    2: [
        "L'organisation du travail permet-elle d'éviter des surcharges ou des conflits dans les tâches?"
    ],
    3: [
        "Y a-t-il des mesures en place pour prévenir les accidents de travail et en limiter les conséquences?"
    ],
    4: [
        "Les risques électriques sont-ils identifiés et des protections adéquates sont-elles en place?"
    ],
    5: [
        "Les installations préviennent-elles efficacement les risques d'incendie ou d'explosion?"
    ],
    6: [
        "Les commandes et signaux sont-ils facilement compréhensibles et accessibles?"
    ],
    7: [
        "Le matériel de travail est-il bien entretenu et adapté aux activités?"
    ],
    8: [
        "Les postes de travail permettent-ils de maintenir des positions ergonomiques?"
    ],
    9: [
        "Les efforts physiques et les tâches de manutention sont-ils gérés pour éviter les troubles musculosquelettiques?"
    ],
    10: [
        "L'éclairage des zones de travail est-il suffisant et adapté aux besoins visuels?"
    ],
    11: [
        "Le bruit sur le lieu de travail est-il contrôlé pour éviter des nuisances ou des risques auditifs?"
    ],
    12: [
        "Les ambiances thermiques sont-elles confortables et adaptées aux conditions de travail?"
    ],
    13: [
        "Les risques chimiques et biologiques sont-ils identifiés et maîtrisés?"
    ],
    14: [
        "Les vibrations des équipements ou machines sont-elles contrôlées pour éviter des impacts sur la santé?"
    ],
    15: [
        "Les relations de travail sont-elles harmonieuses et favorisent-elles la coopération?"
    ],
    16: [
        "L'environnement psychosocial est-il sain et exempt de stress ou de conflits excessifs?"
    ],
    17: [
        "Le contenu du travail est-il motivant et adapté aux compétences des travailleurs?"
    ],
    18: [
        "Les contraintes de temps permettent-elles un travail de qualité sans pression excessive?"
    ],
}


def generate_facette_report(state, facette_number):
    markdown_content = "# Rapport d'évaluation SOBANE\n\n"

    # Détails des facettes
    markdown_content += "## Détails des facettes\n\n"
    markdown_content += "| Facette | Question | Réponse | Points Positifs | Points Négatifs |\n"
    markdown_content += "|---------|----------|---------|----------------|------------------|\n"

    analysis = state["facette_analyses"][facette_number]
    parsed_analysis = parse_analysis(analysis)

    for response in state["responses"][facette_number]:
        markdown_content += f"| {facette_number}. {SOBANE_FACETTES[facette_number]} | {response['question']} | {response['response']} | "
        
        # Add points positifs
        if parsed_analysis['points_positifs']:
            markdown_content += "<br>".join(f"- {point}" for point in parsed_analysis['points_positifs'])
        markdown_content += " | "
        
        # Add points negatifs
        if parsed_analysis['points_negatifs']:
            markdown_content += "<br>".join(f"- {point}" for point in parsed_analysis['points_negatifs'])
        markdown_content += " |\n"

    # Aspects to study further and emoji
    markdown_content += "\n## Synthèse globale\n\n"
    markdown_content += "| Aspects à approfondir | Évaluation globale |\n"
    markdown_content += "|------------------------|--------------------|\n"

    # Insérer un saut de ligne entre chaque point
    markdown_content += f"| {'<br>'.join(parsed_analysis['points_a_approfondir'])} | {parsed_analysis['severity_emoji']} |\n"

    # Recommendations
    markdown_content += "\n## Recommandations\n\n"
    for rec in parsed_analysis['recommandations']:
        markdown_content += f"{rec}\n"

    return markdown_content, parsed_analysis['severity_emoji']


def create_final_summary_report(user_states, user_id):
    state = user_states[user_id]
    
    markdown_content = "# Rapport Final SOBANE\n\n"
    
    # Première partie : Tableau récapitulatif des facettes
    markdown_content += "## Récapitulatif des Facettes\n\n"
    markdown_content += "| Numéro | Nom de la Facette | Évaluation |\n"
    markdown_content += "|--------|-------------------|------------|\n"
    
    # Collecter les recommandations uniques
    all_unique_recommendations = []
    
    for facette_num, analysis in state["facette_analyses"].items():
        parsed_analysis = parse_analysis(analysis)
        
        # Ajouter une ligne au tableau récapitulatif
        markdown_content += f"| {facette_num} | {SOBANE_FACETTES[facette_num]} | {parsed_analysis['severity_emoji']} |\n"
        
        # Collecter les recommandations
        all_unique_recommendations.extend(parsed_analysis['recommandations'])
    
    # Deuxième partie : Tableau des recommandations
    markdown_content += "\n## Plan d'Action Détaillé\n\n"
    markdown_content += "| N° | Qui ? | Fait quoi et comment? | Coût | Projeté Quand ? | Réalisé Quand ? |\n"
    markdown_content += "|----|----|----------------|------|-----------------|------------------|\n"
    
    # Ajouter les recommandations au tableau
    for i, rec in enumerate(set(all_unique_recommendations), 1):
        markdown_content += f"| {i} |  | {rec} |  |  |  |\n"
    
    return markdown_content
def not_command(_, __, message):
    return not message.text.startswith('/')

# Custom filters
text_filter = filters.create(not_command) & filters.text
photo_filter = filters.photo

async def analyze_image(image_path, question):
    try:
        # Open image with PIL
        image = Image.open(image_path)
        
        prompt = IMAGE_ANALYSIS_TEMPLATE.format(question=question)
        
        response = model.generate_content(
            [prompt, image],
            generation_config={"max_output_tokens": 1024}
        )
        return response.text
    except Exception as e:
        print(f"Detailed error: {e}")
        return f"Erreur détaillée lors de l'analyse de l'image: {str(e)}"

@app.on_message(filters.command("start"))
async def start_command(client, message):
    user_id = message.from_user.id
    user_states[user_id] = {
        "current_facette": 1, 
        "responses": {}, 
        "current_question": 1, 
        "facette_analyses": {}, 
        "facette_global_points": {}
    }
    
    welcome_text = f"""
Bienvenue dans l'assistant d'évaluation SOBANE!

Je vais vous guider à travers l'évaluation des 18 facettes de la méthode SOBANE.
Nous commencerons par la première facette: {SOBANE_FACETTES[1]}

{FACETTE_QUESTIONS[1][0]}

Vous pouvez répondre par du texte ou une image. 
Utilisez /help pour voir toutes les commandes disponibles.
"""
    await message.reply_text(welcome_text)

@app.on_message(filters.command("help"))
async def help_command(client, message):
    help_text = """
Commandes disponibles:
/start - Démarrer ou recommencer l'évaluation
/status - Voir votre progression actuelle
/report - Générer un rapport de l'évaluation actuelle
/reset - Réinitialiser l'évaluation

Répondez simplement aux questions par du texte ou des images.
"""
    await message.reply_text(help_text)

@app.on_message(photo_filter)
async def handle_image_response(client, message):
    user_id = message.from_user.id
    if user_id not in user_states:
        await message.reply_text("Veuillez commencer l'évaluation avec /start")
        return

    state = user_states[user_id]
    current_facette = state["current_facette"]
    current_question = FACETTE_QUESTIONS[current_facette][state["current_question"] - 1]

    # Download image
    image_path = await message.download()

    try:
        # Analyze image
        image_analysis = await analyze_image(image_path, current_question)
        
        # Store the response
        facette_responses = state["responses"].get(current_facette, [])
        facette_responses.append({
            "question": current_question,
            "response": "Analyse d'image",
            "image_path": image_path,
            "image_analysis": image_analysis,
            "type": "image"
        })
        state["responses"][current_facette] = facette_responses

        # Process image analysis when last question is answered
        if state["current_question"] >= len(FACETTE_QUESTIONS[current_facette]):
            try:
                analysis = await generate_analysis(
                    current_facette,
                    str(state["responses"][current_facette])
                )
                state["facette_analyses"][current_facette] = analysis
                state["facette_global_points"][current_facette] = parse_analysis(analysis)
                
                # Finish this facette message
                finish_facette_text = f"""
✅ Vous avez terminé la facette {current_facette}: {SOBANE_FACETTES[current_facette]}

Options disponibles:
- /facette_report - Télécharger le rapport de cette facette
- /report - Télécharger le rapport final à la fin
- /reset - Recommencer l'évaluation
"""
                await message.reply_text(finish_facette_text)
                
                # Move to next facette
                state["current_facette"] += 1
                state["current_question"] = 1
                
                if state["current_facette"] <= 18:
                    next_question = FACETTE_QUESTIONS[state["current_facette"]][0]
                    await message.reply_text(f"Passons à la facette {state['current_facette']}: {SOBANE_FACETTES[state['current_facette']]}\n\n{next_question}")
                else:
                    await message.reply_text("Évaluation terminée! Utilisez /report pour générer le rapport final.")
            except Exception as e:
                await message.reply_text(f"Une erreur est survenue: {str(e)}")
        else:
            # Move to next question in current facette
            state["current_question"] += 1
            next_question = FACETTE_QUESTIONS[current_facette][state["current_question"] - 1]
            await message.reply_text(next_question)
        
        # Remove temporary image file
        os.remove(image_path)

    except Exception as e:
        await message.reply_text(f"Erreur lors du traitement de l'image: {str(e)}")

@app.on_message(text_filter)
async def handle_text_response(client, message):
    user_id = message.from_user.id
    if user_id not in user_states:
        await message.reply_text("Veuillez commencer l'évaluation avec /start")
        return

    state = user_states[user_id]
    
    # Store the response
    facette_responses = state["responses"].get(state["current_facette"], [])
    facette_responses.append({
        "question": FACETTE_QUESTIONS[state["current_facette"]][state["current_question"] - 1],
        "response": message.text,
        "type": "text"
    })
    state["responses"][state["current_facette"]] = facette_responses
    
    # Generate analysis if this was the last question for the facette
    if state["current_question"] >= len(FACETTE_QUESTIONS[state["current_facette"]]):
        try:
            analysis = await generate_analysis(
                state["current_facette"],
                str(state["responses"][state["current_facette"]])
            )
            state["facette_analyses"][state["current_facette"]] = analysis
            state["facette_global_points"][state["current_facette"]] = parse_analysis(analysis)

            # Finish this facette message
            finish_facette_text = f"""
✅ Vous avez terminé la facette {state['current_facette']}: {SOBANE_FACETTES[state['current_facette']]}

Options disponibles:
- /facette_report - Télécharger le rapport de cette facette
- /reset - Recommencer l'évaluation
"""
            await message.reply_text(finish_facette_text)
            
            # Move to next facette
            state["current_facette"] += 1
            state["current_question"] = 1
            
            if state["current_facette"] <= 18:
                next_question = FACETTE_QUESTIONS[state["current_facette"]][0]
                await message.reply_text(f"Passons à la facette {state['current_facette']}: {SOBANE_FACETTES[state['current_facette']]}\n\n{next_question}")
            else:
                await message.reply_text("Évaluation terminée! Utilisez /report pour générer le rapport final.")
        except Exception as e:
            await message.reply_text(f"Une erreur est survenue: {str(e)}")
    else:
        # Move to next question in current facette
        state["current_question"] += 1
        next_question = FACETTE_QUESTIONS[state["current_facette"]][state["current_question"] - 1]
        await message.reply_text(next_question)




@app.on_message(filters.command("facette_report"))
async def generate_facette_report_command(client, message):
    user_id = message.from_user.id
    if user_id not in user_states:
        await message.reply_text("Aucune évaluation en cours. Utilisez /start pour commencer.")
        return

    state = user_states[user_id]
    current_facette = state["current_facette"] - 1  # Facette actuelle
    
    if current_facette < 1 or current_facette > 18:
        await message.reply_text("Aucune facette à rapporter pour le moment.")
        return
    
    # Vérifier si l'analyse de la facette existe
    if current_facette not in state["facette_analyses"]:
        await message.reply_text("L'analyse de cette facette n'est pas encore terminée.")
        return
    
    report, _ = generate_facette_report(state, current_facette)
    
    # Sauvegarder le rapport
    report_file = f"SOBANE_rapport_facette_{current_facette}.md"
    with open(report_file, "w", encoding='utf-8') as f:
        f.write(report)
    
    await message.reply_document(report_file)
    os.remove(report_file)

async def generate_analysis(facette_number, user_response):
    prompt = FACETTE_ANALYSIS_TEMPLATE.format(
        facette_name=SOBANE_FACETTES[facette_number],
        facette_number=facette_number,
        user_responses=user_response
    )
    
    response = model.generate_content(prompt)
    return response.text

def parse_analysis(analysis):
    # Helper function to extract sections from the analysis
    def extract_section(section_name, text):
        pattern = rf"## {section_name}\n(.*?)(?=##|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # Split by newlines, remove empty lines, and strip whitespace
            return [line.strip() for line in match.group(1).split('\n') if line.strip()]
        return []

    # Extract main analysis sections
    points_positifs = extract_section("Évaluation des points positifs", analysis)
    points_negatifs = extract_section("Évaluation des points negatifs", analysis)
    points_a_approfondir = extract_section("Points à approfondir", analysis)
    recommandations = extract_section("Recommandations", analysis)

    # Extract severity emoji
    severity_pattern = r"## Évaluation de la gravité globale et emoji recommandé\n(.*?)(?=\Z)"
    severity_match = re.search(severity_pattern, analysis, re.DOTALL)
    severity_emoji = "😐"  # Default if not found

    if severity_match:
        severity_text = severity_match.group(1).strip()
        if "😀" in severity_text:
            severity_emoji = "😀"
        elif "😡" in severity_text:
            severity_emoji = "😡"

    return {
        "points_positifs": points_positifs,
        "points_negatifs": points_negatifs,
        "points_a_approfondir": points_a_approfondir,
        "recommandations": recommandations,
        "severity_emoji": severity_emoji
    }



def create_markdown_report(user_states, user_id):
    state = user_states[user_id]
    markdown_content = "# Rapport d'évaluation SOBANE\n\n"

    # First part: Details of each facette
    markdown_content += "## Détails des facettes\n\n"
    markdown_content += "| Facette | Question | Réponse | Points Positifs | Points Négatifs |\n"
    markdown_content += "|---------|----------|---------|----------------|------------------|\n"

    for facette_num, responses in state["responses"].items():
        analysis = state["facette_analyses"][facette_num]
        parsed_analysis = parse_analysis(analysis)

        for response in responses:
            markdown_content += f"| {facette_num}. {SOBANE_FACETTES[facette_num]} | {response['question']} | {response['response']} | "
            
            # Add points positifs
            if parsed_analysis['points_positifs']:
                markdown_content += "<br>".join(f"- {point}" for point in parsed_analysis['points_positifs'])
            markdown_content += " | "
            
            # Add points negatifs
            if parsed_analysis['points_negatifs']:
                markdown_content += "<br>".join(f"- {point}" for point in parsed_analysis['points_negatifs'])
            markdown_content += " |\n"

    # Second part: Aspects to study further and emoji
    markdown_content += "\n## Synthèse globale\n\n"
    markdown_content += "| Aspects à approfondir | Évaluation globale |\n"
    markdown_content += "|------------------------|--------------------|\n"

    all_points_a_approfondir = []
    global_severity_emoji = "😐"  # Default
    severity_emojis = []

    for facette_num in state["facette_analyses"]:
        analysis = state["facette_analyses"][facette_num]
        parsed_analysis = parse_analysis(analysis)
        all_points_a_approfondir.extend(parsed_analysis['points_a_approfondir'])
        severity_emojis.append(parsed_analysis['severity_emoji'])

    # Determine global severity emoji
    emoji_mapping = {"😀": 1, "😐": 2, "😡": 3}
    global_severity_emoji = max(severity_emojis, key=lambda x: emoji_mapping.get(x, 2))

    # Insérer un saut de ligne entre chaque point
    markdown_content += f"| {'<br>'.join(all_points_a_approfondir)} | {global_severity_emoji} |\n"

    # Third part: Recommendations
    markdown_content += "\n## Recommandations\n\n"
    all_recommandations = []
    for facette_num in state["facette_analyses"]:
        analysis = state["facette_analyses"][facette_num]
        parsed_analysis = parse_analysis(analysis)
        all_recommandations.extend(parsed_analysis['recommandations'])
    
    for rec in all_recommandations:
        markdown_content += f"{rec}\n"

        return markdown_content
@app.on_message(filters.command("report"))
async def generate_report(client, message):
    
    user_id = message.from_user.id
    if user_id not in user_states:
        await message.reply_text("Aucune évaluation en cours. Utilisez /start pour commencer.")
        return

    state = user_states[user_id]
    
    if len(state["facette_analyses"]) < 17:
        await message.reply_text("L'évaluation n'est pas encore terminée. Continuez à répondre aux questions.")
        return
    
    # Créer le rapport final
    report = create_final_summary_report(user_states, user_id)
    
    # Sauvegarder le rapport
    report_file = f"SOBANE_rapport_final_{user_id}.md"
    with open(report_file, "w", encoding='utf-8') as f:
        f.write(report)
    
    await message.reply_document(report_file)
    os.remove(report_file)

"""@app.on_message(filters.command("next"))
async def next_facette_command(client, message):
    user_id = message.from_user.id
    if user_id not in user_states:
        await message.reply_text("Aucune évaluation en cours. Utilisez /start pour commencer.")
        return

    state = user_states[user_id]
    
    # Vérifier si toutes les questions de la facette actuelle ont été répondues
    if state["current_question"] < len(FACETTE_QUESTIONS[state["current_facette"]]):
        await message.reply_text("Vous devez répondre à toutes les questions de la facette actuelle avant de passer à la suivante.")
        return

    # Générer l'analyse si ce n'est pas déjà fait
    if state["current_facette"] not in state["facette_analyses"]:
        try:
            analysis = await generate_analysis(
                state["current_facette"],
                str(state["responses"][state["current_facette"]])
            )
            state["facette_analyses"][state["current_facette"]] = analysis
            state["facette_global_points"][state["current_facette"]] = parse_analysis(analysis)
        except Exception as e:
            await message.reply_text(f"Une erreur est survenue lors de l'analyse: {str(e)}")
            return

    # Passer à la facette suivante
    state["current_facette"] += 1
    state["current_question"] = 1
    
    if state["current_facette"] <= 18:
        next_question = FACETTE_QUESTIONS[state["current_facette"]][0]
        await message.reply_text(f"Passons à la facette {state['current_facette']}: {SOBANE_FACETTES[state['current_facette']]}\n\n{next_question}")
    else:
        await message.reply_text("Évaluation terminée! Utilisez /report pour générer le rapport final.")
"""
@app.on_message(filters.command("reset"))
async def reset_command(client, message):
    user_id = message.from_user.id
    if user_id in user_states:
        del user_states[user_id]
    await message.reply_text("Évaluation réinitialisée. Utilisez /start pour commencer une nouvelle évaluation.")


@app.on_message(filters.command("status"))
async def status_command(client, message):
    user_id = message.from_user.id
    if user_id not in user_states:
        await message.reply_text("Aucune évaluation en cours. Utilisez /start pour commencer.")
        return

    state = user_states[user_id]
    current_facette = state["current_facette"]
    current_question = state["current_question"]

    # Construction du message de statut
    status_message = f"### Statut de votre évaluation SOBANE\n\n"
    status_message += f"**Facette actuelle :** {SOBANE_FACETTES[current_facette]}\n"
    status_message += f"**Question actuelle :** {FACETTE_QUESTIONS[current_facette][current_question - 1]}\n"

    await message.reply_text(status_message)

    
print("Starting bot...")
app.run()