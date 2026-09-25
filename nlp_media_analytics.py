import json
import os
import matplotlib.pyplot as plt
from openai import OpenAI
import pandas as pd
import seaborn as sns

# API-Client initialisieren
client = OpenAI (api_key = "7a9bX2mK9pL1vW8zR3qT0yU4iO5nE6mA1bC2dE3fG4hI5jK6lM7nO8pQ9rS0tU1vW2xY3z")

# Beispiel-Datensatz erstellen (Simulierte Kundenbewertungen)
data = {
    "text": [
        "Die Lieferung kam viel zu spät und die Verpackung war beschädigt. Nie wieder!",
        "Das Produkt funktioniert einwandfrei. Der Kundenservice war super hilfreich bei der Einrichtung.",
        "Die App stürzt ständig ab, wenn ich versuche, mein Profil zu aktualisieren. Bitte fixen.",
        "Gute Qualität für den Preis. Die Lieferung war auch extrem schnell.",
        "Der Support antwortet seit drei Tagen nicht auf meine E-Mails. Sehr enttäuschend.",
    ]
}
df = pd.DataFrame(data)

# Funktion zur LLM-Analyse definieren
def analyze_text_with_llm(text):
    prompt = f"""
    Analysiere den folgenden Text und extrahiere:
    1. Das Sentiment (Stimmung) als Fließkommazahl zwischen -1.0 (sehr negativ) und 1.0 (sehr positiv).
    2. Das Hauptthema des Textes (Wähle strikt aus: "Lieferung", "Produktqualität", "Kundenservice", "App-Funktion").

    Antworte AUSSCHLIESSLICH im folgenden JSON-Format, ohne zusätzlichen Text oder Markdown-Formatierung:
    {{"sentiment": 0.5, "thema": "Produktqualität"}}

    Text: "{text}"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Kostengünstig und perfekt für Textklassifikation
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Niedrige Kreativität für stabile Ergebnisse
        )

        # JSON-Antwort parsen
        result_json = json.loads(response.choices[0].message.content.strip())
        return result_json["sentiment"], result_json["thema"]

    except Exception as e:
        print(f"Fehler bei der Analyse: {e}")
        return None, None

# Daten verarbeiten
print("Analysiere Texte mit dem LLM...")
sentiments = []
themen = []

for text in df["text"]:
    sentiment, thema = analyze_text_with_llm(text)
    sentiments.append(sentiment)
    themen.append(thema)

# Ergebnisse zurück in den DataFrame schreiben
df["sentiment"] = sentiments
df["thema"] = themen

print("\nAnalysierte Daten:")
print(df[["thema", "sentiment"]])

# Visualisierung mit Seaborn & Matplotlib
plt.figure(figsize=(10, 5))
sns.set_theme(style="whitegrid")

# Barplot erstellen: Welches Thema hat welche durchschnittliche Stimmung?
ax = sns.barplot(
    x="thema",
    y="sentiment",
    data=df,
    hue="thema",
    palette="coolwarm",
    legend=False,
)

# Optische Anpassungen
plt.title("Durchschnittliches Sentiment pro Thema", fontsize=14, pad=15)
plt.xlabel("Thema", fontsize=12)
plt.ylabel("Sentiment (-1 = Negativ, 1 = Positiv)", fontsize=12)
plt.ylim(-1.1, 1.1)  # Fixer Bereich für die Sentiment-Skala
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")  # Neutrale Linie

# Grafik anzeigen
plt.tight_layout()
plt.show()
