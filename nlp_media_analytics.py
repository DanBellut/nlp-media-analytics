import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from openai import OpenAI

MODEL = "gpt-4o-mini"  # Kostengünstig und perfekt für Textklassifikation
THEMEN = ["Lieferung", "Produktqualität", "Kundenservice", "App-Funktion"]

SYSTEM_PROMPT = f"""
Du analysierst Kundenfeedback. Extrahiere aus dem Text des Nutzers:
1. Das Sentiment (Stimmung) als Fließkommazahl zwischen -1.0 (sehr negativ) und 1.0 (sehr positiv).
2. Das Hauptthema des Textes (Wähle strikt aus: {", ".join(f'"{t}"' for t in THEMEN)}).

Antworte ausschließlich im folgenden JSON-Format:
{{"sentiment": 0.5, "thema": "Produktqualität"}}
""".strip()

# Beispiel-Datensatz erstellen (Simulierte Kundenbewertungen)
BEISPIEL_TEXTE = [
    "Die Lieferung kam viel zu spät und die Verpackung war beschädigt. Nie wieder!",
    "Das Produkt funktioniert einwandfrei. Der Kundenservice war super hilfreich bei der Einrichtung.",
    "Die App stürzt ständig ab, wenn ich versuche, mein Profil zu aktualisieren. Bitte fixen.",
    "Gute Qualität für den Preis. Die Lieferung war auch extrem schnell.",
    "Der Support antwortet seit drei Tagen nicht auf meine E-Mails. Sehr enttäuschend.",
]


def analyze_text_with_llm(client, text):
    """Gibt (sentiment, thema) zurück oder (None, None), wenn die Analyse fehlschlägt."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            # Der Text steht in einer eigenen Nachricht, damit er die Anweisungen nicht überschreiben kann
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},  # Erzwingt gültiges JSON
            temperature=0.0,  # Niedrige Kreativität für stabile Ergebnisse
        )
        result = json.loads(response.choices[0].message.content)

        # Antwort prüfen, statt ihr blind zu vertrauen
        sentiment = max(-1.0, min(1.0, float(result["sentiment"])))
        thema = result["thema"]
        if thema not in THEMEN:
            raise ValueError(f"Unbekanntes Thema: {thema!r}")
        return sentiment, thema

    except Exception as e:
        print(f"Fehler bei der Analyse von {text[:40]!r}...: {e}")
        return None, None


def analyze_dataframe(client, df, text_column="text"):
    results = [analyze_text_with_llm(client, text) for text in df[text_column]]
    df = df.copy()
    df["sentiment"], df["thema"] = zip(*results)
    return df


def plot_sentiment_per_theme(df, output_path=None):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))

    # Barplot erstellen: Welches Thema hat welche durchschnittliche Stimmung?
    sns.barplot(
        x="thema",
        y="sentiment",
        data=df.dropna(subset=["sentiment", "thema"]),
        hue="thema",
        order=THEMEN,
        palette="coolwarm",
        legend=False,
    )

    # Optische Anpassungen
    plt.title("Durchschnittliches Sentiment pro Thema", fontsize=14, pad=15)
    plt.xlabel("Thema", fontsize=12)
    plt.ylabel("Sentiment (-1 = Negativ, 1 = Positiv)", fontsize=12)
    plt.ylim(-1.1, 1.1)  # Fixer Bereich für die Sentiment-Skala
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")  # Neutrale Linie
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()


def main():
    # Der API-Key wird aus der Umgebungsvariable OPENAI_API_KEY gelesen
    client = OpenAI()

    df = pd.DataFrame({"text": BEISPIEL_TEXTE})

    print("Analysiere Texte mit dem LLM...")
    df = analyze_dataframe(client, df)

    print("\nAnalysierte Daten:")
    print(df[["thema", "sentiment"]])

    if df["sentiment"].isna().all():
        print("\nKeine Texte erfolgreich analysiert, keine Grafik erstellt.")
        return

    plot_sentiment_per_theme(df)


if __name__ == "__main__":
    main()
