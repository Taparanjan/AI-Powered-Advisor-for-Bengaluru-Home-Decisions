from pathlib import Path
import pickle
import re
import os

import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, session


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "linear_regression_model.pkl"
DATASET_PATH = BASE_DIR / "dataset" / "Bengaluru_House_Data.csv"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip().strip("'").strip('"')
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.ai/v1/requests")


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Make sure linear_regression_model.pkl exists in the project root."
        )

    load_errors = []

    try:
        from joblib import load as joblib_load

        return joblib_load(model_path)
    except Exception as exc:
        load_errors.append(f"joblib load failed: {exc}")

    try:
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)
    except Exception as exc:
        load_errors.append(f"pickle load failed: {exc}")

    try:
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file, encoding="latin1")
    except Exception as exc:
        load_errors.append(f"pickle latin1 load failed: {exc}")

    raise RuntimeError(
        "Could not load linear_regression_model.pkl.\n" + "\n".join(load_errors)
    )


def parse_sqft(total_sqft: str):
    if not isinstance(total_sqft, str):
        return None

    cleaned_value = total_sqft.strip().replace(",", "")
    if not cleaned_value:
        return None

    if "-" in cleaned_value:
        parts = cleaned_value.split("-")
        try:
            low, high = float(parts[0]), float(parts[1])
            return (low + high) / 2
        except ValueError:
            return None

    try:
        return float(cleaned_value)
    except ValueError:
        match = re.search(r"[0-9]+\.?[0-9]*", cleaned_value)
        if match:
            return float(match.group())
        return None


def load_location_market_data(dataset_path: Path):
    if not dataset_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(dataset_path)
    working_df = df[["location", "total_sqft", "bath", "price", "size"]].copy()
    working_df["location"] = working_df["location"].astype(str).str.strip()
    working_df["total_sqft"] = working_df["total_sqft"].apply(
        lambda value: parse_sqft(str(value)) if pd.notna(value) else None
    )
    working_df["bhk"] = working_df["size"].astype(str).str.extract(r"(\d+)")[0]
    working_df["bhk"] = pd.to_numeric(working_df["bhk"], errors="coerce")
    working_df["bath"] = pd.to_numeric(working_df["bath"], errors="coerce")
    working_df["price"] = pd.to_numeric(working_df["price"], errors="coerce")
    working_df = working_df.dropna(subset=["location", "total_sqft", "bhk", "bath", "price"])
    working_df = working_df[
        (working_df["total_sqft"] > 0)
        & (working_df["bhk"] > 0)
        & (working_df["bath"] > 0)
        & (working_df["price"] > 0)
    ]
    working_df["price_per_sqft"] = working_df["price"] * 100000 / working_df["total_sqft"]

    grouped = (
        working_df.groupby("location")
        .agg(
            avg_price_lakh=("price", "median"),
            avg_sqft=("total_sqft", "median"),
            avg_bhk=("bhk", "median"),
            avg_bath=("bath", "median"),
            avg_price_per_sqft=("price_per_sqft", "median"),
            listing_count=("location", "size"),
        )
        .reset_index()
    )
    return grouped[grouped["listing_count"] >= 5].copy()


model = load_model(MODEL_PATH)
MARKET_DATA = load_location_market_data(DATASET_PATH)
FEATURE_NAMES = getattr(model, "feature_names_in_", None)
if FEATURE_NAMES is None:
    raise RuntimeError(
        "Loaded model does not expose feature_names_in_. The input form cannot be built safely."
    )

BASE_FEATURES = {"total_sqft", "bath", "bhk"}
LOCATION_CHOICES = sorted(
    feature_name for feature_name in FEATURE_NAMES if feature_name not in BASE_FEATURES
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")


def build_input_dataframe(location: str, total_sqft: float, bath: float, bhk: int):
    row = {feature_name: 0 for feature_name in FEATURE_NAMES}
    row["total_sqft"] = total_sqft
    row["bath"] = bath
    row["bhk"] = bhk

    if location in row:
        row[location] = 1

    return pd.DataFrame([row])


def build_property_summary(location: str, total_sqft: float, bath: float, bhk: int, prediction: float):
    price_per_sqft = (prediction * 100000 / total_sqft) if total_sqft else 0
    return (
        f"This property is a {bhk} BHK home in {location} with {bath:.0f} bathrooms and "
        f"{total_sqft:.0f} square feet. The machine learning model estimates the price at "
        f"{prediction:.2f} lakhs, which is about {price_per_sqft:.0f} INR per square foot. "
        "Treat this as a model-based estimate, not a final market valuation. Compare nearby "
        "listings, building condition, amenities, and registration costs before making a decision."
    )


def suggest_nearby_locations(context, limit=3):
    if MARKET_DATA.empty:
        return []

    candidates = MARKET_DATA[MARKET_DATA["location"] != context["location"]].copy()
    if candidates.empty:
        return []

    budget = context["prediction"]
    target_sqft = context["total_sqft"]
    target_bhk = context["bhk"]
    target_price_per_sqft = (budget * 100000 / target_sqft) if target_sqft else 0

    candidates["score"] = (
        (candidates["avg_price_lakh"] - budget).abs() / max(budget, 1)
        + (candidates["avg_bhk"] - target_bhk).abs() * 0.6
        + (candidates["avg_sqft"] - target_sqft).abs() / max(target_sqft, 1)
        + (candidates["avg_price_per_sqft"] - target_price_per_sqft).abs() / max(target_price_per_sqft, 1)
    )

    suggestions = candidates.sort_values("score").head(limit)
    result = []
    for _, row in suggestions.iterrows():
        result.append(
            {
                "location": row["location"],
                "avg_price_lakh": float(row["avg_price_lakh"]),
                "avg_sqft": float(row["avg_sqft"]),
                "listing_count": int(row["listing_count"]),
            }
        )
    return result


def format_location_suggestions(suggestions):
    if not suggestions:
        return "No strong alternative location suggestions are available from the dataset yet."

    formatted = []
    for suggestion in suggestions:
        formatted.append(
            f"{suggestion['location']} around {suggestion['avg_price_lakh']:.1f} lakhs "
            f"for roughly {suggestion['avg_sqft']:.0f} sqft"
        )
    return "; ".join(formatted) + "."


def groq_chat_message(messages):
    if not GROQ_API_KEY:
        return None

    payload = {
        "model": "llama-3.1-8b-instant",
        "input": messages,
        "max_output_tokens": 260,
        "temperature": 0.3,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            json=payload,
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        if "output" in data and isinstance(data["output"], list):
            output_chunks = []
            for item in data["output"]:
                if isinstance(item, dict) and "content" in item:
                    content = item["content"]
                    if isinstance(content, list):
                        for chunk in content:
                            if isinstance(chunk, dict) and chunk.get("type") == "output_text":
                                output_chunks.append(chunk.get("text", ""))
                    elif isinstance(content, str):
                        output_chunks.append(content)
            text = " ".join(part.strip() for part in output_chunks if part.strip())
            if text:
                return text

        if isinstance(data.get("text"), str):
            return data["text"]

        if isinstance(data.get("response"), str):
            return data["response"]
    except Exception:
        return None

    return None


def generate_summary_response(context):
    nearby_text = format_location_suggestions(context.get("nearby_suggestions", []))
    fallback_summary = build_property_summary(
        location=context["location"],
        total_sqft=context["total_sqft"],
        bath=context["bath"],
        bhk=context["bhk"],
        prediction=context["prediction"],
    ) + " Nearby options worth checking: " + nearby_text

    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are Mortgage Powered Advisor, a premium Bengaluru home-buying and mortgage guidance assistant. "
                "Summarize the property request using the user inputs and the ML prediction. Sound polished and professional. "
                "Mention affordability context carefully, note that the value is an estimate, and recommend nearby alternative locations."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Location: {context['location']}\n"
                f"Total sqft: {context['total_sqft']}\n"
                f"BHK: {context['bhk']}\n"
                f"Bathrooms: {context['bath']}\n"
                f"Predicted price: {context['prediction']:.2f} lakhs\n"
                f"Nearby alternative locations: {nearby_text}\n"
                "Give a short premium-sounding summary, mention that the prediction is an estimate, "
                "add one mortgage or budgeting suggestion, and reference the alternative locations."
            ),
        },
    ]

    return groq_chat_message(prompt_messages) or fallback_summary


def generate_follow_up_response(context, user_message):
    nearby_text = format_location_suggestions(context.get("nearby_suggestions", []))
    fallback_reply = (
        f"Based on the current inputs, the model estimate is {context['prediction']:.2f} lakhs for a "
        f"{context['bhk']} BHK property in {context['location']} with {context['bath']:.0f} bathrooms "
        f"and {context['total_sqft']:.0f} square feet. Nearby alternatives you can also review are {nearby_text} "
        f"Your question was: {user_message}. Use this estimate as a starting point and compare it with recent listings in the same area."
    )

    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are Mortgage Powered Advisor, a professional property and mortgage assistant. "
                "Answer the user's follow-up using the saved property inputs, ML prediction, and nearby location suggestions. "
                "Be grounded in the provided numbers and avoid making up facts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Saved property context:\n"
                f"Location: {context['location']}\n"
                f"Total sqft: {context['total_sqft']}\n"
                f"BHK: {context['bhk']}\n"
                f"Bathrooms: {context['bath']}\n"
                f"Predicted price: {context['prediction']:.2f} lakhs\n"
                f"Nearby alternatives: {nearby_text}\n\n"
                f"Follow-up question: {user_message}"
            ),
        },
    ]

    return groq_chat_message(prompt_messages) or fallback_reply


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    chat_history = session.get("chat_history", [])
    form_data = {
        "location": LOCATION_CHOICES[0] if LOCATION_CHOICES else "",
        "total_sqft": "1200",
        "bhk": "2",
        "bath": "2",
    }
    latest_context = session.get("latest_prediction")

    if latest_context:
        form_data = {
            "location": latest_context.get("location", form_data["location"]),
            "total_sqft": str(latest_context.get("total_sqft", form_data["total_sqft"])),
            "bhk": str(latest_context.get("bhk", form_data["bhk"])),
            "bath": str(latest_context.get("bath", form_data["bath"])),
        }
        prediction = latest_context.get("prediction")

    if request.method == "POST":
        action = request.form.get("action", "predict")

        if action == "chat":
            user_message = request.form.get("chat_message", "").strip()
            latest_context = session.get("latest_prediction")

            if not latest_context:
                error = "Predict a property price first so the chatbot has context."
            elif not user_message:
                error = "Enter a message for the chatbot."
            else:
                chat_history.append(
                    {
                        "role": "user",
                        "title": "Your follow-up",
                        "content": user_message,
                    }
                )
                assistant_reply = generate_follow_up_response(latest_context, user_message)
                chat_history.append(
                    {
                        "role": "assistant",
                        "title": "Mortgage Powered Advisor",
                        "content": assistant_reply,
                    }
                )
                session["chat_history"] = chat_history
        else:
            form_data = {
                "location": request.form.get("location", form_data["location"]).strip(),
                "total_sqft": request.form.get("total_sqft", "").strip(),
                "bhk": request.form.get("bhk", "").strip(),
                "bath": request.form.get("bath", "").strip(),
            }

            total_sqft = parse_sqft(form_data["total_sqft"])

            try:
                bhk = int(form_data["bhk"])
                bath = float(form_data["bath"])
            except ValueError:
                bhk = None
                bath = None

            if total_sqft is None:
                error = "Enter a valid total square feet value."
            elif bhk is None or bhk <= 0:
                error = "Enter a valid BHK value."
            elif bath is None or bath <= 0:
                error = "Enter a valid bathroom count."
            else:
                try:
                    input_frame = build_input_dataframe(
                        location=form_data["location"],
                        total_sqft=total_sqft,
                        bath=bath,
                        bhk=bhk,
                    )
                    prediction = float(model.predict(input_frame)[0])
                    nearby_suggestions = suggest_nearby_locations(
                        {
                            "location": form_data["location"],
                            "total_sqft": total_sqft,
                            "bhk": bhk,
                            "prediction": prediction,
                        }
                    )
                    latest_context = {
                        "location": form_data["location"],
                        "total_sqft": total_sqft,
                        "bhk": bhk,
                        "bath": bath,
                        "prediction": prediction,
                        "nearby_suggestions": nearby_suggestions,
                    }
                    session["latest_prediction"] = latest_context

                    user_summary = (
                        f"Location: {form_data['location']}, Total sqft: {total_sqft:.0f}, "
                        f"BHK: {bhk}, Bathrooms: {bath:.0f}"
                    )
                    assistant_summary = generate_summary_response(latest_context)

                    chat_history = [
                        {
                            "role": "user",
                            "title": "Property inputs",
                            "content": user_summary,
                        },
                        {
                            "role": "assistant",
                            "title": "Mortgage Powered Advisor",
                            "content": assistant_summary,
                        },
                    ]
                    session["chat_history"] = chat_history
                except Exception as exc:
                    error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        location_choices=LOCATION_CHOICES,
        form_data=form_data,
        prediction=prediction,
        error=error,
        chat_history=chat_history,
        nearby_suggestions=(latest_context or {}).get("nearby_suggestions", []),
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
