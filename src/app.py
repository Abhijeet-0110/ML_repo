"""
app.py
------
Streamlit Web App: "🌱 Will My Plant Survive Me?"
A playful interactive app that predicts plant survival (Thrives / Survives / Doomed)
based on your plant-care habits and environmental conditions.
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="🌱 Will My Plant Survive Me?",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "plant_model.joblib"
COLUMNS_FILE = MODELS_DIR / "model_columns.joblib"
METADATA_FILE = MODELS_DIR / "metadata.joblib"


@st.cache_resource
def load_model_artifacts():
    """
    Loads and caches the trained model, feature column order, and metadata.
    Avoids reloading on every Streamlit interaction.
    """
    if not MODEL_FILE.exists() or not COLUMNS_FILE.exists():
        return None, None, None

    model = joblib.load(MODEL_FILE)
    columns = joblib.load(COLUMNS_FILE)
    metadata = joblib.load(METADATA_FILE) if METADATA_FILE.exists() else {}
    return model, columns, metadata


def generate_care_tip(
    plant_type: str,
    watering_per_week: int,
    sunlight_hours: float,
    humidity: str,
    temperature_c: float,
    forgetfulness_score: int,
    has_drainage: str,
) -> str:
    """
    Diagnoses the habit that most threatens the plant's health
    and returns a tailored, actionable care tip.
    """
    issues = []

    # Drainage inspection
    if has_drainage == "no":
        if plant_type in ["cactus", "succulent"]:
            issues.append(
                (95, "🪴 **Critical Drainage**: Desert plants in pots without drainage holes will suffer lethal root rot. Repot immediately into a porous pot with drainage holes!")
            )
        else:
            issues.append(
                (75, "🪴 **Drainage Risk**: Standing water suffocates roots. Always use pots with drainage holes.")
            )

    # Plant-specific care checks
    if plant_type == "cactus":
        if watering_per_week > 1:
            issues.append(
                (90, f"🚰 **Overwatering Alert**: You are watering {watering_per_week}x/week. Cacti thrive on neglect and need watering only once every 1–2 weeks!")
            )
        if sunlight_hours < 6:
            issues.append(
                (70, "☀️ **Low Light**: Cacti crave intense, direct sun. Move it to your sunniest south-facing windowsill.")
            )
        if humidity == "high":
            issues.append(
                (60, "💨 **Excess Humidity**: Humid rooms (like bathrooms) can cause fungal rot on desert cacti.")
            )

    elif plant_type == "succulent":
        if watering_per_week > 2:
            issues.append(
                (85, f"🚰 **Watering Too Frequently**: Succulents store water in their leaves. Cut back to once every 1–2 weeks, only when soil is bone dry.")
            )
        if sunlight_hours < 5:
            issues.append(
                (65, "☀️ **Etiolation Risk**: Your succulent needs at least 5–6 hours of bright light to prevent stretching and pale growth.")
            )
        if humidity == "high":
            issues.append(
                (55, "💨 **High Humidity**: Keep succulents in well-ventilated, dry spots.")
            )

    elif plant_type == "fern":
        if watering_per_week < 3:
            issues.append(
                (85, "💧 **Drying Out**: Ferns dislike dry soil! Water 3–5 times per week to keep soil consistently moist.")
            )
        if sunlight_hours > 6:
            issues.append(
                (80, "🔥 **Sunburn Hazard**: Direct harsh sunlight will scorch delicate fern fronds. Move to gentle, indirect light.")
            )
        if humidity == "low":
            issues.append(
                (75, "💨 **Crispy Fronds**: Ferns need high humidity. Mist daily or place near a humidifier or pebble tray.")
            )
        if forgetfulness_score > 5:
            issues.append(
                (70, "⏰ **Needs Attention**: Ferns are not forgiving of neglect. Put a recurring reminder on your phone.")
            )

    elif plant_type == "tropical":
        if temperature_c < 18:
            issues.append(
                (80, f"🥶 **Cold Draft**: Tropical plants shudder below 18°C ({temperature_c:.1f}°C detected). Keep away from chilly windows or AC vents.")
            )
        if humidity == "low":
            issues.append(
                (70, "💨 **Dry Air**: Tropical foliage suffers from brown crispy tips in dry air. Mist leaves or increase room humidity.")
            )
        if watering_per_week < 2:
            issues.append(
                (65, "💧 **Underwatering**: Tropicals need moderate soil moisture. Water 2–3 times a week.")
            )
        if sunlight_hours < 3:
            issues.append(
                (60, "☀️ **Too Dark**: Boost growth by placing in bright, indirect light.")
            )

    elif plant_type == "flowering":
        if sunlight_hours < 5:
            issues.append(
                (85, "🌸 **No Blooms Warning**: Flowering plants require at least 5–6 hours of good sunlight to produce flower buds.")
            )
        if watering_per_week < 2:
            issues.append(
                (70, "💧 **Thirsty Blooms**: Blossoming consumes water. Keep soil evenly moist while buds are active.")
            )
        if watering_per_week > 5:
            issues.append(
                (65, "🚰 **Soggy Roots**: Too much water invites root rot and flower drop. Allow topsoil to dry slightly.")
            )

    # General checks
    if forgetfulness_score >= 8 and plant_type not in ["cactus", "succulent"]:
        issues.append(
            (60, f"🧠 **Forgetfulness Level {forgetfulness_score}/10**: Consider adopting a cactus or snake plant instead, or set recurring plant alarms!")
        )

    if issues:
        # Sort by urgency weight descending and return top tip
        issues.sort(key=lambda x: x[0], reverse=True)
        return issues[0][1]
    else:
        return "✨ **Botanist Stamp of Approval**: Your care habits are remarkably well-matched to this plant's biological needs. Keep up the great work!"


def main():
    # Header & Disclaimer
    st.title("🌱 Will My Plant Survive Me?")
    st.caption("⚠️ *Fun toy model — not real gardening advice. Please don't sue us if your fern turns to dust.*")

    st.markdown(
        "Confess your real plant-care habits below. Our machine learning model "
        "will calculate whether your green friend will **Thrive**, **Survive**, or is **Doomed**."
    )

    # Load artifacts
    model, model_columns, metadata = load_model_artifacts()

    if model is None or model_columns is None:
        st.error(
            "⚠️ **Model artifacts not found!**\n\n"
            "Please run the data generation and training scripts first:\n"
            "```bash\n"
            "python src/generate_data.py\n"
            "python src/train.py\n"
            "```"
        )
        return

    # User Input Form
    with st.container():
        st.subheader("🪴 1. Choose Your Plant")
        plant_type = st.selectbox(
            "Plant Type",
            options=["succulent", "fern", "cactus", "tropical", "flowering"],
            index=0,
            help="Different plants have vastly different biological requirements.",
        )

        st.subheader("💧 2. Your Care Routine")
        col1, col2 = st.columns(2)
        with col1:
            watering_per_week = st.slider(
                "Watering frequency (times per week)",
                min_value=0,
                max_value=7,
                value=2,
                help="How many times a week do you water this plant?",
            )
            sunlight_hours = st.slider(
                "Daily sunlight (hours)",
                min_value=0.0,
                max_value=12.0,
                value=5.0,
                step=0.5,
                help="Total hours of daylight the plant receives each day.",
            )
        with col2:
            forgetfulness_score = st.slider(
                "Your forgetfulness score (1 = Botanical Monk, 10 = What plant?)",
                min_value=1,
                max_value=10,
                value=4,
                help="Be honest with yourself!",
            )
            has_drainage = st.selectbox(
                "Does the pot have drainage holes?",
                options=["yes", "no"],
                index=0,
                help="Holes at the bottom of the pot to let excess water escape.",
            )

        st.subheader("🌡️ 3. Environment")
        col3, col4 = st.columns(2)
        with col3:
            humidity = st.selectbox(
                "Room humidity level",
                options=["low", "medium", "high"],
                index=1,
                help="Low = dry heated/AC room, High = bathroom/greenhouse.",
            )
        with col4:
            temperature_c = st.number_input(
                "Room temperature (°C)",
                min_value=10.0,
                max_value=35.0,
                value=22.0,
                step=0.5,
                help="Average ambient indoor temperature.",
            )

    st.markdown("---")

    # Predict Button
    if st.button("🔮 Predict My Plant's Fate", type="primary", use_container_width=True):
        # 1. Format user input as single-row DataFrame
        raw_input = pd.DataFrame(
            [
                {
                    "plant_type": plant_type,
                    "watering_per_week": int(watering_per_week),
                    "sunlight_hours": float(sunlight_hours),
                    "humidity": humidity,
                    "temperature_c": float(temperature_c),
                    "forgetfulness_score": int(forgetfulness_score),
                    "has_drainage": has_drainage,
                }
            ]
        )

        # 2. One-hot encode categoricals identically to training
        categorical_cols = ["plant_type", "humidity", "has_drainage"]
        encoded_input = pd.get_dummies(raw_input, columns=categorical_cols, drop_first=False)

        # 3. Robust column alignment to defeat the inference column-mismatch trap!
        # Reindexes columns according to exact training order; missing dummies are filled with 0.
        aligned_input = encoded_input.reindex(columns=model_columns, fill_value=0)

        # 4. Predict outcome & class probabilities
        prediction = model.predict(aligned_input)[0]
        probabilities = model.predict_proba(aligned_input)[0]
        class_prob_map = dict(zip(model.classes_, probabilities))

        # Outcome presentation configuration
        status_configs = {
            "Thrives": {
                "emoji": "🌿",
                "label": "Thrives!",
                "color": "#2e7d32",
                "desc": "Green thumb alert! Your plant is living its best life under your care.",
            },
            "Survives": {
                "emoji": "🍂",
                "label": "Survives...",
                "color": "#ed6c02",
                "desc": "It's hanging on, but not thrilled. A few adjustments could make it truly happy.",
            },
            "Doomed": {
                "emoji": "💀",
                "label": "Doomed!",
                "color": "#d32f2f",
                "desc": "Say your goodbyes or make urgent changes right now.",
            },
        }

        config = status_configs.get(
            prediction,
            {"emoji": "🌱", "label": prediction, "color": "#1976d2", "desc": ""},
        )

        st.subheader("🎯 Prediction Result")
        st.markdown(
            f"""
            <div style="background-color: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; border-left: 6px solid {config['color']}; margin-bottom: 20px;">
                <h2 style="margin: 0; color: {config['color']};">{config['emoji']} {config['label']}</h2>
                <p style="margin: 8px 0 0 0; font-size: 1.05rem;">{config['desc']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability distribution breakdown
        st.markdown("#### 📊 Survival Probability Breakdown")
        col_t, col_s, col_d = st.columns(3)

        p_thrives = class_prob_map.get("Thrives", 0.0)
        p_survives = class_prob_map.get("Survives", 0.0)
        p_doomed = class_prob_map.get("Doomed", 0.0)

        with col_t:
            st.metric("🌿 Thrives", f"{p_thrives * 100:.1f}%")
            st.progress(float(p_thrives))
        with col_s:
            st.metric("🍂 Survives", f"{p_survives * 100:.1f}%")
            st.progress(float(p_survives))
        with col_d:
            st.metric("💀 Doomed", f"{p_doomed * 100:.1f}%")
            st.progress(float(p_doomed))

        # Bonus: Personalized Plant Care Tip
        st.markdown("---")
        st.markdown("#### 💡 Tailored Plant Care Tip")
        care_tip = generate_care_tip(
            plant_type=plant_type,
            watering_per_week=watering_per_week,
            sunlight_hours=sunlight_hours,
            humidity=humidity,
            temperature_c=temperature_c,
            forgetfulness_score=forgetfulness_score,
            has_drainage=has_drainage,
        )
        st.info(care_tip)


if __name__ == "__main__":
    main()
