import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import base64
from pathlib import Path

# Load environment variables
load_dotenv(override=True)

# Set page configuration
st.set_page_config(
    page_title="Diabetes Nutrition Plan Generator",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Mauritanian food data
@st.cache_data
def load_mauritania_food_data():
    try:
        with open('links.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # Use hardcoded data as fallback
        return {
            "regions": {
                "Nouakchott": {
                    "common_foods": [
                        {"name": "Thieboudienne", "category": "Main dish", "nutrition": {"carbs": "high", "protein": "medium", "fat": "medium"}, "glycemic_index": "medium", "affordability": "medium", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/dishes/Thieboudienne (fish and rice dish).jpeg"},
                        {"name": "Couscous", "category": "Staple", "nutrition": {"carbs": "high", "protein": "low", "fat": "low"}, "glycemic_index": "medium", "affordability": "high", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/staples/couscous.jpeg"},
                        {"name": "Dates", "category": "Fruit", "nutrition": {"carbs": "high", "protein": "low", "fat": "low"}, "glycemic_index": "high", "affordability": "medium", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/fruits/dates.jpg"},
                        {"name": "Millet", "category": "Grain", "nutrition": {"carbs": "high", "protein": "medium", "fat": "low"}, "glycemic_index": "medium", "affordability": "high", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/staples/millet.jpeg"},
                        {"name": "Goat meat", "category": "Protein", "nutrition": {"carbs": "none", "protein": "high", "fat": "medium"}, "glycemic_index": "low", "affordability": "medium", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/proteins/goatmeat.jpeg"},
                        {"name": "Camel milk", "category": "Dairy", "nutrition": {"carbs": "medium", "protein": "medium", "fat": "medium"}, "glycemic_index": "low", "affordability": "medium", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/dairy/camel_milk.jpeg"},
                        {"name": "Mahfe", "category": "Main dish", "nutrition": {"carbs": "medium", "protein": "high", "fat": "high"}, "glycemic_index": "medium", "affordability": "medium", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/dishes/mahfe.jpeg"},
                        {"name": "Rice", "category": "Staple", "nutrition": {"carbs": "high", "protein": "low", "fat": "low"}, "glycemic_index": "high", "affordability": "high", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/staples/rice.jpeg"},
                        {"name": "Mangoes", "category": "Fruit", "nutrition": {"carbs": "medium", "protein": "low", "fat": "low"}, "glycemic_index": "medium", "affordability": "medium", "seasonality": "seasonal", "image_url": "https://ahmed-ai.netlify.app/fruits/mango.jpg"},
                        {"name": "Lentils", "category": "Legume", "nutrition": {"carbs": "medium", "protein": "medium", "fat": "low"}, "glycemic_index": "low", "affordability": "high", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/proteins/lentils.jpg"}
                    ],
                    "markets": ["Marché Capital", "Marché Cinquième"]
                },
                "Nouadhibou": {
                    "common_foods": [
                        {"name": "Fish (various)", "category": "Protein", "nutrition": {"carbs": "none", "protein": "high", "fat": "medium"}, "glycemic_index": "low", "affordability": "high", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/proteins/fish.jpg"},
                        {"name": "Rice", "category": "Staple", "nutrition": {"carbs": "high", "protein": "low", "fat": "low"}, "glycemic_index": "high", "affordability": "high", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/staples/rice.jpeg"},
                        {"name": "Mbakhar", "category": "Main dish", "nutrition": {"carbs": "medium", "protein": "high", "fat": "medium"}, "glycemic_index": "medium", "affordability": "medium", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/dishes/mbakhar(fish stew).jpeg"},
                        {"name": "Lentils", "category": "Legume", "nutrition": {"carbs": "medium", "protein": "medium", "fat": "low"}, "glycemic_index": "low", "affordability": "high", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/proteins/lentils.jpg"},
                        {"name": "Flatbread", "category": "Staple", "nutrition": {"carbs": "high", "protein": "low", "fat": "low"}, "glycemic_index": "high", "affordability": "high", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/staples/flatbread.jpeg"},
                        {"name": "Oranges", "category": "Fruit", "nutrition": {"carbs": "medium", "protein": "low", "fat": "low"}, "glycemic_index": "medium", "affordability": "medium", "seasonality": "seasonal", "image_url": "https://ahmed-ai.netlify.app/fruits/orange.jpeg"}
                    ],
                    "markets": ["Marché Poisson", "Central Market"]
                },
                "Rural Areas": {
                    "common_foods": [
                        {"name": "Millet", "category": "Grain", "nutrition": {"carbs": "high", "protein": "medium", "fat": "low"}, "glycemic_index": "medium", "affordability": "high", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/staples/millet.jpeg"},
                        {"name": "Sorghum", "category": "Grain", "nutrition": {"carbs": "high", "protein": "medium", "fat": "low"}, "glycemic_index": "medium", "affordability": "high", "seasonality": "harvest season", "image_url": "https://ahmed-ai.netlify.app/staples/sorghum.jpeg"},
                        {"name": "Wild greens", "category": "Vegetable", "nutrition": {"carbs": "low", "protein": "low", "fat": "low"}, "glycemic_index": "low", "affordability": "high", "seasonality": "seasonal", "image_url": "https://ahmed-ai.netlify.app/vegetables/wild_greens.jpeg"},
                        {"name": "Camel meat", "category": "Protein", "nutrition": {"carbs": "none", "protein": "high", "fat": "low"}, "glycemic_index": "low", "affordability": "low", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/proteins/camel_meat.jpeg"},
                        {"name": "Goat milk", "category": "Dairy", "nutrition": {"carbs": "medium", "protein": "medium", "fat": "medium"}, "glycemic_index": "low", "affordability": "medium", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/dairy/goat_milk.jpeg"},
                        {"name": "Cherchem", "category": "Main dish", "nutrition": {"carbs": "high", "protein": "medium", "fat": "low"}, "glycemic_index": "medium", "affordability": "high", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/dishes/chechem(mp).jpeg"},
                        {"name": "Lakh", "category": "Dessert", "nutrition": {"carbs": "high", "protein": "medium", "fat": "medium"}, "glycemic_index": "high", "affordability": "medium", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/dishes/lakh.jpeg"},
                        {"name": "Dates", "category": "Fruit", "nutrition": {"carbs": "high", "protein": "low", "fat": "low"}, "glycemic_index": "high", "affordability": "medium", "seasonality": "year-round", "image_url": "https://ahmed-ai.netlify.app/fruits/dates.jpg"}
                    ],
                    "markets": ["Weekly village markets"]
                }
            },
            "diabetes_friendly_options": {
                "high_recommendation": [
                    "Fish", "Goat meat", "Wild greens", "Lentils", "Camel milk", "Goat milk", "Mbakhar", "Beans", "Chickpeas", "Eggs"
                ],
                "moderate_consumption": [
                    "Millet", "Sorghum", "Couscous", "Vegetables", "Chicken", "Mangoes", "Oranges", "Guava", "Watermelon"
                ],
                "limited_consumption": [
                    "Rice", "Dates", "Sweet tea", "Lakh", "Wheat bread", "Flatbread", "Banana"
                ]
            },
            "food_availability_by_income": {
                "low": ["Millet", "Sorghum", "Wild greens", "Beans", "Lentils", "Eggs"],
                "medium": ["Rice", "Couscous", "Fish", "Goat meat", "Lentils", "Chicken", "Vegetables"],
                "high": ["Imported vegetables", "Imported fruits", "Meat varieties", "Dairy products", "Beef", "Lamb"]
            },
            "seasonal_availability": {
                "year_round": ["Millet", "Rice", "Fish", "Dates", "Lentils", "Beans", "Chickpeas"],
                "seasonal": ["Wild greens", "Local vegetables", "Fruits", "Mangoes", "Watermelon", "Guava", "Oranges"]
            }
        }

# Initialize the food data
mauritania_food_data = load_mauritania_food_data()

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0d6efd;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #495057;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    .info-text {
        color: #6c757d;
        font-size: 0.9rem;
    }
    .highlight {
        background-color: #e6f2ff;
        padding: 0.5rem;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)
def get_translation(key, language=None):
    """
    Get translated text based on the language setting
    """
    if language is None:
        language = st.session_state.get('language', 'English')
    
    translations = {
        "English": {
            "generate_button": "Generate Personalized Nutrition Plan",
            "download_button": "Download Nutrition Plan",
            "api_error": "Please enter your OpenAI API key in the sidebar to generate a plan.",
            "generating": "Generating your personalized nutrition plan...",
            "visual_generating": "Creating visual guide for improved comprehension...",
            "plan_header": "Your Personalized Nutrition Plan"
        },
        "Français": {
            "generate_button": "Générer un Plan Nutritionnel Personnalisé",
            "download_button": "Télécharger le Plan Nutritionnel",
            "api_error": "Veuillez entrer votre clé API OpenAI dans la barre latérale pour générer un plan.",
            "generating": "Génération de votre plan nutritionnel personnalisé...",
            "visual_generating": "Création d'un guide visuel pour une meilleure compréhension...",
            "plan_header": "Votre Plan Nutritionnel Personnalisé"
        },
        "العربية": {
            "generate_button": "توليد خطة تغذية مخصصة",
            "download_button": "تنزيل خطة التغذية",
            "api_error": "يرجى إدخال مفتاح API الخاص بك من OpenAI في الشريط الجانبي لتوليد خطة.",
            "generating": "جاري إنشاء خطة التغذية المخصصة الخاصة بك...",
            "visual_generating": "إنشاء دليل مرئي لفهم أفضل...",
            "plan_header": "خطة التغذية المخصصة الخاصة بك"
        }
    }
    
    if language in translations and key in translations[language]:
        return translations[language][key]
    return key  # Return the key itself if translation not found
    
# Define the OpenAI API function
def generate_nutrition_plan(user_data, food_data):
    """
    Generate personalized nutrition plan using OpenAI API with Mauritanian food data
    """
    try:
        # Get API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key is missing. Please set it in your environment variables.")
            return None
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Get region-specific foods
        selected_region = user_data.get('mauritania_region', 'Nouakchott')
        region_foods = food_data["regions"][selected_region]["common_foods"]
        
        # Get food recommendations based on diabetes management
        highly_recommended = food_data["diabetes_friendly_options"]["high_recommendation"]
        moderate_consumption = food_data["diabetes_friendly_options"]["moderate_consumption"]
        limited_consumption = food_data["diabetes_friendly_options"]["limited_consumption"]
        
        # Get food availability based on income
        income_level = user_data['income_level'].lower()
        if income_level not in food_data["food_availability_by_income"]:
            income_level = "medium"  # Default to medium if not found
        available_foods = food_data["food_availability_by_income"][income_level]
        
        # Create food lists by category for the nutrition plan
        region_food_names = [food["name"] for food in region_foods]
        
        # Create food information by category
        food_by_category = {}
        for food in region_foods:
            category = food["category"]
            if category not in food_by_category:
                food_by_category[category] = []
            food_by_category[category].append({
                "name": food["name"],
                "glycemic_index": food["glycemic_index"],
                "affordability": food["affordability"],
                "nutrition": food["nutrition"]
            })
        
        # Prepare the prompt for the nutrition plan
        prompt = f"""
        Create a personalized nutrition plan for a person with diabetes with the following characteristics:

        Language Preference: {user_data.get('language', 'English')}
        
        Health Information:
        - Age: {user_data['age']}
        - Gender: {user_data['gender']}
        - Weight: {user_data['weight']} kg
        - Height: {user_data['height']} cm
        - BMI: {user_data['bmi']:.1f}
        - Blood Sugar Level: {user_data['blood_sugar']} mg/dL
        - HbA1c Level: {user_data['hba1c']}%
        - Type of Diabetes: {user_data['diabetes_type']}
        - Daily Activity Level: {user_data['activity_level']}
        - Food Allergies or Restrictions: {user_data['food_allergies']}
        
        Socioeconomic Factors:
        - Income Level: {user_data['income_level']}
        - Location/Region: {selected_region}, Mauritania
        - Education Level: {user_data['education']}
        - Literacy Level: {user_data['literacy']}
        - Access to Cooking Facilities: {user_data['cooking_access']}
        - Access to Refrigeration: {user_data['refrigeration']}
        
        IMPORTANT: Use ONLY the following foods in the meal plan, as these are what's available in {selected_region}, Mauritania:
        {", ".join(region_food_names)}
        
        Diabetes-friendly food categories:
        - Highly recommended (low glycemic index): {", ".join(highly_recommended)}
        - Moderate consumption (medium glycemic index): {", ".join(moderate_consumption)}
        - Limited consumption (high glycemic index): {", ".join(limited_consumption)}
        
        Foods available based on {user_data['income_level']} income level:
        {", ".join(available_foods)}
        
        Food information by category:
        {json.dumps(food_by_category, indent=2)}
        
        Please provide a comprehensive 7-day meal plan including:
        1. Breakfast, lunch, dinner, and snacks for each day
        2. Appropriate portion sizes described using simple visual references (e.g., palm of hand, fist, etc.)
        3. Simple cooking instructions
        4. Blood sugar management tips specific to this diet
        5. Essential nutrition education in simple language
        
        The plan MUST ONLY include foods from the provided list that are appropriate for diabetes management.
        If the literacy level is basic, use simple language and visual descriptions.
        For low-income individuals, focus on affordable, locally available options.
        Include guidance on portion control using visual references (palm, fist, etc.) rather than weights.
        """

        # Call the OpenAI API
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2500
            }
        )
        
        response_data = response.json()
        if 'choices' in response_data and len(response_data['choices']) > 0:
            return response_data['choices'][0]['message']['content'].strip()
        else:
            st.error(f"API Error: {response_data}")
            return None
            
    except Exception as e:
        st.error(f"Error generating nutrition plan: {str(e)}")
        return None

def generate_visual_guide(user_data, nutrition_plan, food_data):
    """
    Generate a visual nutrition guide using the provided Mauritanian food data for users with limited literacy
    """
    try:
        # Get API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key is missing. Please set it in your environment variables.")
            return None, None
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Get region-specific foods with images
        selected_region = user_data.get('mauritania_region', 'Nouakchott')
        region_foods = food_data["regions"][selected_region]["common_foods"]
        
        # Get food recommendations based on diabetes management
        highly_recommended = food_data["diabetes_friendly_options"]["high_recommendation"]
        moderate_consumption = food_data["diabetes_friendly_options"]["moderate_consumption"] 
        limited_consumption = food_data["diabetes_friendly_options"]["limited_consumption"]
        
        # Create lists of foods by recommendation level with their images
        green_foods = []
        yellow_foods = []
        red_foods = []
        
        for food in region_foods:
            food_name = food["name"]
            if any(rec in food_name for rec in highly_recommended):
                green_foods.append(food)
            elif any(mod in food_name for mod in moderate_consumption):
                yellow_foods.append(food)
            elif any(lim in food_name for lim in limited_consumption):
                red_foods.append(food)
        
        # Prepare the prompt for visual guide generation
        prompt = f"""
        Create a text description of a visual guide for a diabetes nutrition plan specifically for Mauritania. 
        This will be used to explain healthy eating to someone with limited literacy.
        Language Preference: {user_data.get('language', 'English')}
        
        The person has:
        - Type of Diabetes: {user_data['diabetes_type']}
        - Location/Region: {selected_region}, Mauritania
        - Literacy Level: {user_data['literacy']}
        
        Green Foods (Eat Freely):
        {", ".join([food["name"] for food in green_foods])}
        
        Yellow Foods (Eat Moderately):
        {", ".join([food["name"] for food in yellow_foods])}
        
        Red Foods (Limit Consumption):
        {", ".join([food["name"] for food in red_foods])}
        
        Please create a detailed text description of:
        1. A simple plate diagram showing ideal meal proportions (1/2 vegetables, 1/4 protein, 1/4 whole grains)
        2. Visual representation of portion sizes using common household items (palm of hand for protein, fist for carbs, etc.)
        3. A stoplight system for the foods listed above (green for eat freely, yellow for eat moderately, red for avoid)
        4. Simple visual reminders for meal timing and blood sugar monitoring
        
        The description should be culturally appropriate for Mauritania.
        """

        # Call the OpenAI API
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000
            }
        )
        
        response_data = response.json()
        if 'choices' in response_data and len(response_data['choices']) > 0:
            visual_description = response_data['choices'][0]['message']['content'].strip()
            
            # Instead of generating a new image, we'll create a visual guide using the existing food images
            # Return the description and the lists of foods with their image URLs
            visual_guide_data = {
                "description": visual_description,
                "green_foods": green_foods,
                "yellow_foods": yellow_foods,
                "red_foods": red_foods
            }
            
            return visual_description, visual_guide_data
        else:
            st.error(f"API Error: {response_data}")
            return None, None
            
    except Exception as e:
        st.error(f"Error generating visual guide: {str(e)}")
        return None, None
        
def create_visual_guide_html(visual_guide_data):
    """
    Create an HTML representation of the visual guide using the provided food data
    """
    if not visual_guide_data:
        return "<p>No visual guide data available</p>"
    
    html = """
    <style>
        .visual-guide {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
        }
        .section {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 10px;
        }
        .green-section {
            background-color: #e8f5e9;
            border: 2px solid #4caf50;
        }
        .yellow-section {
            background-color: #fff8e1;
            border: 2px solid #ffc107;
        }
        .red-section {
            background-color: #ffebee;
            border: 2px solid #f44336;
        }
        .section-title {
            font-size: 1.5rem;
            margin-bottom: 10px;
            text-align: center;
        }
        .food-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
        }
        .food-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: white;
            text-align: center;
        }
        .food-card img {
            width: 100px;
            height: 100px;
            object-fit: cover;
            border-radius: 5px;
            margin-bottom: 5px;
        }
        .food-name {
            font-weight: bold;
        }
        .plate-model {
            width: 300px;
            height: 300px;
            border-radius: 50%;
            margin: 20px auto;
            position: relative;
            border: 3px solid #333;
            overflow: hidden;
        }
        .plate-half {
            position: absolute;
            width: 100%;
            height: 50%;
            background-color: #66bb6a;
            top: 0;
        }
        .plate-quarter1 {
            position: absolute;
            width: 50%;
            height: 50%;
            background-color: #fff176;
            bottom: 0;
            left: 0;
        }
        .plate-quarter2 {
            position: absolute;
            width: 50%;
            height: 50%;
            background-color: #ef5350;
            bottom: 0;"""

def calculate_bmi(weight, height):
    """Calculate BMI from weight (kg) and height (cm)"""
    height_m = height / 100  # Convert cm to m
    bmi = weight / (height_m * height_m)
    return bmi

def interpret_blood_sugar(value, type):
    """Interpret blood sugar values"""
    if type == "Fasting":
        if value < 70:
            return "Low"
        elif value <= 99:
            return "Normal"
        elif value <= 125:
            return "Prediabetes"
        else:
            return "Diabetes"
    elif type == "Post-meal":
        if value < 70:
            return "Low"
        elif value <= 140:
            return "Normal"
        elif value <= 199:
            return "Prediabetes"
        else:
            return "Diabetes"
    return "Unknown"

def main():
    # Sidebar for navigation
    st.sidebar.image("https://t4.ftcdn.net/jpg/01/58/87/77/360_F_158877761_vSJ4nKTUvQG8OYgsIJUTpeBwiqm6cynN.jpg", width=150)
    st.sidebar.title("Navigation")

    # Add language selector
    language_options = ["English", "Français", "العربية"]
    selected_language = st.sidebar.selectbox("Language / Langue / اللغة", language_options, index=0)
    
    # Store language in session state
    if 'language' not in st.session_state:
        st.session_state.language = "English"
        
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()
    if 'user_data' in st.session_state:
        st.session_state.user_data['language'] = st.session_state.language
    
    
    pages = ["Home", "Create Nutrition Plan", "About", "Help"]
    choice = st.sidebar.radio("Go to", pages)
    
    # Get OpenAI API key from user or environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.sidebar.success("API Key set successfully!")
        else:
            st.sidebar.warning("Please enter an API key to use the application.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Important Note")
    st.sidebar.info("This application is for educational purposes only. Always consult with a healthcare professional for medical advice.")
    
    if choice == "Home":
        st.markdown("<h1 class='main-header'>AI-Powered Personalized Nutrition Plan for Diabetes</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h2 class='sub-header'>Welcome to the Diabetes Nutrition Plan Generator</h2>", unsafe_allow_html=True)
            st.markdown("""
            This application creates personalized nutrition plans for individuals with diabetes, 
            considering both health metrics and socioeconomic factors. Our goal is to make diabetes 
            management accessible to everyone, regardless of literacy level or economic status.
            
            ### Key Features:
            - Personalized 7-day meal plans based on your health metrics
            - Consideration of income, location, and food availability
            - Adaptation for different literacy and education levels
            - Visual guides for those with limited literacy
            - Culturally appropriate food recommendations
            
            Click on 'Create Nutrition Plan' in the sidebar to get started!
            """)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.image("https://t4.ftcdn.net/jpg/01/58/87/77/360_F_158877761_vSJ4nKTUvQG8OYgsIJUTpeBwiqm6cynN.jpg", width=1000)
        
        # Statistics and facts
        st.markdown("<h2 class='sub-header'>Diabetes: Facts & Figures</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.metric("Global Diabetes Cases", "537 Million", "+51% since 2010")
            st.markdown("<p class='info-text'>Source: International Diabetes Federation, 2021</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.metric("Low/Middle-Income Countries", "3 in 4 adults with diabetes", "live in these regions")
            st.markdown("<p class='info-text'>Source: WHO Global Diabetes Report</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.metric("Diet & Exercise Impact", "Up to 58%", "reduction in diabetes progression")
            st.markdown("<p class='info-text'>Source: Diabetes Prevention Program Research Group</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif choice == "Create Nutrition Plan":
        st.markdown("<h1 class='main-header'>Create Your Personalized Nutrition Plan</h1>", unsafe_allow_html=True)
        
        # Create tabs for different sections of the form
        tab1, tab2, tab3 = st.tabs(["Health Information", "Socioeconomic Factors", "Generate Plan"])
        
        # Create a session state to store form data across tabs
        if 'user_data' not in st.session_state:
            st.session_state.user_data = {
                'age': 45,
                'gender': 'Male',
                'weight': 75,
                'height': 170,
                'bmi': 0,
                'blood_sugar': 130,
                'hba1c': 7.0,
                'diabetes_type': 'Type 2',
                'activity_level': 'Moderate',
                'food_allergies': 'None',
                'income_level': 'Middle',
                'mauritania_region': 'Nouakchott',
                'education': 'Secondary/High School',
                'literacy': 'Basic',
                'cooking_access': 'Moderate (stovetop)',
                'refrigeration': 'Limited'
                'language': 'English'
            }
            # Calculate initial BMI
            st.session_state.user_data['bmi'] = calculate_bmi(
                st.session_state.user_data['weight'], 
                st.session_state.user_data['height']
            )
        
        with tab1:
            st.markdown("<h2 class='sub-header'>Health Information</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.user_data['age'] = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.user_data['age'])
                st.session_state.user_data['gender'] = st.selectbox("Gender", ['Male', 'Female', 'Other'], index=['Male', 'Female', 'Other'].index(st.session_state.user_data['gender']))
                st.session_state.user_data['weight'] = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=float(st.session_state.user_data['weight']))
                st.session_state.user_data['height'] = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=float(st.session_state.user_data['height']))
                
                # Calculate BMI
                st.session_state.user_data['bmi'] = calculate_bmi(st.session_state.user_data['weight'], st.session_state.user_data['height'])
                
                # Display BMI
                bmi = st.session_state.user_data['bmi']
                if bmi < 18.5:
                    bmi_category = "Underweight"
                    bmi_color = "blue"
                elif bmi < 25:
                    bmi_category = "Normal weight"
                    bmi_color = "green"
                elif bmi < 30:
                    bmi_category = "Overweight"
                    bmi_color = "orange"
                else:
                    bmi_category = "Obese"
                    bmi_color = "red"
                
                st.markdown(f"<p>BMI: <span style='color:{bmi_color};font-weight:bold'>{bmi:.1f} ({bmi_category})</span></p>", unsafe_allow_html=True)
            
            with col2:
                st.session_state.user_data['blood_sugar'] = st.number_input("Blood Sugar Level (mg/dL)", min_value=50, max_value=400, value=st.session_state.user_data['blood_sugar'])
                blood_sugar_type = st.radio("Blood Sugar Measurement Type", ["Fasting", "Post-meal"])
                blood_sugar_interpretation = interpret_blood_sugar(st.session_state.user_data['blood_sugar'], blood_sugar_type)
                
                if blood_sugar_interpretation == "Low":
                    bs_color = "blue"
                elif blood_sugar_interpretation == "Normal":
                    bs_color = "green"
                elif blood_sugar_interpretation == "Prediabetes":
                    bs_color = "orange"
                else:
                    bs_color = "red"
                
                st.markdown(f"<p>Interpretation: <span style='color:{bs_color};font-weight:bold'>{blood_sugar_interpretation}</span></p>", unsafe_allow_html=True)
                
                st.session_state.user_data['hba1c'] = st.number_input("HbA1c Level (%)", min_value=4.0, max_value=15.0, value=float(st.session_state.user_data['hba1c']), step=0.1)
                st.session_state.user_data['diabetes_type'] = st.selectbox("Type of Diabetes", ['Type 1', 'Type 2', 'Gestational', 'Prediabetes'], index=['Type 1', 'Type 2', 'Gestational', 'Prediabetes'].index(st.session_state.user_data['diabetes_type']))
            
            st.session_state.user_data['activity_level'] = st.select_slider("Daily Activity Level", options=['Sedentary', 'Light', 'Moderate', 'Active', 'Very Active'], value=st.session_state.user_data['activity_level'])
            st.session_state.user_data['food_allergies'] = st.text_area("Food Allergies or Restrictions", st.session_state.user_data['food_allergies'])
        
        with tab2:
            st.markdown("<h2 class='sub-header'>Socioeconomic Factors</h2>", unsafe_allow_html=True)
            st.info("This information helps us create a nutrition plan that's practical and accessible for your specific situation.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.user_data['income_level'] = st.selectbox("Income Level", ['Low', 'Middle', 'High'], index=['Low', 'Middle', 'High'].index(st.session_state.user_data['income_level']))
                
                location_options = [
                    'Rural North America', 'Urban North America', 
                    'Rural Europe', 'Urban Europe', 
                    'Rural Asia', 'Urban Asia', 
                    'Rural Africa', 'Urban Africa', 
                    'Rural South America', 'Urban South America', 
                    'Rural Australia/Oceania', 'Urban Australia/Oceania'
                ]
                # For Mauritania-specific regions
                mauritania_regions = list(mauritania_food_data["regions"].keys())
                st.session_state.user_data['mauritania_region'] = st.selectbox(
                    "Mauritania Region", 
                    mauritania_regions, 
                    index=mauritania_regions.index(st.session_state.user_data['mauritania_region']) if st.session_state.user_data['mauritania_region'] in mauritania_regions else 0
                )
                
                st.session_state.user_data['education'] = st.selectbox(
                    "Education Level", 
                    ['No formal education', 'Primary/Elementary', 'Secondary/High School', 'College/University', 'Post-graduate'],
                    index=['No formal education', 'Primary/Elementary', 'Secondary/High School', 'College/University', 'Post-graduate'].index(st.session_state.user_data['education'])
                )
            
            with col2:
                st.session_state.user_data['literacy'] = st.selectbox(
                    "Literacy Level", 
                    ['Limited/None', 'Basic', 'Intermediate', 'Proficient'],
                    index=['Limited/None', 'Basic', 'Intermediate', 'Proficient'].index(st.session_state.user_data['literacy'])
                )
                
                st.session_state.user_data['cooking_access'] = st.selectbox(
                    "Access to Cooking Facilities", 
                    ['Limited/None', 'Basic (hotplate only)', 'Moderate (stovetop)', 'Full kitchen'],
                    index=['Limited/None', 'Basic (hotplate only)', 'Moderate (stovetop)', 'Full kitchen'].index(st.session_state.user_data['cooking_access'])
                )
                
                st.session_state.user_data['refrigeration'] = st.radio("Access to Refrigeration", ['Yes', 'No', 'Limited'], index=['Yes', 'No', 'Limited'].index(st.session_state.user_data['refrigeration']) if st.session_state.user_data['refrigeration'] in ['Yes', 'No', 'Limited'] else 0)
            
            # Display available foods in the selected region
            selected_region = st.session_state.user_data['mauritania_region']
            region_foods = mauritania_food_data["regions"][selected_region]["common_foods"]
            food_names = [food["name"] for food in region_foods]
            
            st.markdown(f"### Available Foods in {selected_region}")
            st.markdown(", ".join(food_names))
            
            # Display diabetes-friendly options
            st.markdown("### Diabetes-Friendly Options")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Highly Recommended**")
                st.markdown("\n".join([f"- {food}" for food in mauritania_food_data["diabetes_friendly_options"]["high_recommendation"]]))
            with col2:
                st.markdown("**Moderate Consumption**")
                st.markdown("\n".join([f"- {food}" for food in mauritania_food_data["diabetes_friendly_options"]["moderate_consumption"]]))
            with col3:
                st.markdown("**Limited Consumption**")
                st.markdown("\n".join([f"- {food}" for food in mauritania_food_data["diabetes_friendly_options"]["limited_consumption"]]))
        
        with tab3:
            st.markdown("<h2 class='sub-header'>Generate Your Personalized Plan</h2>", unsafe_allow_html=True)
            
            # Display summary of entered information
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Summary of Your Information</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Health Information")
                st.markdown(f"- **Age:** {st.session_state.user_data['age']}")
                st.markdown(f"- **Gender:** {st.session_state.user_data['gender']}")
                st.markdown(f"- **BMI:** {st.session_state.user_data['bmi']:.1f}")
                st.markdown(f"- **Blood Sugar:** {st.session_state.user_data['blood_sugar']} mg/dL")
                st.markdown(f"- **HbA1c:** {st.session_state.user_data['hba1c']}%")
                st.markdown(f"- **Diabetes Type:** {st.session_state.user_data['diabetes_type']}")
                st.markdown(f"- **Activity Level:** {st.session_state.user_data['activity_level']}")
            
            with col2:
                st.markdown("#### Socioeconomic Information")
                st.markdown(f"- **Income Level:** {st.session_state.user_data['income_level']}")
                st.markdown(f"- **Location:** {st.session_state.user_data['mauritania_region']}, Mauritania")
                st.markdown(f"- **Education:** {st.session_state.user_data['education']}")
                st.markdown(f"- **Literacy:** {st.session_state.user_data['literacy']}")
                st.markdown(f"- **Cooking Access:** {st.session_state.user_data['cooking_access']}")
                st.markdown(f"- **Refrigeration:** {st.session_state.user_data['refrigeration']}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            # Generate plan button
            if st.button(get_translation("generate_button"), type="primary"):
                with st.spinner(get_translation("generating")):
                    # Check if API key is available
                    if not os.getenv("OPENAI_API_KEY"):
                        st.error(get_translation("api_error"))
                    else:
                        # Generate the nutrition plan with Mauritanian food data
                        nutrition_plan = generate_nutrition_plan(st.session_state.user_data, mauritania_food_data)
                        
                        if nutrition_plan:
                            st.session_state.nutrition_plan = nutrition_plan
                            
                            # If literacy level is basic or limited, generate visual guide
                            if st.session_state.user_data['literacy'] in ['Limited/None', 'Basic']:
                                with st.spinner(get_translation("visual_generating")):
                                    visual_description, visual_guide_data = generate_visual_guide(st.session_state.user_data, nutrition_plan, mauritania_food_data)
                                    if visual_description:
                                        st.session_state.visual_description = visual_description
                                    if visual_guide_data:
                                        st.session_state.visual_guide_data = visual_guide_data
            
            # Display the generated nutrition plan if available
            if 'nutrition_plan' in st.session_state:
                st.markdown(f"<h3>{get_translation('plan_header')}</h3>", unsafe_allow_html=True)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(st.session_state.nutrition_plan)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Download option
                plan_text = st.session_state.nutrition_plan
                st.download_button(
                    label=get_translation("download_button"),
                    data=plan_text,
                    file_name="my_diabetes_nutrition_plan.txt",
                    mime="text/plain"
                )
                
                # Display visual guide if available for low literacy users
                if st.session_state.user_data['literacy'] in ['Limited/None', 'Basic'] and 'visual_guide_data' in st.session_state:
                    st.markdown("<h2 style='text-align:center; color:#2E7D32; padding:10px; margin-top:20px; border-bottom:2px solid #2E7D32;'>Visual Nutrition Guide</h2>", unsafe_allow_html=True)
                    
                    # Create a better-looking plate model visualization
                    st.markdown("<h3 style='text-align:center; color:#424242; margin-top:30px;'>Ideal Meal Plate</h3>", unsafe_allow_html=True)
                    
                    # Create a better-looking plate using matplotlib
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        # Create plate visualization using matplotlib
                        plate_fig, ax = plt.subplots(figsize=(6, 6))
                        
                        # Create a circle for the plate
                        circle = plt.Circle((0.5, 0.5), 0.45, fill=False, color='black', linewidth=2)
                        ax.add_patch(circle)
                        
                        # Create the sections
                        # Half for vegetables
                        veg_section = plt.Rectangle((0.05, 0.5), 0.9, 0.45, color='#66BB6A', alpha=0.85)
                        ax.add_patch(veg_section)
                        
                        # Quarter for grains
                        grain_section = plt.Rectangle((0.05, 0.05), 0.45, 0.45, color='#FFF176', alpha=0.85)
                        ax.add_patch(grain_section)
                        
                        # Quarter for protein
                        protein_section = plt.Rectangle((0.5, 0.05), 0.45, 0.45, color='#EF5350', alpha=0.85)
                        ax.add_patch(protein_section)
                        
                        # Add text labels
                        plt.text(0.5, 0.73, 'Vegetables', horizontalalignment='center', fontsize=14, fontweight='bold')
                        plt.text(0.27, 0.27, 'Grains', horizontalalignment='center', fontsize=14, fontweight='bold')
                        plt.text(0.73, 0.27, 'Protein', horizontalalignment='center', fontsize=14, fontweight='bold')
                        
                        # Set limits and remove axes
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis('off')
                        
                        # Draw and display the figure
                        plt.tight_layout()
                        st.pyplot(plate_fig)
                    
                    # Display better portion guides
                    st.markdown("<h3 style='text-align:center; color:#424242; margin-top:20px;'>Portion Size Guide</h3>", unsafe_allow_html=True)
                    
                    # Create a container with background
                    st.markdown("""
                    <style>
                    .portion-container {
                        background-color: #f9f9f9;
                        padding: 15px;
                        border-radius: 10px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        margin-bottom: 30px;
                    }
                    .portion-box {
                        background-color: white;
                        border-radius: 10px;
                        padding: 15px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        text-align: center;
                        height: 100%;
                    }
                    .portion-emoji {
                        font-size: 48px;
                        margin-bottom: 10px;
                    }
                    .portion-title {
                        font-weight: bold;
                        font-size: 18px;
                        color: #424242;
                        margin-bottom: 5px;
                    }
                    .portion-desc {
                        color: #616161;
                        font-size: 14px;
                    }
                    </style>
                    <div class="portion-container">
                    """, unsafe_allow_html=True)
                    
                    portion_cols = st.columns(4)
                    
                    with portion_cols[0]:
                        st.markdown("""
                        <div class="portion-box">
                            <div class="portion-emoji">✋</div>
                            <div class="portion-title">Palm</div>
                            <div class="portion-desc">Size of protein<br>(meat, fish)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with portion_cols[1]:
                        st.markdown("""
                        <div class="portion-box">
                            <div class="portion-emoji">👊</div>
                            <div class="portion-title">Fist</div>
                            <div class="portion-desc">Size of grains<br>(rice, millet)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with portion_cols[2]:
                        st.markdown("""
                        <div class="portion-box">
                            <div class="portion-emoji">👍</div>
                            <div class="portion-title">Thumb</div>
                            <div class="portion-desc">Size of fats<br>(oils)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with portion_cols[3]:
                        st.markdown("""
                        <div class="portion-box">
                            <div class="portion-emoji">🖐️</div>
                            <div class="portion-title">Open Hand</div>
                            <div class="portion-desc">Size of vegetables</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display Green Foods with better styling
                    st.markdown("""
                    <style>
                    .food-category-header {
                        display: flex;
                        align-items: center;
                        margin-bottom: 15px;
                        padding: 10px;
                        border-radius: 5px;
                    }
                    .green-header {
                        background-color: #E8F5E9;
                        border-left: 6px solid #4CAF50;
                    }
                    .yellow-header {
                        background-color: #FFF8E1;
                        border-left: 6px solid #FFC107;
                    }
                    .red-header {
                        background-color: #FFEBEE;
                        border-left: 6px solid #F44336;
                    }
                    .header-icon {
                        font-size: 26px;
                        margin-right: 10px;
                        margin-left: 5px;
                    }
                    .header-text {
                        font-size: 22px;
                        font-weight: bold;
                        color: #424242;
                    }
                    .food-card {
                        background: white;
                        border-radius: 10px;
                        padding: 10px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        text-align: center;
                        transition: transform 0.2s;
                        height: 100%;
                    }
                    .food-card:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    }
                    .food-img {
                        width: 130px;
                        height: 130px;
                        object-fit: cover;
                        border-radius: 5px;
                        margin: 0 auto 10px auto;
                        display: block;
                    }
                    .food-name {
                        font-weight: bold;
                        color: #424242;
                        margin-top: 5px;
                        font-size: 16px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Green Foods Section
                    st.markdown("""
                    <div class="food-category-header green-header">
                        <span class="header-icon">✅</span>
                        <span class="header-text">Green Foods - Eat Freely</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    green_foods = st.session_state.visual_guide_data["green_foods"]
                    if green_foods:
                        # Calculate number of columns (3 items per row)
                        items_per_row = 3
                        rows = (len(green_foods) + items_per_row - 1) // items_per_row  # Ceiling division
                        
                        for row in range(rows):
                            cols = st.columns(items_per_row)
                            for col in range(items_per_row):
                                idx = row * items_per_row + col
                                if idx < len(green_foods):
                                    food = green_foods[idx]
                                    with cols[col]:
                                        try:
                                            image_path = food.get('image_url', '')
                                            # Try to load the image if it exists
                                            try:
                                                st.markdown(f"""
                                                <div class="food-card">
                                                    <img class="food-img" src="{image_path}" alt="{food['name']}">
                                                    <div class="food-name">{food['name']}</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            except:
                                                # If image fails, display a colored box with the name
                                                st.markdown(f"""
                                                <div class="food-card">
                                                    <div style="width:130px;height:130px;background:#E8F5E9;display:flex;align-items:center;
                                                    justify-content:center;border-radius:5px;margin:0 auto;border:1px solid #4CAF50">
                                                        <span style="font-weight:bold;text-align:center;">{food['name']}</span>
                                                    </div>
                                                    <div class="food-name">{food['name']}</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                        except Exception as e:
                                            st.write(food['name'])
                    else:
                        st.info("No green foods identified in your region")
                    
                    # Yellow Foods Section
                    st.markdown("""
                    <div class="food-category-header yellow-header" style="margin-top:30px;">
                        <span class="header-icon">⚠️</span>
                        <span class="header-text">Yellow Foods - Eat in Moderation</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    yellow_foods = st.session_state.visual_guide_data["yellow_foods"]
                    if yellow_foods:
                        # Calculate number of columns (3 items per row)
                        items_per_row = 3
                        rows = (len(yellow_foods) + items_per_row - 1) // items_per_row
                        
                        for row in range(rows):
                            cols = st.columns(items_per_row)
                            for col in range(items_per_row):
                                idx = row * items_per_row + col
                                if idx < len(yellow_foods):
                                    food = yellow_foods[idx]
                                    with cols[col]:
                                        try:
                                            image_path = food.get('image_url', '')
                                            # Try to load the image if it exists
                                            try:
                                                st.markdown(f"""
                                                <div class="food-card">
                                                    <img class="food-img" src="{image_path}" alt="{food['name']}">
                                                    <div class="food-name">{food['name']}</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            except:
                                                # If image fails, display a colored box with the name
                                                st.markdown(f"""
                                                <div class="food-card">
                                                    <div style="width:130px;height:130px;background:#FFF8E1;display:flex;align-items:center;
                                                    justify-content:center;border-radius:5px;margin:0 auto;border:1px solid #FFC107">
                                                        <span style="font-weight:bold;text-align:center;">{food['name']}</span>
                                                    </div>
                                                    <div class="food-name">{food['name']}</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                        except Exception as e:
                                            st.write(food['name'])
                    else:
                        st.info("No yellow foods identified in your region")
                    
                    # Red Foods Section
                    st.markdown("""
                    <div class="food-category-header red-header" style="margin-top:30px;">
                        <span class="header-icon">⛔</span>
                        <span class="header-text">Red Foods - Limit Consumption</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    red_foods = st.session_state.visual_guide_data["red_foods"]
                    if red_foods:
                        # Calculate number of columns (3 items per row)
                        items_per_row = 3
                        rows = (len(red_foods) + items_per_row - 1) // items_per_row
                        
                        for row in range(rows):
                            cols = st.columns(items_per_row)
                            for col in range(items_per_row):
                                idx = row * items_per_row + col
                                if idx < len(red_foods):
                                    food = red_foods[idx]
                                    with cols[col]:
                                        try:
                                            image_path = food.get('image_url', '')
                                            # Try to load the image if it exists
                                            try:
                                                st.markdown(f"""
                                                <div class="food-card">
                                                    <img class="food-img" src="{image_path}" alt="{food['name']}">
                                                    <div class="food-name">{food['name']}</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            except:
                                                # If image fails, display a colored box with the name
                                                st.markdown(f"""
                                                <div class="food-card">
                                                    <div style="width:130px;height:130px;background:#FFEBEE;display:flex;align-items:center;
                                                    justify-content:center;border-radius:5px;margin:0 auto;border:1px solid #F44336">
                                                        <span style="font-weight:bold;text-align:center;">{food['name']}</span>
                                                    </div>
                                                    <div class="food-name">{food['name']}</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                        except Exception as e:
                                            st.write(food['name'])
                    else:
                        st.info("No red foods identified in your region")
                    
                    # Display reminders with better styling
                    st.markdown("""
                    <h3 style='text-align:center; color:#424242; margin-top:40px; margin-bottom:20px;'>Important Reminders</h3>
                    <style>
                    .reminder-container {
                        background-color: #E3F2FD;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        margin-bottom: 30px;
                    }
                    .reminder-item {
                        background-color: white;
                        border-radius: 10px;
                        padding: 15px;
                        text-align: center;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        height: 100%;
                    }
                    .reminder-icon {
                        font-size: 36px;
                        margin-bottom: 10px;
                    }
                    .reminder-text {
                        color: #424242;
                        font-size: 14px;
                        line-height: 1.4;
                    }
                    </style>
                    <div class="reminder-container">
                    """, unsafe_allow_html=True)
                    
                    reminder_cols = st.columns(4)
                    
                    with reminder_cols[0]:
                        st.markdown("""
                        <div class="reminder-item">
                            <div class="reminder-icon">🍽️</div>
                            <div class="reminder-text">Eat 3 meals at regular times each day</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with reminder_cols[1]:
                        st.markdown("""
                        <div class="reminder-item">
                            <div class="reminder-icon">🩸</div>
                            <div class="reminder-text">Check blood sugar before and after meals</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with reminder_cols[2]:
                        st.markdown("""
                        <div class="reminder-item">
                            <div class="reminder-icon">💧</div>
                            <div class="reminder-text">Drink water instead of sweet drinks</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with reminder_cols[3]:
                        st.markdown("""
                        <div class="reminder-item">
                            <div class="reminder-icon">🚶</div>
                            <div class="reminder-text">Walk for 30 minutes every day</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    if 'visual_description' in st.session_state:
                        with st.expander("Visual Guide Description (for healthcare providers)"):
                            st.markdown(st.session_state.visual_description)
    
    elif choice == "About":
        st.markdown("<h1 class='main-header'>About This Project</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        ### Project Overview
        
        This AI-powered system creates personalized nutrition plans for individuals with diabetes. 
        The system considers various health-related factors such as medical analyses, weight, 
        dietary habits, and calorie consumption. Additionally, it accounts for socioeconomic 
        factors such as income, location, education, and literacy levels to ensure that the 
        plan is accessible and effective for a wide range of users, particularly those in 
        rural or underserved communities.
        
        ### Key Features
        
        - **Personalized Nutrition Planning**: Generate customized nutrition plans based on individual health metrics and socioeconomic factors
        - **Contextual Customization**: Adapts plans based on literacy and education levels, using visual representations for those with limited literacy
        - **Accessibility Focus**: Designed specifically to support underserved populations including rural communities
        - **Cultural Sensitivity**: Considers local food availability and cultural preferences
        
        ### Technical Implementation
        
        - Built with Python and Streamlit for a responsive web interface
        - Utilizes OpenAI's GPT-4 for generating personalized nutrition recommendations
        - Implements DALL-E for creating visual nutrition guides
        - Processes healthcare data with pandas and numpy
        
        ### Ethical Considerations
        
        This tool is designed to supplement, not replace, professional medical advice. Always consult with healthcare providers for proper diabetes management. The system prioritizes data privacy and security while ensuring accessibility across different socioeconomic backgrounds.
        """)
    
    elif choice == "Help":
        st.markdown("<h1 class='main-header'>Help & Frequently Asked Questions</h1>", unsafe_allow_html=True)
        
        with st.expander("How do I use this application?"):
            st.markdown("""
            1. Navigate to the "Create Nutrition Plan" page using the sidebar
            2. Fill in your health information in the first tab
            3. Provide socioeconomic details in the second tab
            4. Review your information and generate your plan in the third tab
            5. Download your plan or view the visual guide if applicable
            """)
        
        with st.expander("Do I need an OpenAI API key?"):
            st.markdown("""
            Yes, this application requires an OpenAI API key to generate personalized nutrition plans.
            
            You can enter your API key in the sidebar. The key is used only for generating your nutrition plan and is not stored permanently.
            
            To get an API key, visit [OpenAI's website](https://platform.openai.com/account/api-keys).
            """)
        
        with st.expander("Is my data secure?"):
            st.markdown("""
            Your health and personal information is not stored on any server. The data exists only within your current browser session and is used solely to generate your nutrition plan.
            
            When you close the application, your data is cleared. For subsequent sessions, you'll need to re-enter your information.
            """)
        
        with st.expander("What if I have specific dietary requirements?"):
            st.markdown("""
            You can specify food allergies or dietary restrictions in the "Food Allergies or Restrictions" field in the Health Information tab. The system will take these into account when generating your plan.
            
            For very specific medical dietary requirements, always consult with a healthcare professional.
            """)
        
        with st.expander("How accurate is the nutrition plan?"):
            st.markdown("""
            The nutrition plans are generated based on general guidelines for diabetes management and the information you provide. While the system uses advanced AI to create personalized recommendations, it is not a substitute for professional medical advice.
            
            Always consult with a healthcare provider or registered dietitian for diabetes management.
            """)
        
        st.markdown("### Contact Support")
        st.markdown("If you need additional help or have questions not covered here, please contact support at diabetes.nutrition.support@example.com.")
        
        # Provide instructions for healthcare providers
        st.markdown("### For Healthcare Providers")
        st.markdown("""
        If you're a healthcare provider using this tool with patients:
        
        1. Help patients enter accurate health information
        2. Review the generated nutrition plans before implementing
        3. Use the visual guides as educational tools during consultations
        4. Provide feedback on the system to help us improve
        """)

if __name__ == "__main__":
    main()
