import pandas as pd
import streamlit as st
import requests
import random
import streamlit as st
from streamlit_option_menu import option_menu
import base64

def get_base64_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

df = pd.read_csv("imdb_finalFR5.csv")

st.set_page_config(layout="wide")
with st.sidebar:
    
    
    # Code hexadécimal pour un Jaune Fauve/Ocre :
    COULEUR_FAUVE ='#DAA520'
    COULEUR_NOIRE ='black' # Couleur au survol (hover) : noir
    selection = option_menu(
        menu_title=None,
        options = ["Accueil", "recherche de films", "Statistiques"],
        icons = ["house-check-fill", "image-alt",""],
        orientation="vertical", # Maintenez cette orientation pour un look de sidebar
        key="main_menu",
        
        
        styles={
            # --- 1. ÉTAT INACTIF (Parfait, icône/texte en Jaune Fauve) ---
            "nav-link": {
                "color": COULEUR_FAUVE, 
                "font-size": "16px",
                "padding-right": "10px",
            },
            
            # --- 2. ÉTAT SÉLECTIONNÉ (CORRIGÉ : Jaune Fauve en fond, Noir en texte) ---
            "nav-link-selected": {
                "background-color": COULEUR_FAUVE, # Nouvelle couleur de fond : Jaune Fauve
                "color": COULEUR_NOIRE,           # Nouvelle couleur de texte/icône : Noir
            },
            
            # --- 3. ÉTAT SURVOLÉ (Parfait, icône/texte en Noir) ---
            "nav-link:hover": {
                "color": COULEUR_NOIRE, 
            },
        }
    )

# ---On indique au programme quoi faire en fonction du choix---


if selection == "Accueil":
#--- Contenu principal de la page "Accueil" ---
    # Titre principal de l'application (affiché en haut de la page)

    st.title("Choisir son prochain film")
    #st.write(df.columns)
    API_KEY = "a5159d7e870a7a2aef7170e6c50cf186"

    url = f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=fr-FR"
    data = requests.get(url).json()

    # Récupérer les posters valides
    posters = [
        "https://image.tmdb.org/t/p/w300" + m["poster_path"]
        for m in data["results"]
        if m["poster_path"] is not None
    ]
    # Mélange aléatoire
    random.shuffle(posters)
    # Garde seulement les 12 premiers
    posters = posters[:12]


    # CSS identique
    carousel_css = """
    <style>
.carousel {
    display: flex;
    overflow-x: auto;  /* important : auto = scroll fonctionne */
    gap: 16px;
    padding: 16px;
    width: 100%;
    -webkit-overflow-scrolling: touch; /* améliore le scroll */
}

/* cache seulement la scrollbar MAIS la laisse active */
.carousel::-webkit-scrollbar {
    height: 0px; /* cache la scrollbar mais garde le scroll */
}

.carousel img {
    height: 220px;
    border-radius: 10px;
    transition: transform 0.2s;
}
.carousel img:hover {
    transform: scale(1.12);
    cursor: pointer;
}
</style>
"""

    st.markdown(carousel_css, unsafe_allow_html=True)

    # HTML du carrousel
    html = '<div class="carousel">'
    for url in posters:
        html += f'<img src="{url}">'
    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)


elif selection == "recherche de films":

    st.title("🔎 Outil de recommandation de films")

    q = st.text_input("Saisissez le titre d'un film :", "")

    if q:
        # 1️⃣ recherche exacte (insensible à la casse)
        film = df[
            (df['primaryTitle'].str.lower() == q.lower()) |
            (df['originalTitle'].str.lower() == q.lower())
        ]

        # 2️⃣ recherche "contient"
        if film.empty:
            film = df[
                df['primaryTitle'].str.lower().str.contains(q.lower(), na=False) |
                df['originalTitle'].str.lower().str.contains(q.lower(), na=False)
            ]

        # --- Si trouvé ---
        if not film.empty:
            st.success("Film trouvé ! 🎉")
            st.dataframe(film)

            # Récupération sécurisée
            acteurs = film.iloc[0]['actors'] if 'actors' in df.columns else "Non renseigné"
            genre = film.iloc[0]['genre'] if 'genre' in df.columns else "Non renseigné"

            st.subheader("🎭 Acteurs principaux")
            st.write(acteurs)

            st.subheader("🎬 Genre")
            st.write(genre)
            # --- Si aucun résultat ---
        else:
            st.error("Aucun film trouvé. Vérifiez l’orthographe ou essayez un mot-clé.")


elif selection == "Statistiques":
    st.title("Statistiques des films")
    st.write(
        pd.read_csv("imdb_finalFR5.csv").describe()     
    )
# Affiche un dataframe (st.write accepte plusieurs arguments et plusieurs types de données)
    st.write(
        pd.read_csv("imdb_finalFR5.csv").sample(15)     
    )
    
    

    
# --- 2. Injection du CSS ---
img_base64 = get_base64_image("images/salle_jaune_sombre.png")
header_b64 = get_base64_image("images/bandeau.png")

st.markdown(
    f"""
    <style>
    /* Fond de l'app */
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}

    /* Bandeau fixe en haut */
    .header-div {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 120px;
        background-image: url("data:image/png;base64,{header_b64}");
        background-size: cover;
        background-position: center;
        z-index: 999;
    }}

    /* Barre Streamlit : transparente et derrière le bandeau */
    header[data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0) !important;
        z-index: 0 !important;
    }}
    header[data-testid="stHeader"] * {{
        visibility: hidden; /* si tu veux supprimer tout le texte “Deploy” */
    }}

    /* Contenu sous le bandeau */
    .block-container {{
        background: rgba(0,0,0,0) !important;
        padding-top: 130px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Injection du div bandeau fixe
st.markdown('<div class="header-div"></div>', unsafe_allow_html=True)
