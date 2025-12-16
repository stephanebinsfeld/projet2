# ---------------------------
# Projet 2 - Choisir son prochain film
# ---------------------------
# Importations  
# ---------------------------

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
import random
from streamlit_option_menu import option_menu
import base64
import streamlit.components.v1 as components
import urllib.parse

def get_base64_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ---------------------------
# choix de la base de données
# ---------------------------

df = pd.read_csv("imdb_final.csv")
df['decade'] = (df['startYear'] // 10) * 10

# ---------------------------
# MODEL DE RECOMMANDATION
# ---------------------------

import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Combine features pour TF-IDF
df['combined'] = (
    df['genres'].astype(str) + " " +
    df['overview'].astype(str) + " " +
    df['actors'].astype(str) + " " +
    df['directors'].astype(str)
)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movies(title, top_n=5):
    titles = df['primaryTitle'].fillna("").values

    match = difflib.get_close_matches(title, titles, n=1, cutoff=0.6)
    if not match:
        return pd.DataFrame()

    title = match[0]
    idx = df[df['primaryTitle'] == title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_idx = [i for i, score in sim_scores]
    return df.iloc[movie_idx]

# ---------------------------
# GESTION DES CLICS CAROUSEL (DOIT ÊTRE ICI, AVANT LA SIDEBAR)
# ---------------------------

# Initialiser selection dans session_state si nécessaire
if 'selection' not in st.session_state:
    st.session_state['selection'] = "Accueil"

# Vérifier si on vient d'un clic sur le carousel
query_params = st.query_params
if 'movie_title' in query_params and 'goto' in query_params:
    if query_params['goto'] == 'search':
        st.session_state['search_query'] = query_params['movie_title']
        st.session_state['selection'] = "recherche de films"
        # Nettoyer les paramètres de l'URL
        st.query_params.clear()
        st.rerun()

# ---------------------------
# Paramètres sidebar et menu
# ---------------------------

st.set_page_config(layout="wide")
with st.sidebar:
    
    COULEUR_FAUVE ='#DAA520'
    COULEUR_NOIRE ='black'
    
    # Utiliser la valeur de session_state pour le menu
    selection = option_menu(
        menu_title=None,
        options = ["Accueil", "recherche de films", "Base de données"],
        icons = ["house-check-fill", "image-alt",""],
        orientation="vertical",
        key="main_menu",
        default_index=["Accueil", "recherche de films", "Base de données"].index(st.session_state['selection']),
        styles={
            "nav-link": {
                "color": COULEUR_FAUVE, 
                "font-size": "16px",
                "padding-right": "10px",
            },
            "nav-link-selected": {
                "background-color": COULEUR_FAUVE,
                "color": COULEUR_NOIRE,
            },
            "nav-link:hover": {
                "color": COULEUR_NOIRE, 
            },
        }
    )
    
    # Mettre à jour session_state avec la sélection du menu
    st.session_state['selection'] = selection

# ---------------------------
# ACCUEIL AVEC CAROUSEL CLIQUABLE
# ---------------------------

if selection == "Accueil":
    st.title("🎬 Bienvenue sur Cinéma Conseil")
    
    # Préparer les données du carousel
    posters_valides = df['poster_path'].dropna().tolist()

    if posters_valides:
        random.shuffle(posters_valides)
        poster_data = []
        
        for poster_url in posters_valides[:15]:
            film = df[df['poster_path'] == poster_url].iloc[0]
            poster_data.append({
                'url': poster_url,
                'title': film['primaryTitle']
            })
    else:
        st.warning("Aucun chemin de poster valide n'a été trouvé.")
        poster_data = []

    # CSS pour le carousel avec liens
    carousel_css = """
    <style>
    .carousel {
        display: flex;
        overflow-x: auto;
        gap: 16px;
        padding: 16px;
        width: 100%;
        -webkit-overflow-scrolling: touch;
    }

    .carousel::-webkit-scrollbar {
        height: 0px;
    }
    
    .carousel a {
        flex-shrink: 0;
        text-decoration: none;
    }

    .carousel img {
        height: 400px;
        border-radius: 10px;
        transition: transform 0.2s;
        display: block;
    }
    
    .carousel a:hover img {
        transform: scale(1.12);
        cursor: pointer;
    }
    </style>
    """

    st.markdown(carousel_css, unsafe_allow_html=True)

    # Construire l'URL de base de l'application
    # Récupérer l'URL actuelle sans les paramètres
    current_url = "?"  # Relatif pour éviter les problèmes
    
    # Affichage du carousel avec liens
    html = '<div class="carousel">'
    for poster_info in poster_data:
        # Encoder le titre pour l'URL
        import urllib.parse
        encoded_title = urllib.parse.quote(poster_info['title'])
        
        # Créer un lien avec les paramètres
        link_url = f"{current_url}movie_title={encoded_title}&goto=search"
        
        html += f'<a href="{link_url}"><img src="{poster_info["url"]}" alt="{poster_info["title"]}"></a>'
    
    html += "</div>"
    
    st.markdown(html, unsafe_allow_html=True)
# ---------------------------
# PAGE RECHERCHE
# ---------------------------
elif selection == "recherche de films":
    st.title("🔍 Recherche de films")
    
    # Initialisation de l'état de recherche si ce n'est pas déjà fait
    if 'search_query' not in st.session_state:
        st.session_state['search_query'] = ""

    # Déterminer la valeur initiale du champ de recherche
    # Si nous venons de la page d'accueil, elle contient le titre du film
    initial_query = st.session_state.get('search_query', '')
    
    # Choix du mode de recherche
    mode = st.radio(
        "Choisissez votre méthode de recherche :",
        ("Recherche par titre", "Recherche par filtres", "Film aléatoire"),
        horizontal=True
    )

    # --------------------------------------------------------------------
    # 1 RECHERCHE PAR TITRE
    # --------------------------------------------------------------------
    if mode == "Recherche par titre":
        
        # 1. INITIALISATION ET RÉCUPÉRATION DE LA REQUÊTE
        # Récupère la requête stockée après un clic sur l'accueil, sinon c'est une chaîne vide.
        initial_query = st.session_state.get('search_query', '')
        
        # Champ de saisie : utilise 'initial_query' comme valeur initiale.
        query = st.text_input("Titre du film :", value=initial_query, key="title_search_input")

        # 2. LOGIQUE D'EXÉCUTION DE LA RECHERCHE
        # La recherche s'exécute si :
        # a) L'utilisateur a tapé une requête (query est vrai)
        # b) OU une requête initiale a été passée par l'état de session (clic sur affiche)
        if query: 
            
            # Important : Vider l'état de la session après l'exécution pour éviter les re-runs automatiques
            # st.session_state['search_query'] = "" # Optionnel : peut être commenté si l'on souhaite que le champ reste prérempli

            # Trouver les résultats
            results = df[
                df['primaryTitle'].str.contains(query, case=False, na=False) |
                df['originalTitle'].str.contains(query, case=False, na=False)
            ]

            # 3. AFFICHAGE DES RÉSULTATS (Votre code d'affichage)
            if not results.empty:
                st.write(f"Résultats pour : **{query}**")

                for index, film in results.iterrows():
                    col1, col2 = st.columns([1, 4])

                    with col1:
                        poster = film['poster_path'] if pd.notna(film['poster_path']) else "placeholder.png"
                        st.image(poster, use_container_width=True)

                    with col2:
                        st.markdown(f"### **{film['originalTitle']}**")
                        st.write(f"**Année :** {int(film['startYear'])}")
                        st.write(f"**Note IMDb :** {film['averageRating']} ⭐ ({int(film['numVotes'])} votes)")

                        # Nettoyage
                        directors_clean = film["directors"].strip("[]").replace("'", "")
                        actors_clean = film["actors"].strip("[]").replace("'", "")

                        st.write(f"**Producteur :** {directors_clean}")
                        st.write(f"**Distribution :** {actors_clean}")

                        # Résumé
                        with st.expander("Voir le résumé"):
                            st.write(film["overview"])

                        # Films recommandés juste en dessous du résumé
                        st.subheader("🎯 Films recommandés")
                        reco = recommend_movies(film['primaryTitle'], top_n=5)

                        if reco.empty:
                            st.info("Pas de recommandations disponibles.")
                        else:
                            for idx2, film2 in reco.iterrows():
                                with st.expander(f"{film2['originalTitle']} ({int(film2['startYear'])})"):
                                    rcol1, rcol2 = st.columns([1, 4])

                                    with rcol1:
                                        poster2 = film2['poster_path'] if pd.notna(film2['poster_path']) else "placeholder.png"
                                        st.image(poster2, use_container_width=True)

                                    with rcol2:
                                        st.write(f"⭐ {film2['averageRating']} — {film2['genres']}")
                                        st.write(f"🎬 Producteur(s) : {film2['directors'].strip("[]").replace("'", "")}")
                                        st.write(f"🎭 Acteur(s) : {film2['actors'].strip("[]").replace("'", "")}")
                                        st.write(film2['overview'])

                        st.markdown("---")

            else:
                st.info("Aucun film trouvé.")
        
        else:
            st.info("Veuillez entrer un titre de film ou cliquer sur une affiche pour lancer la recherche.")
    # --------------------------------------------------------------------
    # 2 RECHERCHE PAR FILTRES
    # --------------------------------------------------------------------
    elif mode == "Recherche par filtres":

        st.subheader("🎛️ Filtres disponibles")

        # ---- FILTRE ACTEURS ----
        all_actors = (
            df["actors"].dropna()
            .apply(lambda x: [a.strip() for a in x.strip("[]").replace("'", "").split(",")])
        )
        list_actors = sorted(set(sum(all_actors, [])))
        selected_actor = st.selectbox("🎭 Choisir un acteur :", ["Aucun"] + list_actors)

        # ---- FILTRE PRODUCTEURS ----
        all_directors = (
            df["directors"].dropna()
            .apply(lambda x: [d.strip() for d in x.strip("[]").replace("'", "").split(",")])
        )
        list_directors = sorted(set(sum(all_directors, [])))
        selected_director = st.selectbox("🎬 Choisir un producteur :", ["Aucun"] + list_directors)

        # ---- FILTRE GENRES ----
        all_genres = df["genres"].dropna().apply(lambda x: x.split(","))
        list_genres = sorted(set(sum(all_genres, [])))
        selected_genre = st.selectbox("🏷 Choisir un genre :", ["Aucun"] + list_genres)

        # ---- FILTRE ANNÉES ----
        min_year = int(df["startYear"].min())
        max_year = int(df["startYear"].max())
        year_range = st.slider(
            "📅 Plage d'années",
            min_value=min_year,
            max_value=max_year,
            value=(1990, 2020)
        )

        # ---- BOUTON DE RECHERCHE ----
        run_search = st.button("🔎 Rechercher")

        if run_search:

            results = df.copy()

            if selected_actor != "Aucun":
                results = results[results["actors"].str.contains(selected_actor, case=False, na=False)]

            if selected_director != "Aucun":
                results = results[results["directors"].str.contains(selected_director, case=False, na=False)]

            if selected_genre != "Aucun":
                results = results[results["genres"].str.contains(selected_genre, case=False, na=False)]

            results = results[
                (results["startYear"] >= year_range[0]) &
                (results["startYear"] <= year_range[1])
            ]

            # ---- AFFICHAGE ----
            st.subheader("🎬 Résultats filtrés")

            if results.empty:
                st.info("Aucun film ne correspond aux filtres.")
            else:
                st.success(f"{len(results)} films trouvés !")

                for index, film in results.iterrows():
                    col1, col2 = st.columns([1, 4])

                    with col1:
                        poster = film['poster_path'] if pd.notna(film['poster_path']) else "placeholder.png"
                        st.image(poster, use_container_width=True)

                    with col2:
                        st.markdown(f"### **{film['originalTitle']}**")
                        st.write(f"**Année :** {int(film['startYear'])}")
                        st.write(f"**Note IMDb :** {film['averageRating']} ⭐")

                        directors_clean = film["directors"].strip("[]").replace("'", "")
                        actors_clean = film["actors"].strip("[]").replace("'", "")

                        st.write(f"**Producteur :** {directors_clean}")
                        st.write(f"**Distribution :** {actors_clean}")

                        # Résumé
                        with st.expander("Voir le résumé"):
                            st.write(film["overview"])

                        # Films recommandés
                        st.subheader("🎯 Films recommandés")
                        reco = recommend_movies(film['primaryTitle'], top_n=5)

                        if reco.empty:
                            st.info("Pas de recommandations disponibles.")
                        else:
                            for idx2, film2 in reco.iterrows():
                                with st.expander(f"{film2['originalTitle']} ({int(film2['startYear'])})"):
                                    rcol1, rcol2 = st.columns([1, 4])

                                    with rcol1:
                                        poster2 = film2['poster_path'] if pd.notna(film2['poster_path']) else "placeholder.png"
                                        st.image(poster2, use_container_width=True)

                                    with rcol2:
                                        st.write(f"⭐ {film2['averageRating']} — {film2['genres']}")
                                        st.write(f"🎬 Producteur(s) : {film2['directors']}")
                                        st.write(f"🎭 Acteur(s) : {film2['actors']}")
                                        st.write(film2['overview'])

                        st.markdown("---")

    # --------------------------------------------------------------------
    # 3 FILM ALÉATOIRE
    # --------------------------------------------------------------------
    elif mode == "Film aléatoire":
        st.subheader("🎲 Un film au hasard...")

        if st.button("Tirer un film au hasard 🎬"):
            film = df.sample(1).iloc[0]

            col1, col2 = st.columns([1, 4])

            with col1:
                poster = film['poster_path'] if pd.notna(film['poster_path']) else "placeholder.png"
                st.image(poster, use_container_width=True)

            with col2:
                st.markdown(f"## **{film['originalTitle']}**")
                st.write(f"**Année :** {int(film['startYear'])}")
                st.write(f"**Note IMDb :** {film['averageRating']} ⭐ ({int(film['numVotes'])} votes)")

                directors_clean = film["directors"].strip("[]").replace("'", "")
                actors_clean = film["actors"].strip("[]").replace("'", "")

                st.write(f"**Producteur :** {directors_clean}")
                st.write(f"**Distribution :** {actors_clean}")

                # Résumé
                with st.expander("Voir le résumé"):
                    st.write(film["overview"])

                # Films recommandés
                st.subheader("🎯 Films recommandés")
                reco = recommend_movies(film['primaryTitle'], top_n=5)

                if reco.empty:
                    st.info("Pas de recommandations disponibles.")
                else:
                    for idx2, film2 in reco.iterrows():
                        with st.expander(f"{film2['originalTitle']} ({int(film2['startYear'])})"):
                            rcol1, rcol2 = st.columns([1, 4])

                            with rcol1:
                                poster2 = film2['poster_path'] if pd.notna(film2['poster_path']) else "placeholder.png"
                                st.image(poster2, use_container_width=True)

                            with rcol2:
                                st.write(f"⭐ {film2['averageRating']} — {film2['genres']}")
                                st.write(f"🎬 Producteur(s) : {film2['directors']}")
                                st.write(f"🎭 Acteur(s) : {film2['actors']}")
                                st.write(film2['overview'])

            st.markdown("---")

# ---------------------------
# PAGE BASE DE DONNÉES
# ---------------------------

elif selection == "Base de données":
    st.title("📂 Base de données des films")
    
    # --- infos et extraits sur la base de données ---
    st.write(
        pd.read_csv("imdb_final.csv").describe()     
    )
# Affiche un dataframe (st.write accepte plusieurs arguments et plusieurs types de données)
    st.write(
        pd.read_csv("imdb_final.csv").sample(15)     
    )
    
    # --- KPIs ---
    st.subheader("📌 KPIs principaux")
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de films", len(df))
    col2.metric("Note moyenne", round(df['averageRating'].mean(), 2))
    col3.metric("Nombre moyen de votes", int(df['numVotes'].mean()))
    plt.style.use('dark_background')
    st.subheader("🎬 Nombre de films par décennie")
    films_par_decade = df.groupby('decade').size()
    fig, ax = plt.subplots(figsize=(10,5))
    films_par_decade.plot(kind='bar', color='gold', ax=ax)
    ax.set_xlabel("Décennie")
    ax.set_ylabel("Nombre de films")
    ax.set_title("Nombre de films par décennie")
    st.pyplot(fig)

    st.subheader("⭐ Note moyenne par décennie")
    note_moyenne = df.groupby('decade')['averageRating'].mean()
    fig2, ax2 = plt.subplots(figsize=(10,5))
    note_moyenne.plot(kind='line', marker='o', color='gold', ax=ax2)
    ax2.set_xlabel("Décennie")
    ax2.set_ylabel("Note moyenne")
    ax2.set_title("Note moyenne des films par décennie")
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader("🏷 Top 10 des genres")
    genres_exp = df.copy()
    genres_exp['genre'] = genres_exp['genres'].str.split(',')
    genres_exp = genres_exp.explode('genre')
    top_genres = genres_exp['genre'].value_counts().head(10)
    fig3, ax3 = plt.subplots(figsize=(10,5))
    top_genres.plot(kind='barh', color='gold', ax=ax3)
    ax3.set_xlabel("Nombre de films")
    ax3.set_ylabel("Genre")
    ax3.set_title("Top 10 des genres")
    st.pyplot(fig3)


# ---------------------------
# Mise en forme de l'application
# ---------------------------

# --- Injection du CSS ---
img_base64 = get_base64_image("images/salle_jaune_sombre.png")
header_b64 = get_base64_image("images/bandeau.png")

st.markdown(
    f"""
    <style>
    /* --------------------------------- */
    /* FOND DE L'APPLICATION (IMAGE) */
    /* --------------------------------- */
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}

    /* --------------------------------- */
    /* BANDEAU FIXE EN HAUT */
    /* --------------------------------- */
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

    /* --------------------------------- */
    /* BARRE STREAMLIT (NATIVE) */
    /* --------------------------------- */
    header[data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0) !important;
        z-index: 0 !important;
    }}
    header[data-testid="stHeader"] * {{
        visibility: hidden; /* si tu veux supprimer tout le texte “Deploy” */
    }}

    /* --------------------------------- */
    /* CONTENU PRINCIPAL (ESPACE DE TRAVAIL) */
    /* --------------------------------- */
    .block-container {{
        background: rgba(0,0,0,0) !important;
        padding-top: 130px; 
    }}

/* --------------------------------- */
/* CAROUSSEL (Restauration à l'original) */
/* --------------------------------- */
.carousel {{
    display: flex;
    overflow-x: auto;
    gap: 16px;
    padding: 16px;
    width: 100%;
    -webkit-overflow-scrolling: touch; 
}}

/* Cache la scrollbar mais laisse le scroll actif */
.carousel::-webkit-scrollbar {{
    height: 0px; 
}}

.carousel img {{
    height: 400px;
    border-radius: 10px;
    transition: transform 0.2s;
    width: auto; /* IMPORTANT : Pour laisser flexbox décider de l'espacement */
}}
.carousel img:hover {{
    transform: scale(1.12);
    cursor: pointer;
}}

    </style>
    """,
    unsafe_allow_html=True
)

# CSS supplémentaire pour les widgets (Radio, Slider, etc.)
widget_colors_css = """
<style>
/* ================================= */
/* BOUTONS RADIO EN JAUNE FAUVE */
/* ================================= */

/* Cercle externe du radio button (non sélectionné) - transparent avec bordure jaune légère */
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    background-color: transparent !important;
    border: 1px solid rgba(218, 165, 32, 0.5) !important;
}

/* Point intérieur quand sélectionné - rond plein jaune */
div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child > div {
    background-color: #DAA520 !important;
}

/* Survol du radio button */
div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover > div:first-child {
    border-color: #DAA520 !important;
}

/* ================================= */
/* SLIDER EN JAUNE FAUVE */
/* ================================= */

/* Piste du slider (partie remplie) */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #DAA520 !important;
}

/* Bouton/poignée du slider */
div[data-testid="stSlider"] div[role="slider"] div {
    background-color: #DAA520 !important;
    border-color: #DAA520 !important;
}

/* Piste de fond du slider */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background: linear-gradient(to right, #DAA520 0%, #DAA520 var(--value), rgba(255,255,255,0.2) var(--value), rgba(255,255,255,0.2) 100%) !important;
}

/* Valeurs affichées du slider */
div[data-testid="stSlider"] div[data-testid="stTickBar"] > div {
    color: #DAA520 !important;
}

/* ================================= */
/* SELECTBOX EN JAUNE FAUVE */
/* ================================= */

/* Bordure du selectbox au focus */
div[data-baseweb="select"] > div {
    border-color: #DAA520 !important;
}

/* Flèche du selectbox */
div[data-baseweb="select"] svg {
    color: #DAA520 !important;
}

/* Options sélectionnées dans le dropdown */
li[role="option"][aria-selected="true"] {
    background-color: rgba(218, 165, 32, 0.2) !important;
}

/* ================================= */
/* BOUTONS DE RECHERCHE */
/* ================================= */

/* Bouton principal */
button[kind="primary"] {
    background-color: #DAA520 !important;
    border-color: #DAA520 !important;
    color: black !important;
}

button[kind="primary"]:hover {
    background-color: #B8860B !important;
    border-color: #B8860B !important;
}

/* Bouton secondaire */
button[kind="secondary"] {
    border-color: #DAA520 !important;
    color: #DAA520 !important;
}

button[kind="secondary"]:hover {
    background-color: rgba(218, 165, 32, 0.1) !important;
    border-color: #B8860B !important;
    color: #B8860B !important;
}

/* ================================= */
/* TEXT INPUT EN JAUNE FAUVE */
/* ================================= */

/* Bordure au focus */
div[data-baseweb="input"] > div:focus-within {
    border-color: #DAA520 !important;
    box-shadow: 0 0 0 1px #DAA520 !important;
}

/* ================================= */
/* EXPANDER EN JAUNE FAUVE */
/* ================================= */

/* Bordure de l'expander */
div[data-testid="stExpander"] {
    border-color: #DAA520 !important;
}

/* Titre de l'expander */
div[data-testid="stExpander"] summary {
    color: #DAA520 !important;
}

/* Icône de l'expander */
div[data-testid="stExpander"] svg {
    color: #DAA520 !important;
}

/* ================================= */
/* FOND SEMI-TRANSPARENT POUR LISIBILITÉ */
/* ================================= */

/* Conteneur principal des colonnes de résultats */
div[data-testid="column"] {
    background-color: rgba(0, 0, 0, 0.7) !important;
    padding: 15px !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
}

/* Conteneur de texte dans les résultats */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    color: white !important;
}

/* Expander avec fond noir transparent */
div[data-testid="stExpander"] {
    background-color: rgba(0, 0, 0, 0.7) !important;
    border-radius: 8px !important;
    padding: 10px !important;
    margin: 10px 0 !important;
}

/* Contenu à l'intérieur des expanders */
div[data-testid="stExpander"] div[role="region"] {
    background-color: rgba(0, 0, 0, 0.5) !important;
    padding: 10px !important;
    border-radius: 5px !important;
}

</style>
"""

st.markdown(widget_colors_css, unsafe_allow_html=True)


# Injection du div bandeau fixe
st.markdown('<div class="header-div"></div>', unsafe_allow_html=True)
