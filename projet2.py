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
import difflib
import re
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
from scipy.sparse import hstack # type: ignore

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Cinéma Conseil",
    page_icon="🎬"
)

def get_base64_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ---------------------------
# choix de la base de données
# ---------------------------

df = pd.read_csv("imdb_final.csv")
df['decade'] = (df['startYear'] // 10) * 10
# Nettoyer les titres pour la recherche
df['title_clean'] = df['originalTitle'].str.lower().str.strip()

# -------------------------------
# 1️⃣ Fonction sûre pour convertir les listes stockées en string
# -------------------------------
def safe_literal_eval(x):
    if pd.isna(x) or x == "":
        return []   # retourne une liste vide si NaN ou chaîne vide
    try:
        return ast.literal_eval(x)
    except:
        # si ast échoue (ex: tronqué avec ...), on tente nettoyage simple
        x_clean = (
            x.replace("[", "")
             .replace("]", "")
             .replace("...", "")
             .replace("'", "")
        )
        return [item.strip() for item in x_clean.split(",") if item.strip()]

# -------------------------------
# 2️⃣ Appliquer à chaque colonne
# -------------------------------
for col in ["genres", "directors", "actors"]:
    df[col] = df[col].apply(safe_literal_eval)

# -------------------------------
# 3️⃣ Exploser chaque colonne
# -------------------------------
df_genres = df.explode("genres").reset_index(drop=True)
df_directors = df.explode("directors").reset_index(drop=True)
df_actors = df.explode("actors").reset_index(drop=True)

# -------------------------------
# 4️⃣ Nettoyer les espaces
# -------------------------------
for df_tmp, col_name in zip([df_genres, df_directors, df_actors], ["genres", "directors", "actors"]):
    df_tmp[col_name] = df_tmp[col_name].str.strip()




# ---------------------------
# MODEL DE RECOMMANDATION
# ---------------------------
# ---------------------------
# NETTOYAGE DES COLONNES TEXTE (OBLIGATOIRE POUR TF-IDF)
# ---------------------------

text_columns = ['overview', 'genres', 'actors', 'directors']

for col in text_columns:
    df[col] = (
        df[col]
        .fillna("")        # remplace NaN par chaîne vide
        .astype(str) 
    )# garantit du texte)
# -----------------------------
# 2️⃣ Détecter films en français
# -----------------------------
def is_french(text):
    french_markers = [
        " le ", " la ", " les ", " une ", " un ", " des ",
        " amour ", " vie ", " homme ", " femme ", " famille "
    ]
    accents = re.search(r"[éèàçùôêîû]", text)
    return any(word in text for word in french_markers) or bool(accents)

df['is_french'] = df['overview'].apply(is_french)

# -----------------------------
# 3️⃣ Stop words français
# -----------------------------
french_stop_words = [
    "le","la","les","de","des","un","une","et","en","du","au","aux",
    "pour","sur","avec","par","dans","ce","ces","a","est","qui","que"
]

# -----------------------------
# 4️⃣ TF-IDF par catégorie
# -----------------------------
tfidf_overview = TfidfVectorizer(stop_words=french_stop_words, max_features=5000)
tfidf_genres = TfidfVectorizer()
tfidf_actors = TfidfVectorizer(max_features=3000)
tfidf_directors = TfidfVectorizer()

X_overview = tfidf_overview.fit_transform(df['overview'])
X_genres = tfidf_genres.fit_transform(df['genres'])
X_actors = tfidf_actors.fit_transform(df['actors'])
X_directors = tfidf_directors.fit_transform(df['directors'])

# -----------------------------
# 5️⃣ Pondération stricte
# -----------------------------
X = hstack([
    2 * X_overview,   # Synopsis
    3 * X_genres,     # Genres (très important)
    2 * X_actors,     # Acteurs
    1 * X_directors   # Réalisateurs
])

# -----------------------------
# 6️⃣ Modèle KNN
# -----------------------------
knn = NearestNeighbors(metric="cosine", algorithm="brute")
knn.fit(X)

# -----------------------------
# 7️⃣ Recommandation stricte + FR + populaire + tri
# -----------------------------
def recommend_movies(title, df, X, model, top_n=5):
    title = title.lower().strip()

    match = difflib.get_close_matches(
        title,
        df['title_clean'],
        n=1,
        cutoff=0.6
    )

    if not match:
        return "❌ Film introuvable dans la base."

    idx = df[df['title_clean'] == match[0]].index[0]

    film_reference = df.iloc[idx]

    distances, indices = model.kneighbors(X[idx], n_neighbors=200)

    results = df.iloc[indices[0]].copy()
    results['distance'] = distances[0]

    # Supprimer le film lui-même
    results = results[results.index != idx]

    
    # 🔝 Trier par distance croissante (plus proche en premier)
    results = results.sort_values(by='distance', ascending=True)

    return film_reference, results.head(top_n)


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
        overflow-y: hidden;
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
        link_url = f"./?movie_title={encoded_title}&goto=search"
        
        html += f'<a href="{link_url}"target="_self"><img src="{poster_info["url"]}" alt="{poster_info["title"]}"></a>'
    
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
                df['originalTitle'].str.contains(query, case=False, na=False)|
                df['frenchTitle'].str.contains(query, case=False, na=False)
            ]

            # 3. AFFICHAGE DES RÉSULTATS (Votre code d'affichage)
            if not results.empty:
                st.write(f"Résultats pour : **{query}**")

                for index, film in results.iterrows():
                    col1, col2 = st.columns([1, 4])

                    with col1:
                        poster = film['poster_path'] if pd.notna(film['poster_path']) else "placeholder.png"
                        st.image(poster, width='stretch')

                    with col2:
                        # Nettoyage
                        genre_clean = film["genres"].strip("[]").replace("'", "")
                        
                        st.markdown(f"### **{film['frenchTitle']}**")
                        st.write(f"**Genre(s) :** {genre_clean}")
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
                        film_ref,reco = recommend_movies(
                            film['primaryTitle'],
                            df,
                            X,
                            knn,
                            top_n=5
                        )

                        if reco.empty:
                            st.info("Pas de recommandations disponibles.")
                        else:
                            for idx2, film2 in reco.iterrows():
                                with st.expander(f"{film2['frenchTitle']} ({int(film2['startYear'])})"):
                                    rcol1, rcol2 = st.columns([1, 4])

                                    with rcol1:
                                        poster2 = film2['poster_path'] if pd.notna(film2['poster_path']) else "placeholder.png"
                                        st.image(poster2, width='stretch')

                                    with rcol2:
                                        st.write(f"⭐ {film2['averageRating']} — {film2['genres'].strip("[]").replace("'", "")}")
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
            .apply(lambda x: [a.strip() for a in x.strip("[]").replace("'", "").replace('"', "").split(",")])
        )
        list_actors = sorted(set(sum(all_actors, [])))
        selected_actor = st.selectbox("🎭 Choisir un acteur :", ["Aucun"] + list_actors)

        # ---- FILTRE PRODUCTEURS ----
        all_directors = (
            df["directors"].dropna()
            .apply(lambda x: [d.strip() for d in x.strip("[]").replace("'", "").replace('"', "").split(",")])
        )
        list_directors = sorted(set(sum(all_directors, [])))
        selected_director = st.selectbox("🎬 Choisir un producteur :", ["Aucun"] + list_directors)

        # ---- FILTRE GENRES ----
        all_genres = (
            df['genres'].dropna()
            .apply(lambda x: [d.strip() for d in x.strip("[]").replace("'", "").split(",")])
        )
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
                        st.image(poster, width='stretch')

                    with col2:
                        # Nettoyage
                        genre_clean = film["genres"].strip("[]").replace("'", "")
                        
                        st.markdown(f"### **{film['frenchTitle']}**")
                        st.write(f"**Genre(s) :** {genre_clean}")
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
                        film_ref,reco = recommend_movies(
                            film['originalTitle'],
                            df,
                            X,
                            knn,
                            top_n=5
                        )

                        if reco.empty:
                            st.info("Pas de recommandations disponibles.")
                        else:
                            for idx2, film2 in reco.iterrows():
                                with st.expander(f"{film2['frenchTitle']} ({int(film2['startYear'])})"):
                                    rcol1, rcol2 = st.columns([1, 4])

                                    with rcol1:
                                        poster2 = film2['poster_path'] if pd.notna(film2['poster_path']) else "placeholder.png"
                                        st.image(poster2, width='stretch')

                                    with rcol2:
                                        st.write(f"⭐ {film2['averageRating']} — {film2['genres'].strip("[]").replace("'", "")}")
                                        st.write(f"🎬 Producteur(s) : {film2['directors'].strip("[]").replace("'", "")}")
                                        st.write(f"🎭 Acteur(s) : {film2['actors'].strip("[]").replace("'", "")}")
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
                st.image(poster, width='stretch')

            with col2:
                # Nettoyage
                genre_clean = film["genres"].strip("[]").replace("'", "")
                        
                st.markdown(f"### **{film['frenchTitle']}**")
                st.write(f"**Genre(s) :** {genre_clean}")
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
                film_ref,reco = recommend_movies(
                    film['originalTitle'],
                    df,
                    X,
                    knn,
                    top_n=5
                )

                if reco.empty:
                    st.info("Pas de recommandations disponibles.")
                else:
                    for idx2, film2 in reco.iterrows():
                        with st.expander(f"{film2['frenchTitle']} ({int(film2['startYear'])})"):
                            rcol1, rcol2 = st.columns([1, 4])

                            with rcol1:
                                poster2 = film2['poster_path'] if pd.notna(film2['poster_path']) else "placeholder.png"
                                st.image(poster2, width='stretch')

                            with rcol2:
                                st.write(f"⭐ {film2['averageRating']} — {film2['genres'].strip("[]").replace("'", "")}")
                                st.write(f"🎬 Producteur(s) : {film2['directors'].strip("[]").replace("'", "")}")
                                st.write(f"🎭 Acteur(s) : {film2['actors'].strip("[]").replace("'", "")}")
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
    /* BANDEAU FIXE EN HAUT - CORRIGÉ */
    /* --------------------------------- */
    .header-div {{
        position: fixed;
        top: 0;
        left: 250px; /* ⭐ DÉCALÉ pour laisser place à la sidebar */
        width: calc(100% - 250px); /* ⭐ Largeur ajustée */
        height: 120px;
        background-image: url("data:image/png;base64,{header_b64}");
        background-size: cover;
        background-position: center;
        z-index: 999;
    }}

    /* Quand la sidebar est fermée, le bandeau prend toute la largeur */
    .stApp[data-sidebar-state="collapsed"] .header-div {{
        left: 0;
        width: 100%;
    }}

    /* --------------------------------- */
    /* BARRE STREAMLIT (NATIVE) */
    /* --------------------------------- */
    header[data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0) !important;
        z-index: 9999 !important; /* ⭐ Plus élevé que le bandeau */
    }}

    /* --------------------------------- */
    /* SIDEBAR - TOUJOURS VISIBLE */
    /* --------------------------------- */
    section[data-testid="stSidebar"] {{
        z-index: 999999 !important; /* ⭐ Au-dessus de tout */
    }}

    /* Bouton toggle sidebar - PRIORITÉ MAXIMALE */
    button[kind="header"] {{
        z-index: 9999999 !important; /* ⭐ Au-dessus de TOUT */
    }}

    /* --------------------------------- */
    /* CONTENU PRINCIPAL */
    /* --------------------------------- */
    .block-container {{
        background: rgba(0,0,0,0) !important;
        padding-top: 130px; 
    }}

    /* --------------------------------- */
    /* CAROUSEL */
    /* --------------------------------- */
    .carousel {{
        display: flex;
        overflow-x: auto;
        gap: 16px;
        padding: 16px;
        width: 100%;
        -webkit-overflow-scrolling: touch; 
    }}

    .carousel::-webkit-scrollbar {{
        height: 0px; 
    }}

    .carousel img {{
        height: 400px;
        border-radius: 10px;
        transition: transform 0.2s;
        width: auto;
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
/* SLIDER EN JAUNE FAUVE - VERSION STABLE */
/* ================================= */

/* Conteneur principal du slider */
div[data-testid="stSlider"] {
    padding: 10px 0 !important;
}

/* Piste de fond (partie non sélectionnée) - Blanche semi-transparente */
div[data-testid="stSlider"] [data-baseweb="slider"] [data-baseweb="slider-track"] {
    background: rgba(255, 255, 255, 0.3) !important;
    height: 4px !important;
}

/* Barre de progression (partie sélectionnée entre les deux curseurs) - Jaune Fauve */
div[data-testid="stSlider"] [data-baseweb="slider"] [data-baseweb="slider-track"]:first-child {
    background: #DAA520 !important;
}

/* Poignées/curseurs - Ronds jaunes */
div[data-testid="stSlider"] [role="slider"] {
    background-color: #DAA520 !important;
    border: 3px solid #FFD700 !important;
    width: 20px !important;
    height: 20px !important;
    border-radius: 50% !important;
    cursor: grab !important;
}

/* Poignée active (pendant le drag) */
div[data-testid="stSlider"] [role="slider"]:active {
    cursor: grabbing !important;
    border-color: #DAA520 !important;
}

/* Valeurs min/max (dates extrêmes) - Blanches et toujours visibles */
div[data-testid="stSlider"] [data-baseweb="slider"] > div:last-child {
    color: white !important;
    font-size: 0.9rem !important;
}

/* Labels des valeurs sélectionnées */
div[data-testid="stSlider"] [data-baseweb="tooltip"] span {
    color: #ffffff !important;
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

/* Conteneur principal des colonnes de résultats (SEULEMENT dans le contenu principal) */
.main div[data-testid="column"] {
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

sidebar_hover_css = """
<style>
/* ================================= */
/* SIDEBAR ULTRA-MINIMALISTE - FIXÉE */
/* ================================= */

/* Sidebar réduite par défaut : 60px */
section[data-testid="stSidebar"] {
    width: 30px !important;
    min-width: 30px !important;
    max-width: 30px !important;
    transition: all 0.3s ease-in-out !important;
    overflow: hidden !important;
}

/* Sidebar étendue au survol */
section[data-testid="stSidebar"]:hover {
    width: 250px !important;
    min-width: 250px !important;
    max-width: 250px !important;
}

/* Container interne */
section[data-testid="stSidebar"] > div:first-child {
    width: 250px !important;
}

/* ================================= */
/* MENU OPTION_MENU */
/* ================================= */

/* Container nav du menu */
section[data-testid="stSidebar"] nav {
    width: 250px !important;
}

/* Items du menu quand sidebar réduite */
section[data-testid="stSidebar"]:not(:hover) .nav-link,
section[data-testid="stSidebar"]:not(:hover) .nav-link-selected {
    width: 60px !important;
    padding: 0.75rem 0 !important;
    justify-content: center !important;
    overflow: visible !important;
}

/* Items du menu au survol */
section[data-testid="stSidebar"]:hover .nav-link,
section[data-testid="stSidebar"]:hover .nav-link-selected {
    width: 100% !important;
    padding: 0.75rem 1rem !important;
    justify-content: flex-start !important;
}

/* ================================= */
/* ICÔNES - TOUJOURS VISIBLES */
/* ================================= */

/* Icônes : TOUJOURS visibles et centrées */
section[data-testid="stSidebar"] .nav-link svg,
section[data-testid="stSidebar"] .nav-link i,
section[data-testid="stSidebar"] .nav-link-selected svg,
section[data-testid="stSidebar"] .nav-link-selected i {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    font-size: 1.5rem !important;
    min-width: 24px !important;
    flex-shrink: 0 !important;
}

/* Espacement icône-texte au survol */
section[data-testid="stSidebar"]:hover .nav-link svg,
section[data-testid="stSidebar"]:hover .nav-link i {
    margin-right: 0.75rem !important;
}

/* ================================= */
/* TEXTE - CACHÉ PUIS VISIBLE */
/* ================================= */

/* Texte caché quand sidebar réduite */
section[data-testid="stSidebar"]:not(:hover) .nav-link span,
section[data-testid="stSidebar"]:not(:hover) .nav-link-selected span {
    display: none !important;
}

/* Texte visible au survol */
section[data-testid="stSidebar"]:hover .nav-link span,
section[data-testid="stSidebar"]:hover .nav-link-selected span {
    display: inline-block !important;
    opacity: 1 !important;
    white-space: nowrap !important;
}

/* ================================= */
/* AJUSTEMENT CONTENU PRINCIPAL */
/* ================================= */

.main {
    margin-left: 60px !important;
    transition: margin-left 0.3s ease-in-out !important;
}

section[data-testid="stSidebar"]:hover ~ .main {
    margin-left: 250px !important;
}

/* ================================= */
/* AJUSTEMENT BANDEAU */
/* ================================= */

.header-div {
    left: 60px !important;
    width: calc(100% - 60px) !important;
    transition: all 0.3s ease-in-out !important;
}

body:has(section[data-testid="stSidebar"]:hover) .header-div {
    left: 250px !important;
    width: calc(100% - 250px) !important;
}

/* ================================= */
/* SCROLLBAR */
/* ================================= */

section[data-testid="stSidebar"]::-webkit-scrollbar {
    width: 0px !important;
}

section[data-testid="stSidebar"]:hover::-webkit-scrollbar {
    width: 6px !important;
}

section[data-testid="stSidebar"]:hover::-webkit-scrollbar-thumb {
    background: #DAA520 !important;
    border-radius: 10px !important;
}

</style>
"""

st.markdown(sidebar_hover_css, unsafe_allow_html=True)

# Injection du div bandeau fixe
st.markdown('<div class="header-div"></div>', unsafe_allow_html=True)
