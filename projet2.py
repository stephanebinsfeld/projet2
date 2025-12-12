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
import streamlit as st
from streamlit_option_menu import option_menu
import base64

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
# Paramètres sidebar et menu
# ---------------------------

st.set_page_config(layout="wide")
with st.sidebar:
    
    # Code hexadécimal pour un Jaune Fauve/Ocre :
    COULEUR_FAUVE ='#DAA520'
    COULEUR_NOIRE ='black' # Couleur au survol (hover) : noir
    selection = option_menu(
        menu_title=None,
        options = ["Accueil", "recherche de films", "Base de données"],
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


# ---------------------------
# ACCUEIL
# ---------------------------

if selection == "Accueil":
#--- Contenu principal de la page "Accueil" ---
    # Titre principal de l'application (affiché en haut de la page)
    st.title("🎬 Bienvenue sur Cinéma Conseil")
    

# 1. Nettoyer et isoler les chemins de poster valides (non NaN)
    # On utilise .tolist() pour obtenir une liste native Python des URLs.
    posters_valides = df['poster_path'].dropna().tolist()

    if posters_valides:
        # 2. Mélange aléatoire des chemins valides
        # Note : random.shuffle fonctionne en place (modifie la liste originale)
        random.shuffle(posters_valides)
        
        # 3. Garder seulement les 12 premiers (ou moins si la liste est plus courte)
        posters = posters_valides[:15]
        
    else:
        # Cas où aucune URL de poster n'est trouvée dans le CSV
        st.warning("Aucun chemin de poster valide n'a été trouvé dans le fichier imdb_final.csv.")
        posters = [] # Liste vide pour éviter une erreur

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

# ---------------------------
# PAGE RECHERCHE
# ---------------------------
elif selection == "recherche de films":
    st.title("🔍 Recherche de films")

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
        query = st.text_input("Titre du film :")

        if query:
            results = df[
                df['primaryTitle'].str.contains(query, case=False, na=False) |
                df['originalTitle'].str.contains(query, case=False, na=False)
            ]

            if not results.empty:
                st.write(f"Résultats pour : **{query}**")

                for index, film in results.iterrows():
                    col1, col2 = st.columns([1, 4])

                    with col1:
                        poster = film['poster_path'] if pd.notna(film['poster_path']) else "placeholder.png"
                        st.image(poster, use_container_width=True)

                    with col2:
                        st.markdown(f"### **{film['primaryTitle']}**")
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
                                with st.expander(f"{film2['primaryTitle']} ({int(film2['startYear'])})"):
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

            else:
                st.info("Aucun film trouvé.")

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
                        st.markdown(f"### **{film['primaryTitle']}**")
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
                                with st.expander(f"{film2['primaryTitle']} ({int(film2['startYear'])})"):
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
                st.markdown(f"## **{film['primaryTitle']}**")
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
                        with st.expander(f"{film2['primaryTitle']} ({int(film2['startYear'])})"):
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
