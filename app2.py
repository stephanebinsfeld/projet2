import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


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


st.set_page_config(page_title="Accueil", layout="wide")



st.sidebar.title("Menu")
page = st.sidebar.radio("Navigation", ["Accueil", "Recherche de films", "Base de données"])



# ---------------------------
# ACCUEIL
# ---------------------------
if page == "Accueil":
    st.title("🎬 Bienvenue sur Cinéma Conseil")
    st.write("Utilise le menu à gauche pour naviguer entre les pages.")


# ---------------------------
# PAGE RECHERCHE
# ---------------------------
# ---------------------------
# PAGE RECHERCHE
# ---------------------------
elif page == "Recherche de films":
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
elif page == "Base de données":
    st.title("📂 Base de données des films")

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