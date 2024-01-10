import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="https://cdn-icons-png.flaticon.com/128/777/777242.png",
                   layout="wide")
# Top section of page
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/128/408/408426.png", width=80)
with col2:
    st.title("Movie Recommendation System")

# loading CSV file
movies = pd.read_csv('bollyMovies.csv') # add your file's absolute path

# making list of movies
movieList = [""] + movies['original_title'].tolist()

selected = st.selectbox("Select a Movie", options=movieList,)

# when no movie is selected then show the info about the application (as on homepage)
if selected == "":
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.image("https://img.freepik.com/free-vector/realistic-bollywood-cinema-sign_52683-35071.jpg?size=626&ext="
                 "jpg&ga=GA1.2.1339865653.1692191364&semt=sph")
    with col2:
        st.subheader("Welcome to Movie Recommender")
        st.markdown("""Welcome to your Ultimate Movie Companion! Say goodbye to endless scrolling on streaming platforms in search of the perfect movie. The Movie Recommendation System is your one-stop solution for personalized movie suggestions and comprehensive film information.

**Discover Your Next Favorite Film**

Explore a diverse collection of movies across genres, from thrilling action and heartwarming dramas to mind-bending sci-fi and classic comedies. Our recommendation system analyzes your preferences and trending films to suggest movies tailored just for you.

**Happy Watching! 🎬**""")

# when a movie is selected
if selected != "":
    index = movies.index.get_loc(movies[movies['original_title'] == selected].index[0]) # finding index of movie
    # displaying details of selected movie with help of movie index
    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        st.image(movies['poster_path'].iloc[index])
    with col2:
        st.header(movies['original_title'].iloc[index])

        genre = movies['genres'].iloc[index].replace(",", " | ")
        st.markdown(f"**Genre:** {genre}")

        st.markdown(f"**IMDB Rating:** ⭐ {str(movies['imdb_rating'].iloc[index])}")

        actor = movies['actors'].iloc[index].replace(",", " | ")
        st.markdown(f"**Actors:** {actor}")

        time = movies['runtime'].iloc[index]
        if time != "Data Not Available":
            time = int(time)
            st.markdown(f"**Screen Time:** ⏰ {int(time / 60)} Hours {time % 60} Minutes")
        else:
            st.markdown(f"**Screen Time:** ⏰ {time}")

        st.markdown(f"**Release Date:** 📅 {movies['release_date'].iloc[index]}")

    st.markdown("**Story**")
    st.write(movies['story'].iloc[index])
    st.divider()

    # making ML model
    vectors = TfidfVectorizer().fit_transform(movies['tags'])
    similarity = cosine_similarity(vectors)

    # function that recommends the movie
    def recommend():
        movie_list = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])[1:11] # take 10 similar movies
        st.header("Recommended Movies")
        st.write("\n")
        # Display details of recommended movie based on their respective indexes
        for i in movie_list:
            col1, col2 = st.columns([0.25, 0.75])
            with col1:
                st.image(movies.iloc[i[0]].poster_path)
            with col2:
                st.subheader(movies.iloc[i[0]].original_title)
                st.markdown(f"**IMDB Rating:** ⭐ {str(movies.iloc[i[0]].imdb_rating)}")
                time = movies.iloc[i[0]].runtime
                try:
                    time = int(time)
                    st.markdown(f"**Screen Time:** ⏰ {int(time / 60)} Hours {time % 60} Minutes")
                except ValueError:
                    st.markdown(f"**Screen Time:** ⏰ Not Available")
                genre = movies.iloc[i[0]].genres.replace(",", " | ")
                st.markdown(f"**Genres:** {genre}")
                actor = movies.iloc[i[0]].actors.replace(",", " | ")
                st.markdown(f"**Actors:** {actor}")
            st.markdown(f"**Story:** {str(movies.iloc[i[0]].story)}")
            st.divider()

    recommend()




