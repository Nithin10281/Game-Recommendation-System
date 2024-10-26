# import streamlit as st
# import pandas as pd
# import faiss
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.stem.porter import PorterStemmer
# import numpy as np
# import time
# import requests
# from streamlit_lottie import st_lottie


# # Load data
# @st.cache_data
# def load_data():
#     games = pd.read_csv('steam.csv')
#     games = games[['App ID', 'Title', 'Reviews Total', 'Reviews Score Fancy', 'Release Date',
#                    'Launch Price', 'Tags', 'Modified Tags', 'Steam Page']]
#     games.dropna(inplace=True)
#     games['Tags'] = games['Modified Tags'] + ' ' + games['Tags']
#     games['Tags'] = games['Tags'].astype(str)
#     new_df = games[['Title', 'Reviews Score Fancy', 'Tags', 'Steam Page']]

#     # Stemming function
#     ps = PorterStemmer()
#     new_df['Tags'] = new_df['Tags'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))

#     return new_df

# new_df = load_data()

# def load_css():
#     # Define custom CSS for enhanced styling with a darker background
#     st.markdown("""
#         <style>
#             body {
#                 background-color: #121212;
#                 font-family: 'Arial', sans-serif;
#             }
#             .title h1 {
#                 font-size: 2.5rem;
#                 color: #ffffff;
#                 text-align: center;
#                 margin-bottom: 10px;
#             }
#             .header h2 {
#                 color: #ffffff;
#                 margin-bottom: 20px;
#                 text-align: center;
#             }
#             .stButton>button {
#                 background-color: #0052cc;
#                 color: white;
#                 border-radius: 8px;
#                 padding: 8px 15px;
#                 border: none;
#             }
#             .stButton>button:hover {
#                 background-color: #0040a0;
#                 color: white;
#             }
#             .footer {
#                 margin-top: 50px;
#                 text-align: center;
#                 color: #aaaaaa;
#             }
#         </style>
#     """, unsafe_allow_html=True)

# load_css()

# # Vectorize tags and initialize Faiss index
# @st.cache_resource
# def setup_faiss_index():
#     cv = CountVectorizer(max_features=8000, stop_words='english')
#     vectors = cv.fit_transform(new_df['Tags']).toarray().astype('float32')
#     index = faiss.IndexFlatL2(vectors.shape[1])
#     index.add(vectors)
#     return vectors, index

# vectors, index = setup_faiss_index()

# # Recommendation function for multiple game inputs
# def recommend_multiple(selected_games, num_recommendations=10):
#     all_distances = []
#     all_indices = []

#     for name in selected_games:
#         if name in new_df['Title'].values:
#             game_index = new_df[new_df['Title'] == name].index[0]
#             query_vector = vectors[game_index].reshape(1, -1)
#             D, I = index.search(query_vector, num_recommendations + 1)  # Top k recommendations including itself

#             # Remove the first element (the game itself)
#             D = D[0][1:]
#             I = I[0][1:]

#             all_distances.extend(D)
#             all_indices.extend(I)
#         else:
#             st.warning(f"Game '{name}' not found in the dataset.")

#     if not all_distances or not all_indices:
#         return pd.DataFrame()

#     # Convert lists to arrays
#     all_distances = np.array(all_distances)
#     all_indices = np.array(all_indices)

#     # Get unique indices and sort by distance
#     unique_indices, unique_positions = np.unique(all_indices, return_index=True)
#     unique_distances = all_distances[unique_positions]

#     # Sort the unique indices by distance
#     sorted_indices = np.argsort(unique_distances)
#     sorted_unique_indices = unique_indices[sorted_indices]

#     # Handle case where fewer recommendations are available than requested
#     num_recommendations = min(num_recommendations, len(sorted_unique_indices))

#     # Return the top recommendations
#     top_indices = sorted_unique_indices[:num_recommendations]
#     recommended_games = new_df.iloc[top_indices][['Title', 'Reviews Score Fancy', 'Steam Page']].reset_index(drop=True)

#     return recommended_games

# # Streamlit UI
# st.markdown("<div class='title'><h1>🎮 Game Recommendation System</h1></div>", unsafe_allow_html=True)

# st.markdown("<div class='header'><h2>Find Games Similar to Your Favorites! 🎯</h2></div>", unsafe_allow_html=True)

# selected_games = st.multiselect(
#     "Choose one or more games from the list",
#     new_df['Title'].unique(),
#     help="Select your favorite games for similar recommendations"
# )

# num_recommendations = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

# if st.button('Get Recommendations'):
#     if selected_games:
#         # Simulate loading
#         with st.spinner('Generating recommendations...'):
#             time.sleep(1.5)  # Simulate some processing time
#             progress = st.progress(0)  # Initialize progress bar

#             # Update progress bar over time
#             for percent in range(1, 101, 10):
#                 time.sleep(0.1)  # Simulate progress
#                 progress.progress(percent)

#             recommendations = recommend_multiple(selected_games, num_recommendations)
#             if not recommendations.empty:
#                 st.subheader(f"Games similar to {', '.join(selected_games)}:")
#                 for idx, row in recommendations.iterrows():
#                     title = row['Title']
#                     rating = row['Reviews Score Fancy']
#                     steam_page = row['Steam Page']
#                     number = idx + 1

#                     # Use expander to show and hide details
#                     with st.expander(f"{number} : {title} (Click to show details)"):
#                         st.markdown(f"""
#                         **Title**: {title}  
#                         **Rating**: {rating}/100  
#                         **Steam Page**: [Visit here]({steam_page})  
#                         """)
#             else:
#                 st.error(f"No recommendations found for the selected games.")
#     else:
#         st.warning("Please select at least one game to get recommendations.")
# else:
#     st.info('Select one or more games and click the button to get recommendations!')

# st.markdown("<div class='footer'>Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)

























import streamlit as st
import pandas as pd
import faiss
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import time
import requests
from streamlit_lottie import st_lottie

# Load data
@st.cache_data
def load_data():
    games = pd.read_csv('steam.csv')
    games = games[['App ID', 'Title', 'Reviews Total', 'Reviews Score Fancy', 'Release Date',
                   'Launch Price', 'Tags', 'Modified Tags', 'Steam Page']]
    games.dropna(inplace=True)
    games['Tags'] = games['Modified Tags'] + ' ' + games['Tags']
    games['Tags'] = games['Tags'].astype(str)
    new_df = games[['Title', 'Reviews Score Fancy', 'Tags', 'Steam Page']]

    # Stemming function
    ps = PorterStemmer()
    new_df['Tags'] = new_df['Tags'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))

    return new_df

new_df = load_data()

# Fetch and cache Lottie animation
@st.cache_data
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            st.error(f"Error fetching Lottie animation: {r.status_code}")
            return None
        return r.json()
    except Exception as e:
        st.error(f"Failed to fetch Lottie animation. Error: {e}")
        return None

# Load Lottie animation
lottie_animation = load_lottie_url("https://lottie.host/8516a3c4-f9a0-4705-9aab-7e32f865add2/bFfZhJtm0c.json")

# Vectorize tags and initialize Faiss index
@st.cache_resource
def setup_faiss_index():
    cv = CountVectorizer(max_features=8000, stop_words='english')
    vectors = cv.fit_transform(new_df['Tags']).toarray().astype('float32')
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return vectors, index

vectors, index = setup_faiss_index()

# Recommendation function for multiple game inputs
def recommend_multiple(selected_games, num_recommendations=10):
    all_distances = []
    all_indices = []

    for name in selected_games:
        if name in new_df['Title'].values:
            game_index = new_df[new_df['Title'] == name].index[0]
            query_vector = vectors[game_index].reshape(1, -1)
            D, I = index.search(query_vector, num_recommendations + 1)  # Top k recommendations including itself

            # Remove the first element (the game itself)
            D = D[0][1:]
            I = I[0][1:]

            all_distances.extend(D)
            all_indices.extend(I)
        else:
            st.warning(f"Game '{name}' not found in the dataset.")

    if not all_distances or not all_indices:
        return pd.DataFrame()

    # Convert lists to arrays
    all_distances = np.array(all_distances)
    all_indices = np.array(all_indices)

    # Get unique indices and sort by distance
    unique_indices, unique_positions = np.unique(all_indices, return_index=True)
    unique_distances = all_distances[unique_positions]

    # Sort the unique indices by distance
    sorted_indices = np.argsort(unique_distances)
    sorted_unique_indices = unique_indices[sorted_indices]

    # Handle case where fewer recommendations are available than requested
    num_recommendations = min(num_recommendations, len(sorted_unique_indices))

    # Return the top recommendations
    top_indices = sorted_unique_indices[:num_recommendations]
    recommended_games = new_df.iloc[top_indices][['Title', 'Reviews Score Fancy', 'Steam Page']].reset_index(drop=True)

    return recommended_games

# Streamlit UI
st.markdown("<div class='title'><h1>🎮 Game Recommendation System</h1></div>", unsafe_allow_html=True)

st.markdown("<div class='header'><h2>Find Games Similar to Your Favorites! 🎯</h2></div>", unsafe_allow_html=True)

selected_games = st.multiselect(
    "Choose one or more games from the list",
    new_df['Title'].unique(),
)

num_recommendations = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

if st.button('Get Recommendations'):
    if selected_games:
        if lottie_animation:
            # Show Lottie animation while loading
            loading_placeholder = st.empty()  # Create an empty container for the animation
            with loading_placeholder:
                st_lottie(lottie_animation, height=150, key="loading")
            
            # Delay to show animation
            time.sleep(5)  # Simulate animation showing time
            
            # Simulate recommendation processing
            recommendations = recommend_multiple(selected_games, num_recommendations)
            
            # Stop animation by clearing the placeholder
            loading_placeholder.empty()  # Clear the animation
            
            # Display results
            if not recommendations.empty:
                st.subheader(f"Games similar to {', '.join(selected_games)}:")
                for idx, row in recommendations.iterrows():
                    title = row['Title']
                    rating = row['Reviews Score Fancy']
                    steam_page = row['Steam Page']
                    number = idx + 1

                    # Use expander to show and hide details
                    with st.expander(f"{number} : {title} (Click to show details)"):
                        st.markdown(f"""
                        **Title**: {title}  
                        **Rating**: {rating}/100  
                        **Steam Page**: [Visit here]({steam_page})  
                        """)
            else:
                st.error(f"No recommendations found for the selected games.")
        else:
            st.warning("Lottie animation failed to load. Please try again later.")
    else:
        st.warning("Please select at least one game to get recommendations.")
else:
    st.info('Select one or more games and click the button to get recommendations!')
