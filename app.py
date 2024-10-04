import streamlit as st
import pandas as pd
import faiss
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import numpy as np

#Load data
@st.cache_data
def load_data():
    games = pd.read_csv('steam.csv')
    games = games[['App ID', 'Title', 'Reviews Total', 'Reviews Score Fancy', 'Release Date', 'Launch Price', 'Tags', 'Modified Tags', 'Steam Page']]
    games.dropna(inplace=True)
    games['Tags'] = games['Modified Tags'] + ' ' + games['Tags']
    games['Tags'] = games['Tags'].astype(str)
    new_df = games[['Title', 'Reviews Score Fancy', 'Tags', 'Steam Page']]
    
    # Stemming function
    ps = PorterStemmer()
    new_df['Tags'] = new_df['Tags'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
    
    return new_df

new_df = load_data()

def load_css():
    with open("static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

#Vectorize tags and initialize Faiss index
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
            D, I = index.search(query_vector, num_recommendations + 1)  # Top k recommendations excluding itself
            
            # Remove the first element (the game itself)
            all_distances.extend(D[0][1:])
            all_indices.extend(I[0][1:])
        else:
            st.warning(f"Game '{name}' not found in the dataset.")

    if not all_distances or not all_indices:
        return []

    # Convert lists to arrays
    all_distances = np.array(all_distances)
    all_indices = np.array(all_indices)

    # Get unique indices and sort by distance
    unique_indices = np.unique(all_indices)
    
    # Sort distances and indices
    sorted_indices = np.argsort(all_distances)
    sorted_unique_indices = unique_indices[np.in1d(unique_indices, all_indices[sorted_indices])]

    # Handle case where fewer recommendations are available than requested
    num_recommendations = min(num_recommendations, len(sorted_unique_indices))

    # Return the top recommendations
    top_indices = sorted_unique_indices[:num_recommendations]
    recommended_games = new_df.iloc[top_indices][['Title', 'Reviews Score Fancy', 'Steam Page']].values.tolist()

    return recommended_games

# Streamlit UI
st.title('🎮 Game Recommendation System')

st.header("Find Games Similar to Your Favorites! 🎯")
selected_games = st.multiselect("Choose one or more games from the list", new_df['Title'].unique())

num_recommendations = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

if st.button('Get Recommendations'):
    if selected_games:
        recommendations = recommend_multiple(selected_games, num_recommendations)
        if recommendations:
            st.subheader(f"Games similar to {', '.join(selected_games)}:")
            for game in recommendations:
                title, rating, steam_page = game
                st.markdown(f"""
                    **Title:** [{title}]({steam_page})  
                    **Rating:** {rating}/100  
                    **Steam Page:** [Link]({steam_page})  
                """, unsafe_allow_html=True)
        else:
            st.error(f"No recommendations found for the selected games.")
    else:
        st.warning("Please select at least one game to get recommendations.")
else:
    st.info('Select one or more games and click the button to get recommendations!')

st.markdown("Built with ❤️ using Streamlit")

# #65K+ games on this dataset