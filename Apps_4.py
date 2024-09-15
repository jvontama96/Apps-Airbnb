import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from sklearn.cluster import KMeans


st.set_page_config(page_title="Airbnb Apps", layout="wide")

# specify the primary menu definition
menu_data = [
        {'icon': "far fa-chart-bar", 'label':"Booking Page", 'ttip':"Recommendation for New User"},
        {'icon': "bi bi-hand-thumbs-up", 'label':"Recommendation", 'ttip':"Recommendation for Existing User"},
        {'icon': "fas fa-tachometer-alt", 'label':"Evaluation", 'ttip':"Recommendation Result Evaluation"},
]


# Assuming df contains the listings data

df = pd.read_csv('new_airbnb.csv')

# Streamlit app
st.title('Listing Recommendation System')

# Input Country
country_input = st.selectbox(
    'Select country:',
    options=df['country'].unique()
)

# Min and Max Price Slider
price_range = st.slider('Select price range', 50, 500, (100, 300))
min_price, max_price = price_range

# Input field for the total number of guests
num_guests = st.number_input('Total number of guests', min_value=1, max_value=16, value=1)

# Slider for family suitability
family_suitability_input = st.select_slider(
    'Family Suitability Level',
    options=['Basic', 'Comfortable', 'Family-friendly'],
    value='Basic'
)

# Convert slider label to values (picking one random value from the range)
family_suitability_dict = {
    'Basic': random.choice([0, 1]),
    'Comfortable': random.choice([2, 3]),
    'Family-friendly': random.choice([4, 5])
}

family_suitability_values = family_suitability_dict[family_suitability_input]

# Slider for safety
safety_input = st.select_slider(
    'Safety Level',
    options=['Standard', 'Enhanced', 'High-Security'],
    value='Standard'
)

# Convert slider label to values
safety_dict = {
    'Standard': random.choice([0, 2]),
    'Enhanced': random.choice([3, 5]),
    'High-Security': random.choice([6, 7])
}
safety_values = safety_dict[safety_input]

# Slider for natural condition
natural_condition_input = st.select_slider(
    'Natural Condition Level',
    options=['Minimal', 'Scenic', 'Nature-Rich'],
    value='Minimal'
)

# Convert slider label to values
natural_condition_dict = {
    'Minimal': 0,
    'Scenic': random.choice([1, 2]),
    'Nature-Rich': random.choice([3])
}
natural_condition_values = natural_condition_dict[natural_condition_input]

# Slider for work suitability
work_suitability_input = st.select_slider(
    'Work Suitability Level',
    options=['Basic', 'Enhanced', 'Professional'],
    value='Basic'
)

# Convert slider label to values
work_suitability_dict = {
    'Basic': 0,
    'Enhanced': 1,
    'Professional': random.choice([2, 3])
}
work_suitability_values = work_suitability_dict[work_suitability_input]

# Filter DataFrame based on inputs
filtered_listings = df[df['country'].str.lower() == country_input]
filtered_listings = filtered_listings[(filtered_listings['price_fix'] >= min_price) & (filtered_listings['price_fix'] <= max_price)]
filtered_listings = filtered_listings[filtered_listings['guests'] >= num_guests]

# Calculate the absolute differences between the amenities preferences and the listings
filtered_listings['family_diff'] = np.abs(filtered_listings['family_suitability'] - family_suitability_values)
filtered_listings['safety_diff'] = np.abs(filtered_listings['safety'] - safety_values)
filtered_listings['natural_diff'] = np.abs(filtered_listings['natural_condition'] - natural_condition_values)
filtered_listings['work_diff'] = np.abs(filtered_listings['work_suitability'] - work_suitability_values)

# Calculate a total "distance" score from the user's preferences (sum of differences)
filtered_listings['total_diff'] = (filtered_listings['family_diff'] +
                                   filtered_listings['safety_diff'] +
                                   filtered_listings['natural_diff'] +
                                   filtered_listings['work_diff'])

# Sort by the weighted score
filtered_listings['weighted_score'] = (
    (filtered_listings['rating'] * 0.5) +  # 50% weight for rating
    (filtered_listings['reviews'] * 0.3) +  # 30% weight for reviews
    (filtered_listings['price_fix'] * 0.2)  # 20% weight for price
)
sorted_listings = filtered_listings.sort_values(by='weighted_score', ascending=False)

# Select the top 10 listings
top_listings = sorted_listings[['id','name', 'guests', 'reviews', 'rating', 'price_fix', 'country','family_suitability','safety','natural_condition','work_suitability']].head(10)

# Initialize user_data in session state if it doesn't exist
if 'user_data' not in st.session_state:
    st.session_state['user_data'] = pd.DataFrame(columns=top_listings.columns)

# Create columns with headers at the top
col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 2, 2, 2, 1])

# Add headers for the columns
with col1:
    st.write("**Name**")
with col2:
    st.write("**Guests**")
with col3:
    st.write("**Reviews**")
with col4:
    st.write("**Rating**")
with col5:
    st.write("**Price**")
with col6:
    st.write("**Country**")

# Placeholder for booking success messages
booking_success_placeholder = st.empty()

# Display the listings in a table format with "Book" button
for index, row in top_listings.iterrows():
    col1, col2, col3, col4, col5, col6, col7 = st.columns([3, 2, 2, 2, 2, 2, 2])
    
    with col1:
        st.write(row['name'])
    
    with col2:
        st.write(row['guests'])
    
    with col3:
        st.write(row['reviews'])
    
    with col4:
        st.write(row['rating'])
    
    with col5:
        st.write(row['price_fix'])
    
    with col6:
        st.write(row['country'])
    
    with col7:
        if st.button(f'Book', key=f'book_{index}'):
            st.session_state['user_data'] = pd.concat([st.session_state['user_data'], row.to_frame().T], ignore_index=True)
            booking_success_placeholder.success(f"Booking successful for {row['name']}!")
    
# Show the user_data after booking
if not st.session_state['user_data'].empty:
    st.write("Your Booked Listings:")
    # Create a subset DataFrame for display purposes
    display_columns = ['name','guests','reviews', 'rating', 'price_fix', 'country']
    display_data = st.session_state['user_data'][display_columns]
    
    # Display the subset DataFrame
    st.dataframe(display_data)
    


st.title("Recommendation System Based on Clustering")

# Load or define your data here
df = pd.read_csv('new_airbnb.csv')

# Drop columns not needed for clustering
new_df = df.drop(columns=['id', 'name', 'host_id', 'country', 'studios', 'checkin', 'checkout', 'toilets', 'price_fix', 'rating', 'reviews'])

common_columns = ['bathrooms', 'beds', 'guests', 'bedrooms',
                  'family_suitability', 'safety',
                  'natural_condition', 'work_suitability']

# Perform clustering
kmeans = KMeans(n_clusters=5, random_state=100)
new_df['cluster'] = kmeans.fit_predict(new_df[common_columns])

# Merge new_df with df based on common features
merged_df = pd.merge(df, new_df, on=common_columns, how='inner')

# Function to find the user's cluster based on their data
def find_user_cluster(user_data, new_df):
    # Match user_data with the clustered dataframe (new_df) based on common columns
    matched_row = new_df[(new_df['family_suitability'] == user_data['family_suitability']) &
                         (new_df['safety'] == user_data['safety']) &
                         (new_df['natural_condition'] == user_data['natural_condition']) &
                         (new_df['work_suitability'] == user_data['work_suitability'])]
    
    if not matched_row.empty:
        return matched_row['cluster'].values[0]
    else:
        return None

# Function to recommend listings based on user's cluster and price
def recommend_based_on_cluster(user_data, merged_df, country_input):
    country_input = country_input.lower().strip()

    # Filter merged_df based on the input country
    country_filtered_df = merged_df[merged_df['country'].str.lower().str.strip() == country_input]

    if country_filtered_df.empty:
        st.write(f"No listings found for country: {country_input}")
        return pd.DataFrame()

    # Find the user's cluster from the new_df
    user_cluster = find_user_cluster(user_data, new_df)
    if user_cluster is None:
        st.write("No matching cluster found for the user's data.")
        return pd.DataFrame()

    user_price = user_data['price_fix']

    # Step 7: Filter based on similar clusters
    similar_listings = country_filtered_df[
        country_filtered_df['cluster'] == user_cluster
    ]

    # Exclude the user's own listing to avoid duplicate recommendation
    user_listing_id = user_data['id']
    unique_listings = similar_listings[similar_listings['id'] != user_listing_id]

    if unique_listings.empty:
        st.write("No unique listings found with the exact features.")
        return pd.DataFrame()

    # Calculate price difference between user_data and listings
    unique_listings['price_diff'] = abs(unique_listings['price_fix'] - user_price)

    # Sort listings by price difference and reviews
    sorted_listings = unique_listings.sort_values(by=['price_diff', 'reviews'], ascending=[True, False])

    # Select the top 10 unique listings
    top_recommendations = sorted_listings[['name', 'rating', 'reviews', 'price_fix', 'country', 'cluster']].drop_duplicates().head(10)
   
    return top_recommendations

# Use the user_data from the session state
if 'user_data' in st.session_state and not st.session_state['user_data'].empty:
    user_data = st.session_state['user_data'].iloc[0]
    country_input = st.selectbox('Select country:', options=df['country'].unique(), key='recommend_country_selectbox')
    country_input = country_input.lower().strip()

    recommendations = recommend_based_on_cluster(user_data, merged_df, country_input)

    if not recommendations.empty:
        st.write("Recommended Listings based on your booking:")
        st.dataframe(recommendations)
    else:
        st.write("No recommendations available based on your booking.")
else:
    st.write("No user data available for recommendations.")

# Evaluation Section
st.header("Cluster Evaluation")


# Function to calculate cluster percentage
def calculate_cluster_percentage(df, cluster_col='cluster'):
    cluster_counts = df[cluster_col].value_counts(normalize=True) * 100
    cluster_percentage = cluster_counts.reset_index()
    cluster_percentage.columns = [cluster_col, 'Percentage']
    return cluster_percentage

# Function to get user data cluster percentage
def get_user_data_cluster_percentage(user_data, new_df):
    user_cluster = find_user_cluster(user_data, new_df)
    if user_cluster is None:
        return pd.DataFrame()
    
    # Filter new_df based on user's cluster
    user_data_cluster_df = new_df[new_df['cluster'] == user_cluster]
    
    return calculate_cluster_percentage(user_data_cluster_df)

# Function to get recommendations cluster percentage
def get_recommendations_cluster_percentage(recommendations):
    return calculate_cluster_percentage(recommendations)

# Function to find the user's cluster based on their data
def find_user_cluster(user_data, new_df):
    # Match user_data with the clustered dataframe (new_df) based on common columns
    matched_row = new_df[(new_df['family_suitability'] == user_data['family_suitability']) &
                         (new_df['safety'] == user_data['safety']) &
                         (new_df['natural_condition'] == user_data['natural_condition']) &
                         (new_df['work_suitability'] == user_data['work_suitability'])]
    
    if not matched_row.empty:
        return matched_row['cluster'].values[0]
    else:
        return None

# Display user data and recommendations cluster percentages if available
if 'user_data' in st.session_state and not st.session_state['user_data'].empty:
    user_data = st.session_state['user_data'].iloc[0]
    
    user_cluster_percentage = get_user_data_cluster_percentage(user_data, new_df)
    st.write("Cluster Percentage in User Data:")
    if not user_cluster_percentage.empty:
        st.dataframe(user_cluster_percentage)
        st.bar_chart(user_cluster_percentage.set_index('cluster')['Percentage'])
    else:
        st.write("No user data available for cluster evaluation.")

if 'recommendations' in st.session_state and not st.session_state['recommendations'].empty:
    recommendations = st.session_state['recommendations']
    
    recommendations_cluster_percentage = get_recommendations_cluster_percentage(recommendations)
    st.write("Cluster Percentage in Recommendations:")
    if not recommendations_cluster_percentage.empty:
        st.dataframe(recommendations_cluster_percentage)
        st.bar_chart(recommendations_cluster_percentage.set_index('cluster')['Percentage'])
    else:
        st.write("No recommendations available for cluster evaluation.")