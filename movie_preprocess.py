import pandas as pd
import requests

movies_df = pd.read_csv("movies.csv")
print(movies_df)

# step 1
movies_df['movieId'] = movies_df['movieId'].astype(str)
links_df = pd.read_csv('links.csv', dtype=str)
merged_df = movies_df.merge(links_df, on='movieId', how='left')
print(merged_df)
print(merged_df.columns)

#step 2conda install pandas
def add_url(row):
    return f"http://www.imdb.com/title/tt{row}"

merged_df['url'] = merged_df['imdbId'].apply(lambda x:add_url(x))

print(merged_df)
print(merged_df.columns)

# step 3
ratings_df = pd.read_csv('ratings.csv')
ratings_df['movieId'] = ratings_df['movieId'].astype(str)

agg_df = ratings_df.groupby('movieId')
print(agg_df)

agg_df = ratings_df.groupby('movieId').mean()
print(agg_df)

agg_df = ratings_df.groupby('movieId').count()
print(agg_df)

# agg_df = ratings_df.groupby('movieId').agg()
# print(agg_df)

agg_df = ratings_df.groupby('movieId').agg(rcount = ('rating', 'count'), rmean=('rating', 'mean'))
print(agg_df)

merged_df = merged_df.merge(agg_df, on='movieId')
print(merged_df)
print(merged_df.columns)



# step 4 포스터 경로 추가
import requests
from tqdm import tqdm

def add_poster(df):
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        tmdb_id = row["tmdbId"]
        tmdb_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key=f2a1fddeef038db026fb3e05415e80f20000&language=en-US"
        result = requests.get(tmdb_url)
        # final url : https://image.tmdb.org/t/p/original/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg
        try:
            df.at[i, "poster_path"] = "https://image.tmdb.org/t/p/original" + result.json()['poster_path']
        except (TypeError, KeyError) as e:
            # toy story poster as default
            df.at[i, "poster_path"] = "https://image.tmdb.org/t/p/original/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg"
    return df

merged_df['poster_path'] = None
merged_df = add_poster(merged_df)
print(merged_df)

merged_df.to_csv('movies_final.csv', index=None)