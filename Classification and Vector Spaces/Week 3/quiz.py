import numpy as np

# Vector representations of A, B
A = np.array([1, 2, 3])
B = np.array([4, 7, 2])
# Calculate Euclidean distance between A and B
norm = np.linalg.norm(A - B)
print(f'Euclidean distance between D and E: {norm}')

# Vector representations of D and E
D = np.array([1, 0, -1])
E = np.array([2, 8, 1])
# Calculate dot product of D and E
dot_product = np.dot(D, E)
# Calculate norms of D and E
norm_D = np.linalg.norm(D)
norm_E = np.linalg.norm(E)

# Calculate cosine similarity between D and E
cosine_similarityDE = dot_product / (norm_D * norm_E)
print(f'Cosine similarity between D and E: {cosine_similarityDE}')

# Vector representations of countries and their capitals
USA = np.array([5, 6])
Washington = np.array([10, 5])
Turkey = np.array([3, 1])
Ankara = np.array([9, 1])
Russia = np.array([5, 5])
Japan = np.array([4, 3])

# Calculate cosine similarity between countries and their capitals
cosine_similarity_USA = np.dot(USA, Ankara) / (np.linalg.norm(USA) * np.linalg.norm(Ankara))
cosine_similarity_Turkey = np.dot(Turkey, Ankara) / (np.linalg.norm(Turkey) * np.linalg.norm(Ankara))
cosine_similarity_Russia = np.dot(Russia, Ankara) / (np.linalg.norm(Russia) * np.linalg.norm(Ankara))
cosine_similarity_Japan = np.dot(Japan, Ankara) / (np.linalg.norm(Japan) * np.linalg.norm(Ankara))

# Save cosine similarities in a dictionary
cosine_similarities = {
    'USA': cosine_similarity_USA,
    'Turkey': cosine_similarity_Turkey,
    'Russia': cosine_similarity_Russia,
    'Japan': cosine_similarity_Japan
}

# Find the country with the highest cosine similarity with Ankara
capital = max(cosine_similarities, key=cosine_similarities.get)
print(f'Ankara is the capital of {capital}')
