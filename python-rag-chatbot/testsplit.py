
from langchain_core.documents import Document

# 1. Load the csv and format as documents
detailed = pd.read_csv('pdfFiles/preprocessed.csv')
todrop = ['text_processed', 'text_no_stopwords', 'text_stemmed', 'text_lemmatized', 'status', 'scraped_timestamp']
detailed.drop(columns=todrop, axis=1, inplace=True)
detailed = detailed.dropna().astype(str)


# Convert each row into a Document object
documents = [
    Document(
        page_content=row['text'],  # this will be chunked later
        metadata={k: v for k, v in row.items() if k != 'text'}
    )
    for _, row in detailed.iterrows()
]
# 2. Split pages in chunks
document_splits = split_in_chunks(documents)
document_splits