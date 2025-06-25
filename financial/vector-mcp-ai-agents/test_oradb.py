import argparse
import json
from OraDBVectorStore import OraDBVectorStore
import time
import sys
import yaml
from pathlib import Path

def check_credentials():
    """Check if Oracle DB credentials are configured in config.yaml"""
    try:
        config_path = Path("config.yaml")
        if not config_path.exists():
            print("✗ config.yaml not found.")
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config:
            print("✗ config.yaml is empty or invalid YAML.")
            return False
            
        # Check for Oracle DB credentials
        if not config.get("ORACLE_DB_USERNAME"):
            print("✗ ORACLE_DB_USERNAME not found in config.yaml")
            return False
            
        if not config.get("ORACLE_DB_PASSWORD"):
            print("✗ ORACLE_DB_PASSWORD not found in config.yaml")
            return False
            
        if not config.get("ORACLE_DB_DSN"):
            print("✗ ORACLE_DB_DSN not found in config.yaml")
            return False
            
        print("✓ Oracle DB credentials found in config.yaml")
        return True
    except Exception as e:
        print(f"✗ Error checking credentials: {str(e)}")
        return False

def test_connection():
    """Test connection to Oracle DB"""
    print("Testing Oracle DB connection...")
    try:
        store = OraDBVectorStore()
        print("✓ Connection successful!")
        return store
    except Exception as e:
        print(f"✗ Connection failed: {str(e)}")
        return None

def check_collection_stats(store):
    """Check statistics for each collection including total chunks and latest insertion"""
    if not store:
        print("Skipping collection stats check as connection failed")
        return
    
    print("\n=== Collection Statistics ===")
    
    collections = {
        "PDF Collection": "pdf_documents",
        "Repository Collection": "repository_documents",
        "Web Knowledge Base": "web_documents",
        "General Knowledge": "general_knowledge"
    }
    
    for name, collection in collections.items():
        try:
            # Get total count
            count = store.get_collection_count(collection)
            print(f"\n{name}:")
            print(f"Total chunks: {count}")
            
            # Get latest insertion if collection is not empty
            if count > 0:
                latest = store.get_latest_chunk(collection)
                print("Latest chunk:")
                print(f"  Content: {latest['content'][:150]}..." if len(latest['content']) > 150 else f"  Content: {latest['content']}")
                
                # Print metadata
                if isinstance(latest['metadata'], str):
                    try:
                        metadata = json.loads(latest['metadata'])
                    except:
                        metadata = {"source": latest['metadata']}
                else:
                    metadata = latest['metadata']
                
                source = metadata.get('source', 'Unknown')
                print(f"  Source: {source}")
                
                # Print other metadata based on collection type
                if collection == "pdf_documents" and 'page' in metadata:
                    print(f"  Page: {metadata['page']}")
                elif collection == "repository_documents" and 'file_path' in metadata:
                    print(f"  File: {metadata['file_path']}")
                elif collection == "web_documents" and 'title' in metadata:
                    print(f"  Title: {metadata['title']}")
            else:
                print("No chunks found in this collection.")
                
        except Exception as e:
            print(f"Error checking {name}: {str(e)}")

def test_add_and_query(store, query_text="machine learning"):
    """Test adding simple data and querying it"""
    if not store:
        print("Skipping add and query test as connection failed")
        return
    
    print("\nTesting add and query functionality...")
    
    # Create simple test document
    test_chunks = [
        {
            "text": "Machine learning is a field of study in artificial intelligence concerned with the development of algorithms that can learn from data.",
            "metadata": {
                "source": "Test Document",
                "page": 1
            }
        },
        {
            "text": "Deep learning is a subset of machine learning that uses neural networks with many layers.",
            "metadata": {
                "source": "Test Document",
                "page": 2
            }
        }
    ]
    
    try:
        # Test adding PDF chunks
        print("Adding test chunks to PDF collection...")
        store.add_pdf_chunks(test_chunks, document_id="test_document")
        print("✓ Successfully added test chunks")
        
        # Test querying
        print(f"\nQuerying with: '{query_text}'")
        start_time = time.time()
        results = store.query_pdf_collection(query_text)
        query_time = time.time() - start_time
        
        print(f"✓ Query completed in {query_time:.2f} seconds")
        print(f"Found {len(results)} results")
        
        # Display results
        if results:
            print("\nResults:")
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Content: {result['content']}")
                print(f"Source: {result['metadata'].get('source', 'Unknown')}")
                print(f"Page: {result['metadata'].get('page', 'Unknown')}")
        else:
            print("No results found.")
            
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        
def main():
    parser = argparse.ArgumentParser(description="Test Oracle DB Vector Store")
    parser.add_argument("--query", default="machine learning", help="Query to use for testing")
    parser.add_argument("--stats-only", action="store_true", help="Only show collection statistics without inserting test data")
    
    args = parser.parse_args()
    
    print("=== Oracle DB Vector Store Test ===\n")
    
    # Check if oracledb is installed
    try:
        import oracledb
        print("✓ oracledb package is installed")
    except ImportError:
        print("✗ oracledb package is not installed.")
        print("Please install it with: pip install oracledb")
        sys.exit(1)
    
    # Check if sentence_transformers is installed
    try:
        import sentence_transformers
        print("✓ sentence_transformers package is installed")
    except ImportError:
        print("✗ sentence_transformers package is not installed.")
        print("Please install it with: pip install sentence-transformers")
        sys.exit(1)
    
    # Check if credentials are configured
    if not check_credentials():
        print("\n✗ Oracle DB credentials not properly configured in config.yaml")
        print("Please update config.yaml with the following:")
        print("""
ORACLE_DB_USERNAME: ADMIN
ORACLE_DB_PASSWORD: your_password_here
ORACLE_DB_DSN: your_connection_string_here
        """)
        sys.exit(1)
    
    # Test connection
    store = test_connection()
    
    # Check collection statistics
    check_collection_stats(store)
    
    # If stats-only flag is not set, also test add and query functionality
    if not args.stats_only:
        # Test add and query functionality
        test_add_and_query(store, args.query)
    
    print("\n=== Test Completed ===")

if __name__ == "__main__":
    main() 