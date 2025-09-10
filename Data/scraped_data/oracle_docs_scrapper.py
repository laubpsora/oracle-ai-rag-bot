import requests
from bs4 import BeautifulSoup
import time
import json
import csv
from urllib.parse import urlparse, urljoin, urlunparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from collections import deque, defaultdict
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RecursiveOracleDocsScraper:
    def __init__(self, delay=1, max_workers=3, max_depth=2, max_pages=200):
        """
        Initialize the recursive scraper.
        
        Args:
            delay (int): Delay between requests in seconds
            max_workers (int): Maximum number of concurrent threads
            max_depth (int): Maximum recursion depth
            max_pages (int): Maximum total pages to scrape
        """
        self.delay = delay
        self.max_workers = max_workers
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Track visited URLs and their depth
        self.visited_urls = set()
        self.url_depth = {}
        self.scraped_data = []
        self.failed_urls = []
        
        # Define allowed domains for Oracle docs
        self.allowed_domains = {
            'docs.oracle.com',
            'www.oracle.com'
        }
        
        # Define patterns for Oracle documentation URLs
        self.oracle_doc_patterns = [
            r'/en-us/iaas/Content/',
            r'/en-us/iaas/api/',
            r'/en-us/iaas/tools/',
            r'/artificial-intelligence/',
            r'/generative-ai/',
            r'/cloud-infrastructure/'
        ]
        
    def is_valid_oracle_url(self, url):
        """Check if URL is a valid Oracle documentation URL."""
        try:
            parsed = urlparse(url)
            
            # Check domain
            if parsed.netloc not in self.allowed_domains:
                return False
                
            # Check if it matches Oracle doc patterns
            path = parsed.path
            for pattern in self.oracle_doc_patterns:
                if re.search(pattern, path):
                    return True
                    
            # Additional check for main Oracle AI pages
            if 'artificial-intelligence' in path or 'generative-ai' in path:
                return True
                
            return False
        except:
            return False
    
    def normalize_url(self, url):
        """Normalize URL by removing fragments and query parameters."""
        try:
            parsed = urlparse(url)
            # Remove fragment and query for consistency
            normalized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
            return normalized
        except:
            return url
    
    def extract_links(self, soup, base_url):
        """Extract all relevant links from a BeautifulSoup object."""
        links = set()
        
        # Find all links (a tags with href, and area tags with href)
        for tag in soup.find_all(['a', 'area'], href=True):
            href = tag['href']
            full_url = urljoin(base_url, href)
            normalized_url = self.normalize_url(full_url)
            
            if self.is_valid_oracle_url(normalized_url):
                links.add(normalized_url)
        
        # Find xref links (Oracle docs often use xref attributes)
        for tag in soup.find_all(attrs={'xref': True}):
            xref = tag['xref']
            full_url = urljoin(base_url, xref)
            normalized_url = self.normalize_url(full_url)
            
            if self.is_valid_oracle_url(normalized_url):
                links.add(normalized_url)
        
        # Find data-href attributes
        for tag in soup.find_all(attrs={'data-href': True}):
            data_href = tag['data-href']
            full_url = urljoin(base_url, data_href)
            normalized_url = self.normalize_url(full_url)
            
            if self.is_valid_oracle_url(normalized_url):
                links.add(normalized_url)
        
        return links
    
    def clean_text(self, text):
        """Clean and normalize extracted text."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove common navigation elements
        text = re.sub(r'Skip to Main Content|Skip to content|Menu|Navigation|Breadcrumb|Cookie Preferences|Privacy Policy', '', text, flags=re.IGNORECASE)
        return text
    
    def extract_text_and_links(self, url, depth=0):
        """Extract text content and links from a single URL."""
        try:
            if url in self.visited_urls:
                return None, set()
                
            if len(self.visited_urls) >= self.max_pages:
                logger.info(f"Reached maximum pages limit ({self.max_pages})")
                return None, set()
            
            logger.info(f"Scraping (depth {depth}): {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract links before removing navigation elements
            links = self.extract_links(soup, url)
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract title
            title = soup.title.string if soup.title else "No title"
            title = self.clean_text(title)
            
            # Try to find main content area first
            main_content = (soup.find('main') or 
                          soup.find('article') or 
                          soup.find('div', {'class': re.compile(r'content|main|body', re.I)}) or
                          soup.find('div', {'id': re.compile(r'content|main|body', re.I)}))
            
            if main_content:
                text_content = main_content.get_text()
            else:
                # Fallback to body content
                body = soup.find('body')
                text_content = body.get_text() if body else soup.get_text()
            
            # Clean the extracted text
            text_content = self.clean_text(text_content)
            
            # Mark as visited
            self.visited_urls.add(url)
            self.url_depth[url] = depth
            
            # Add delay to be respectful to the server
            time.sleep(self.delay)
            
            result = {
                'url': url,
                'title': title,
                'text': text_content,
                'word_count': len(text_content.split()),
                'depth': depth,
                'links_found': len(links),
                'status': 'success',
                'error': None
            }
            
            return result, links
            
        except requests.RequestException as e:
            logger.error(f"Request error for {url}: {str(e)}")
            error_result = {
                'url': url,
                'title': None,
                'text': None,
                'word_count': 0,
                'depth': depth,
                'links_found': 0,
                'status': 'error',
                'error': str(e)
            }
            return error_result, set()
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {str(e)}")
            error_result = {
                'url': url,
                'title': None,
                'text': None,
                'word_count': 0,
                'depth': depth,
                'links_found': 0,
                'status': 'error',
                'error': str(e)
            }
            return error_result, set()
    
    def scrape_recursive(self, start_urls):
        """
        Scrape URLs recursively, following internal links.
        
        Args:
            start_urls (list): List of starting URLs
            
        Returns:
            list: List of dictionaries containing scraped data
        """
        # Initialize queue with starting URLs
        url_queue = deque()
        for url in start_urls:
            normalized_url = self.normalize_url(url)
            if normalized_url not in self.visited_urls:
                url_queue.append((normalized_url, 0))
                self.url_depth[normalized_url] = 0
        
        # Track links by depth for statistics
        links_by_depth = defaultdict(set)
        
        while url_queue and len(self.visited_urls) < self.max_pages:
            # Process current batch
            current_batch = []
            batch_size = min(self.max_workers, len(url_queue))
            
            for _ in range(batch_size):
                if url_queue:
                    url, depth = url_queue.popleft()
                    if url not in self.visited_urls and depth <= self.max_depth:
                        current_batch.append((url, depth))
            
            if not current_batch:
                break
            
            # Process batch with threading
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_url = {
                    executor.submit(self.extract_text_and_links, url, depth): (url, depth)
                    for url, depth in current_batch
                }
                
                for future in as_completed(future_to_url):
                    url, depth = future_to_url[future]
                    try:
                        result, links = future.result()
                        
                        if result:
                            self.scraped_data.append(result)
                            
                            if result['status'] == 'error':
                                self.failed_urls.append(result)
                            else:
                                # Add new links to queue for next depth level
                                if depth < self.max_depth:
                                    for link in links:
                                        if (link not in self.visited_urls and 
                                            link not in [item[0] for item in url_queue]):
                                            url_queue.append((link, depth + 1))
                                            links_by_depth[depth + 1].add(link)
                                
                    except Exception as e:
                        logger.error(f"Error processing {url}: {str(e)}")
            
            logger.info(f"Processed batch. Total scraped: {len(self.visited_urls)}, Queue size: {len(url_queue)}")

        self.print_statistics(links_by_depth)
        
        return self.scraped_data
    
    def print_statistics(self, links_by_depth):
        """Print scraping statistics."""
        print(f"\n{'='*60}")
        print("SCRAPING STATISTICS")
        print(f"{'='*60}")
        
        successful = sum(1 for r in self.scraped_data if r['status'] == 'success')
        failed = len(self.scraped_data) - successful
        total_words = sum(r['word_count'] for r in self.scraped_data if r['status'] == 'success')
        
        print(f"Total URLs processed: {len(self.scraped_data)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total words extracted: {total_words:,}")
        
        print(f"\nPages by depth:")
        depth_stats = defaultdict(int)
        for result in self.scraped_data:
            if result['status'] == 'success':
                depth_stats[result['depth']] += 1
        
        for depth in sorted(depth_stats.keys()):
            print(f"  Depth {depth}: {depth_stats[depth]} pages")
        
        domain_stats = defaultdict(int)
        for result in self.scraped_data:
            if result['status'] == 'success':
                domain = urlparse(result['url']).netloc
                domain_stats[domain] += 1
        
        print(f"\nPages by domain:")
        for domain, count in sorted(domain_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {domain}: {count} pages")
    
    def save_to_json(self, filename='oracle_docs_recursive.json'):
        """Save scraped data to JSON file."""
        try:
            # Create comprehensive data structure
            output_data = {
                'metadata': {
                    'total_pages': len(self.scraped_data),
                    'successful_pages': sum(1 for r in self.scraped_data if r['status'] == 'success'),
                    'failed_pages': sum(1 for r in self.scraped_data if r['status'] == 'error'),
                    'total_words': sum(r['word_count'] for r in self.scraped_data if r['status'] == 'success'),
                    'max_depth': self.max_depth,
                    'scraping_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'pages': self.scraped_data
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {str(e)}")
    
    def save_to_csv(self, filename='oracle_docs_recursive.csv'):
        """Save scraped data to CSV file."""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['url', 'title', 'text', 'word_count', 'depth', 'links_found', 'status', 'error']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.scraped_data)
            logger.info(f"Data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
    
    def save_text_files(self, folder='scraped_texts_recursive'):
        """Save each page's text content to individual text files."""
        try:
            os.makedirs(folder, exist_ok=True)
            
            for i, item in enumerate(self.scraped_data):
                if item['status'] == 'success' and item['text']:
                    parsed_url = urlparse(item['url'])
                    filename = f"{i+1:03d}_depth{item['depth']}_{parsed_url.path.split('/')[-1].replace('.htm', '')}.txt"
                    filename = re.sub(r'[^\w\-_.]', '_', filename)
                    
                    filepath = os.path.join(folder, filename)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"URL: {item['url']}\n")
                        f.write(f"Title: {item['title']}\n")
                        f.write(f"Word Count: {item['word_count']}\n")
                        f.write(f"Depth: {item['depth']}\n")
                        f.write(f"Links Found: {item['links_found']}\n")
                        f.write("-" * 50 + "\n\n")
                        f.write(item['text'])
                    
                    logger.info(f"Saved text to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving text files: {str(e)}")
    
    def save_failed_urls(self, filename='failed_urls.txt'):
        """Save failed URLs to a text file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Failed URLs and Errors:\n")
                f.write("=" * 50 + "\n\n")
                for item in self.failed_urls:
                    f.write(f"URL: {item['url']}\n")
                    f.write(f"Error: {item['error']}\n")
                    f.write(f"Depth: {item['depth']}\n")
                    f.write("-" * 30 + "\n\n")
            logger.info(f"Failed URLs saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving failed URLs: {str(e)}")


def main():
    """Main function to run the recursive scraper."""
    
    # Starting Oracle documentation URLs
    start_urls = [
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/overview.htm#overview",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/overview.htm#use-cases",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/concepts.htm#concepts",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/getting-started.htm#get-started",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/overview.htm#regions",
        "https://www.oracle.com/artificial-intelligence/generative-ai/generative-ai-service/",
        "https://www.oracle.com/artificial-intelligence/generative-ai/generative-ai-service/features/",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/use-playground.htm#use-playground",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/use-playground-embed.htm#playground-embed",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/use-playground-chat.htm#chat",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm#pretrained-models",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/deprecating.htm#retired-models",
        "https://docs.oracle.com/en-us/iaas/Content/GSG/Tasks/contactingsupport.htm",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/llama-index.htm#llama-index",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/langchain.htm#langchain",
        "https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai-inference/20231130/",
        "https://docs.oracle.com/en-us/iaas/tools/oci-cli/3.62.1/oci_cli_docs/cmdref/generative-ai-inference.html",
        "https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai/20231130/",
        "https://docs.oracle.com/en-us/iaas/tools/oci-cli/3.62.1/oci_cli_docs/cmdref/generative-ai.html",
        "https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdks.htm",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/create-ai-cluster-fine-tuning.htm#create-ai-cluster-fine-tuning",
        "https://docs.oracle.com/en-us/iaas/Content/API/Concepts/cloudshellintro.htm",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/create-new-model.htm#create-new-model",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/performance.htm#performance",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/create-ai-cluster-hosting.htm#create-ai-cluster-hosting",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/create-endpoint.htm#create-endpoint",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/integrate-models.htm#integrate",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/calculate-cost.htm#calculate-cost",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/metric-details.htm#metric-details",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/get-cluster-metrics.htm#get-cluster-metrics",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/get-endpoint-metrics.htm#get-endpoint-metrics",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/create-query.htm#create-query",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/cohere-models.htm",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/meta-models.htm",
        "https://docs.oracle.com/en-us/iaas/Content/generative-ai/xai-models.htm"
    ]
    
    # Initialize recursive scraper
    scraper = RecursiveOracleDocsScraper(
        delay=1,           
        max_workers=3,     
        max_depth=2,
        max_pages=200
    )
    
    print(f"Starting recursive scraping with {len(start_urls)} seed URLs...")
    print(f"Max depth: {scraper.max_depth}")
    print(f"Max pages: {scraper.max_pages}")
    print(f"Max workers: {scraper.max_workers}")
    
    # Start recursive scraping
    results = scraper.scrape_recursive(start_urls)
    
    
    scraper.save_to_json()
    scraper.save_to_csv()                    
    scraper.save_detailed_csv()              
    scraper.save_failed_csv()                
    scraper.save_text_files()
    scraper.save_failed_urls()
    
    print(f"\nAll files saved successfully!")
    print(f"CSV files created:")
    print(f"  - oracle_docs_recursive.csv (basic format)")
    print(f"  - oracle_docs_detailed.csv (enhanced metadata)")
    print(f"  - oracle_docs_failed.csv (failed URLs)")

if __name__ == "__main__":
    main()
