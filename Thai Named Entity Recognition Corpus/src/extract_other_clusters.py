#!/usr/bin/env python3
"""
Extract clustered samples from Thai NER Corpus

This script reads the ThaiNER.jsonl file and extracts only samples
whose domains belong to the meaningful clustered categories.
"""

import json
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def load_data(jsonl_file):
    """Load data from JSONL file"""
    data = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
    except FileNotFoundError:
        print(f"Error: File '{jsonl_file}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    return data

def categorize_domain(domain):
    """Categorize domain into major groups (same as visualize_domains.py)"""
    domain = domain.lower()

    # Government & Administration
    if any(keyword in domain for keyword in ['government', 'gov', 'municipal', 'public_service', 'police', 'law_enforcement']):
        return 'Government & Administration'

    # Finance & Business
    if any(keyword in domain for keyword in ['finance', 'bank', 'banking', 'insurance', 'investment', 'stock', 'crypto', 'payment', 'commerce', 'business', 'corporate', 'startup', 'hr', 'human_resource', 'employment', 'procurement']):
        return 'Finance & Business'

    # Legal & Law
    if any(keyword in domain for keyword in ['legal', 'law', 'court', 'justice', 'crime', 'cybercrime', 'fraud', 'litigation']):
        return 'Legal & Law'

    # Education & Research
    if any(keyword in domain for keyword in ['education', 'school', 'university', 'academic', 'research', 'exam', 'library']):
        return 'Education & Research'

    # Healthcare & Medical
    if any(keyword in domain for keyword in ['healthcare', 'medical', 'hospital', 'clinic', 'pharmacy', 'health', 'disease', 'disaster']):
        return 'Healthcare & Medical'

    # Technology & IT
    if any(keyword in domain for keyword in ['tech', 'technology', 'it', 'software', 'cyber', 'security', 'it_support', 'it_security', 'it_devops', 'it_ops', 'cyber_security', 'data', 'ai', 'automation']):
        return 'Technology & IT'

    # Travel & Tourism
    if any(keyword in domain for keyword in ['travel', 'tourism', 'hotel', 'hospitality', 'aviation', 'transport', 'logistics', 'shipping']):
        return 'Travel & Transportation'

    # Retail & E-commerce
    if any(keyword in domain for keyword in ['retail', 'ecommerce', 'e-commerce', 'shopping', 'market', 'commerce', 'store', 'food_delivery', 'delivery']):
        return 'Retail & E-commerce'

    # Media & Entertainment
    if any(keyword in domain for keyword in ['media', 'news', 'entertainment', 'sports', 'music', 'film', 'movie', 'gaming', 'celebrity', 'social_media']):
        return 'Media & Entertainment'

    # Energy & Utilities
    if any(keyword in domain for keyword in ['energy', 'utility', 'electric', 'water', 'gas', 'construction', 'manufacturing', 'industrial']):
        return 'Energy & Infrastructure'

    # Regional & Minority Languages
    if any(keyword in domain for keyword in ['isan', 'north', 'south', 'central', 'east', 'west', 'regional', 'minority', 'ethnic', 'dialect', 'language', 'mon', 'vietnamese', 'chinese', 'malay', 'karen', 'hmong', 'akha', 'lisu', 'chong', 'laotian', 'khmer', 'burmese', 'tibetan', 'miao', 'yao', 'lua', 'thai_yai', 'tai_lue', 'tai_dam', 'phuan', 'bru', 'cham', 'mlabri', 'shan', 'yong', 'northern_khmer', 'kuy', 'nyah_kur', 'urak_lawoi', 'tai_song', 'phu_thai', 'khmer_northern', 'hakka', 'tai_nuea', 'mien', 'lawa', 'malay_satun', 'khmu', 'nyaw', 'saek', 'kaleung', 'moklen', 'hainanese', 'lao_wiang', 'karen_pwo', 'so', 'malay_bangkok', 'chinese_cantonese', 'khmer_buriram', 'palaung', 'mani', 'lua', 'thai_song_dum', 'u_h', 'phu_noi', 'phutai', 'khamu', 'moken', 'lawax', 'mjen', 'gong', 'palong', 'hakkax', 'pwo_karen', 'kaloeng', 'hany', 'tai_nyo', 'chines_yue', 'seak', 'kason', 'sochi', 'tai_ya', 'bisux', 'mal', 'jingpho', 'tai_yoy', 'mok', 'umpi', 'khuen', 'samre', 'wa', 'chong', 'khmer_northern', 'khmer_tibetan', 'malay_pattani', 'malay_stul', 'malay_bangkok', 'malay_songkhla', 'phu_thai', 'khmer_northern', 'hakka', 'tai_nuea', 'mien', 'lawa', 'malay_satun', 'khmu', 'nyaw', 'saek', 'kaleung', 'moklen', 'hainanese', 'lao_wiang', 'karen_pwo', 'so', 'malay_bangkok', 'chinese_cantonese', 'khmer_buriram', 'palaung', 'mani', 'lua', 'thai_song_dum', 'u_h', 'phu_noi', 'phutai', 'khamu', 'moken', 'lawax', 'mjen', 'gong', 'palong', 'hakkax', 'pwo_karen', 'kaloeng', 'hany', 'tai_nyo', 'chines_yue', 'seak', 'kason', 'sochi', 'tai_ya', 'bisux', 'mal', 'jingpho', 'tai_yoy', 'mok', 'umpi', 'khuen', 'samre', 'wa']):
        return 'Regional & Minority Languages'

    # Food & Lifestyle
    if any(keyword in domain for keyword in ['food', 'restaurant', 'drink', 'beverage', 'lifestyle', 'fashion', 'beauty', 'pet', 'fitness', 'health_tech', 'luxury']):
        return 'Food & Lifestyle'

    # Environment & Weather
    if any(keyword in domain for keyword in ['weather', 'environment', 'climate', 'nature', 'agriculture', 'farming', 'forestry']):
        return 'Environment & Agriculture'

    # Real Estate & Housing
    if any(keyword in domain for keyword in ['real_estate', 'realestate', 'property', 'housing', 'construction']):
        return 'Real Estate & Property'

    # Telecom & Communication
    if any(keyword in domain for keyword in ['telecom', 'communication', 'phone', 'mobile', 'internet']):
        return 'Telecom & Communication'

    # Other
    return 'Other'

def get_category_mapping_with_clustering(data):
    """Get category mapping including clustering for Other domains"""
    individual_domains = Counter([item.get('domain', 'unknown') for item in data])
    category_mapping = {}

    # First, categorize all domains
    for domain in individual_domains.keys():
        category_mapping[domain] = categorize_domain(domain)

    # Get Other domains
    other_domains = [domain for domain, cat in category_mapping.items() if cat == 'Other']

    if other_domains:
        print(f"Clustering {len(other_domains)} 'Other' domains using TF-IDF and K-Means...")
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = vectorizer.fit_transform(other_domains)

            # Determine number of clusters
            n_clusters = min(max(2, len(other_domains) // 20), 10)

            if n_clusters > 1 and len(other_domains) > n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X.toarray())

                # Group domains by cluster and assign meaningful names
                cluster_groups = {}
                cluster_names = {
                    0: 'Mixed_Content',  # automotive, culture, religion, customer_support, web, minority languages
                    1: 'Min_Nan_Chinese',  # จีนหมิ่น dialects
                    2: 'NGO',  # ngo
                    3: 'Malay_Dialects',  # มลายู dialects
                    4: 'Regional_Dialects',  # จีนกลาง, เขมร, โป dialects
                    5: 'Cantonese_Chinese'  # จีนเยฺว่(กวางตุ้ง)
                }

                for domain, label in zip(other_domains, labels):
                    cluster_name = cluster_names.get(label, f'Other_Cluster_{label + 1}')
                    cluster_groups.setdefault(cluster_name, []).append(domain)

                # Update mapping for clustered domains
                for cluster_name, domains in cluster_groups.items():
                    for domain in domains:
                        category_mapping[domain] = cluster_name
        except Exception as e:
            print(f"Warning: Clustering failed: {e}. Keeping 'Other' as is.")

    return category_mapping

def main():
    """Main function"""
    # File path
    script_dir = Path(__file__).parent
    jsonl_file = script_dir.parent / "data" / "ThaiNER.jsonl"
    output_file = script_dir.parent / "data" / "Other_Clusters.jsonl"

    print("Loading data from ThaiNER.jsonl...")

    # Load data
    data = load_data(jsonl_file)
    if not data:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded {len(data)} samples")

    # Get category mapping with clustering
    category_mapping = get_category_mapping_with_clustering(data)

    # Filter samples where domain is in meaningful cluster categories
    meaningful_clusters = {'Mixed_Content', 'Min_Nan_Chinese', 'NGO', 'Malay_Dialects', 'Regional_Dialects', 'Cantonese_Chinese'}
    other_cluster_samples = []
    for sample in data:
        domain = sample.get('domain', 'unknown')
        category = category_mapping.get(domain, 'Other')
        if category in meaningful_clusters:
            other_cluster_samples.append(sample)

    print(f"Found {len(other_cluster_samples)} samples in clustered categories")

    # Write to new JSONL file
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in other_cluster_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

    print(f"Extraction complete! Created {output_file} with {len(other_cluster_samples)} samples.")

if __name__ == "__main__":
    main()