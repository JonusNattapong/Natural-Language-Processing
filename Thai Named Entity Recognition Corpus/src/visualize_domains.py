#!/usr/bin/env python3
"""
Domain Distribution Visualization Script for Thai NER Corpus

This script reads the ThaiNER.jsonl file and creates visualizations
showing the distribution of samples across different domains.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from pathlib import Path
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
    """Categorize domain into major groups"""
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

def analyze_domains(data):
    """Analyze domain distribution with categorization"""
    # Count individual domains
    individual_domains = Counter([item.get('domain', 'unknown') for item in data])

    # Categorize domains
    categorized_counts = Counter()
    category_mapping = {}  # domain -> category

    for domain, count in individual_domains.items():
        category = categorize_domain(domain)
        categorized_counts[category] += count
        category_mapping[domain] = category

    # Automatic clustering for 'Other' domains using TF-IDF and K-Means
    other_domains = [domain for domain, cat in category_mapping.items() if cat == 'Other']
    if other_domains:
        print(f"Clustering {len(other_domains)} 'Other' domains using TF-IDF and K-Means...")
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  # Though domains are not English, but for tokenization
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
                
                # Update counts and mapping
                for cluster_name, domains in cluster_groups.items():
                    count = sum(individual_domains[d] for d in domains)
                    categorized_counts[cluster_name] = count
                    for d in domains:
                        category_mapping[d] = cluster_name
                
                # Remove the original 'Other' category
                if 'Other' in categorized_counts:
                    del categorized_counts['Other']
        except Exception as e:
            print(f"Warning: Clustering failed: {e}. Keeping 'Other' as is.")

    # Sort categories by count (descending)
    sorted_categories = sorted(categorized_counts.items(), key=lambda x: x[1], reverse=True)

    return dict(sorted_categories), len(data), dict(individual_domains), category_mapping

def create_bar_chart(domain_counts, total_samples, top_n=20, save_path=None):
    """Create a bar chart of domain distribution"""

    # Get top N domains
    top_domains = dict(list(domain_counts.items())[:top_n])
    other_count = sum(count for domain, count in list(domain_counts.items())[top_n:])

    if other_count > 0:
        top_domains['Other'] = other_count

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Use seaborn style
    sns.set_style("whitegrid")

    # Create bar chart
    domains = list(top_domains.keys())
    counts = list(top_domains.values())

    bars = plt.bar(range(len(domains)), counts, color=sns.color_palette("husl", len(domains)))

    # Add value labels on bars
    for i, (domain, count) in enumerate(zip(domains, counts)):
        percentage = (count / total_samples) * 100
        plt.text(i, count + max(counts) * 0.01, f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Customize the plot
    plt.title(f'Top {top_n} Domain Distribution in Thai NER Corpus\n(Total: {total_samples} samples)',
             fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Domain', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(range(len(domains)), domains, rotation=45, ha='right', fontsize=10)

    # Add grid
    plt.grid(axis='y', alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    else:
        plt.show()

def create_pie_chart(domain_counts, total_samples, top_n=10, save_path=None):
    """Create a pie chart of domain distribution"""

    # Get top N domains
    top_domains = dict(list(domain_counts.items())[:top_n])
    other_count = sum(count for domain, count in list(domain_counts.items())[top_n:])

    if other_count > 0:
        top_domains['Other'] = other_count

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Create pie chart
    domains = list(top_domains.keys())
    counts = list(top_domains.values())

    # Calculate percentages
    percentages = [(count / total_samples) * 100 for count in counts]

    # Create labels with percentages
    labels = [f'{domain}\n{count} ({percentage:.1f}%)'
             for domain, count, percentage in zip(domains, counts, percentages)]

    # Create pie chart
    wedges, texts, autotexts = plt.pie(counts,
                                      labels=labels,
                                      autopct='%1.1f%%',
                                      startangle=90,
                                      colors=sns.color_palette("husl", len(domains)),
                                      textprops={'fontsize': 9})

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    plt.title(f'Domain Distribution in Thai NER Corpus\n(Top {top_n} + Other)',
             fontsize=14, fontweight='bold', pad=20)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pie chart saved to: {save_path}")
    else:
        plt.show()

def print_statistics(categorized_counts, total_samples, individual_domains):
    """Print domain statistics to console"""
    print(f"\n{'='*70}")
    print(f"THAI NER CORPUS DOMAIN STATISTICS")
    print(f"{'='*70}")
    print(f"Total Samples: {total_samples}")
    print(f"Total Individual Domains: {len(individual_domains)}")
    print(f"Categorized into: {len(categorized_counts)} major groups")
    print(f"\nDomain Categories (sorted by sample count):")
    print(f"{'-'*50}")

    for i, (category, count) in enumerate(categorized_counts.items(), 1):
        percentage = (count / total_samples) * 100
        print(f"{i:2d}. {category:<30} {count:>6} ({percentage:>5.1f}%)")

    print(f"\n{'='*70}")
    print(f"TOP 10 INDIVIDUAL DOMAINS:")
    print(f"{'-'*70}")

    sorted_individual = sorted(individual_domains.items(), key=lambda x: x[1], reverse=True)
    for i, (domain, count) in enumerate(sorted_individual[:10], 1):
        percentage = (count / total_samples) * 100
        category = categorize_domain(domain)
        print(f"{i:2d}. {domain:<25} ({category:<25}) {count:>6} ({percentage:>5.1f}%)")

def main():
    """Main function"""
    # File path
    script_dir = Path(__file__).parent
    jsonl_file = script_dir.parent / "data" / "ThaiNER.jsonl"

    print("Loading data from ThaiNER.jsonl...")

    # Load data
    data = load_data(jsonl_file)
    if not data:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded {len(data)} samples")

    # Analyze domains
    categorized_counts, total_samples, individual_domains, category_mapping = analyze_domains(data)

    # Print statistics
    print_statistics(categorized_counts, total_samples, individual_domains)

    # Create visualizations
    print("\nCreating visualizations...")

    # Bar chart
    bar_chart_path = script_dir.parent / "image" / "domain_distribution_bar.png"
    create_bar_chart(categorized_counts, total_samples, top_n=15, save_path=bar_chart_path)

    # Pie chart
    pie_chart_path = script_dir.parent / "image" / "domain_distribution_pie.png"
    create_pie_chart(categorized_counts, total_samples, top_n=10, save_path=pie_chart_path)

    print("\nVisualization complete!")
    print(f"Bar chart saved: {bar_chart_path}")
    print(f"Pie chart saved: {pie_chart_path}")

if __name__ == "__main__":
    main()