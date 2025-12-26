---
annotations_creators:
- AI-generated
language_creators:
- AI-generated
language:
- th
license:
- cc-by-3.0
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- synthetic
task_categories:
- token-classification
task_ids:
- named-entity-recognition
paperswithcode_id: null
dataset_info:
  features:
  - name: id
    dtype: string
  - name: domain
    dtype: string
  - name: tokens
    sequence: string
  - name: tags
    sequence: string
  splits:
  - name: train
    num_bytes: 2048576
    num_examples: 10345
  download_size: 2048576
  dataset_size: 2048576
configs:
- config_name: default
  data_files:
  - split: train
    path: ThaiNER.jsonl
---

# Thai Named Entity Recognition (NER) Corpus

A comprehensive corpus for Thai Named Entity Recognition tasks with 6,748 annotated sentences across 160 different domains.

## Overview

This dataset contains Thai text samples annotated with named entity labels for training and evaluating NER models. The corpus covers a wide variety of domains including government, finance, legal, healthcare, education, and more.

## Author

**Nattapong Tapachoom**

## Dataset Statistics

- **Total Samples:** 10,345 annotated sentences
- **Number of Domains:** 441
- **Number of Domain Categories:** 21 (including auto-clustered groups)
- **Format:** JSONL (JSON Lines)

### Domain Categories by Sample Count

| Category | Count | Percentage |
|----------|-------|-----------|
| Finance & Business | 1663 | 16.1% |
| Technology & IT | 1537 | 14.9% |
| Travel & Transportation | 1157 | 11.2% |
| Mixed_Content | 1033 | 10.0% |
| Government & Administration | 808 | 7.8% |
| Healthcare & Medical | 804 | 7.8% |
| Legal & Law | 678 | 6.6% |
| Media & Entertainment | 524 | 5.1% |
| Education & Research | 498 | 4.8% |
| Regional & Minority Languages | 353 | 3.4% |
| Environment & Agriculture | 271 | 2.6% |
| Food & Lifestyle | 252 | 2.4% |
| Telecom & Communication | 194 | 1.9% |
| Real Estate & Property | 186 | 1.8% |
| Energy & Infrastructure | 183 | 1.8% |
| Retail & E-commerce | 92 | 0.9% |
| Malay_Dialects | 33 | 0.3% |
| Min_Nan_Chinese | 32 | 0.3% |
| Regional_Dialects | 28 | 0.3% |
| NGO | 10 | 0.1% |
| Cantonese_Chinese | 9 | 0.1% |

*Note: Domains are automatically categorized using keyword matching. The "Other" domains (120 total) are further clustered using TF-IDF vectorization and K-Means clustering to create more meaningful groups.*

### Domain Distribution Visualization

```
Top 10 Domain Categories by Sample Count:
Finance & Business          █████████████████ 1663 (16.1%)
Technology & IT             ████████████████  1537 (14.9%)
Travel & Transportation     ████████████     1157 (11.2%)
Mixed_Content               ███████████      1033 (10.0%)
Government & Administration ████████         808 (7.8%)
Healthcare & Medical        ████████         804 (7.8%)
Legal & Law                 ███████          678 (6.6%)
Media & Entertainment       █████            524 (5.1%)
Education & Research        █████            498 (4.8%)
Regional & Minority Languages ███             353 (3.4%)
```

*Note: Domains are grouped into categories using keyword matching and automatic clustering. The "Other" domains are clustered using TF-IDF and K-Means into meaningful groups like Mixed_Content, Min_Nan_Chinese, Malay_Dialects, etc.*

### Main File: `ThaiNER.jsonl`

Each line contains a JSON object with the following structure:

```json
{
  "id": "thner_0001",
  "domain": "government",
  "tokens": ["นาย", "พิพัฒพงศ์", "ทุนิจจา", "ยื่น", "คำร้อง", "ต่อ", "กรม", "การปกครอง", "..."],
  "tags": ["O", "B-PERSON", "I-PERSON", "O", "O", "O", "B-ORGANIZATION", "I-ORGANIZATION", "..."]
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (thner_XXXX format) |
| `domain` | string | Category/domain of the text |
| `tokens` | array | List of Thai words/tokens |
| `tags` | array | NER tags in BIO scheme |

### Named Entity Tags (BIO Scheme)

- **O** - Outside (not a named entity)
- **B-PERSON** - Beginning of person name
- **I-PERSON** - Inside person name
- **B-ORGANIZATION** - Beginning of organization
- **I-ORGANIZATION** - Inside organization
- **B-LOCATION** - Beginning of location
- **I-LOCATION** - Inside location
- **B-DATE** - Beginning of date
- **I-DATE** - Inside date
- **B-TIME** - Beginning of time
- **I-TIME** - Inside time
- **B-MONEY** - Beginning of money amount
- **I-MONEY** - Inside money amount
- **B-ID** - Beginning of ID/code
- **I-ID** - Inside ID/code
- **B-ACCOUNT** - Beginning of account number
- **B-PRODUCT** - Beginning of product
- **B-PHONE** - Beginning of phone number
- **B-EMAIL** - Beginning of email
- **I-EMAIL** - Inside email
- **B-TICKER** - Beginning of stock ticker

*Note: Tags may include other entity types depending on the domain context.*

## Files in This Directory

- **data/** - Data files directory
  - `ThaiNER.jsonl` - Main corpus file with all 10,345 annotated samples
  - `ThaiNER_backup.jsonl` - Backup of the original corpus
- **README.md** - This documentation file
- **requirements.txt** - Python dependencies for visualization scripts
- **LICENSE.txt** - License information

## Analysis and Visualization

### Domain Analysis Script

The repository includes a Python script (`src/visualize_domains.py`) for analyzing domain distributions and generating visualizations:

```bash
python src/visualize_domains.py
```

This script:
- Loads the JSONL data
- Categorizes domains using keyword matching
- Automatically clusters remaining "Other" domains using TF-IDF vectorization and K-Means clustering
- Generates bar and pie charts showing domain distribution
- Prints detailed statistics to console

**Output Files:**
- `image/domain_distribution_bar.png` - Bar chart of top domain categories
- `image/domain_distribution_pie.png` - Pie chart of domain distribution

**Clustering Method:**
- Domains not matching predefined categories are grouped as "Other"
- "Other" domains are clustered using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- K-Means algorithm groups similar domains into clusters (typically 2-10 clusters based on data size)
- Clusters are assigned meaningful names based on their content:
  - **Mixed_Content**: Automotive, culture, religion, customer support, web, minority languages
  - **Min_Nan_Chinese**: Min Nan (Southern Min) Chinese dialects
  - **NGO**: Non-governmental organizations
  - **Malay_Dialects**: Malay language variants
  - **Regional_Dialects**: Regional dialects (Chinese, Khmer, Karen)
  - **Cantonese_Chinese**: Cantonese Chinese dialect

### Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Loading the Data

**Python:**
```python
import json

# Load all samples
data = []
with open('ThaiNER.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# Access sample
sample = data[0]
print(f"ID: {sample['id']}")
print(f"Domain: {sample['domain']}")
print(f"Tokens: {sample['tokens']}")
print(f"Tags: {sample['tags']}")
```
### Example

```json
{
  "id": "thner_0001",
  "domain": "government",
  "tokens": ["นาย", "กิตติพงษ์", "ศรีทอง", "ยื่น", "คำร้อง", "ต่อ", "กรม", "การปกครอง", "..."],
  "tags": ["O", "B-PERSON", "I-PERSON", "O", "O", "O", "B-ORGANIZATION", "I-ORGANIZATION", "..."]
}
```

### Domain Distribution Visualization

To create visual charts of domain distribution:

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the visualization script:
```bash
python src/visualize_domains.py
```

This will generate:
- `domain_distribution_bar.png` - Bar chart showing top 15 domains
- `domain_distribution_pie.png` - Pie chart showing top 10 domains + others
- Console output with detailed statistics

## Data Domains

The corpus covers 160 different domains including:

**Major Domains:**
Government, Finance, Legal, Travel, Education, Healthcare, Logistics, E-commerce, Technology, Retail

**Other Domains:**
Medical, Telecom, Entertainment, Real Estate, News, Sports, Weather, IT, Energy, Food, HR, Automotive, Insurance, Business, Media, Law, Banking, Social Media, Aviation, Agriculture, Security, Transport, Politics, Customer Support, Construction, Hospitality, Manufacturing, Utilities, Gaming, Environment, Shopping, and many more...

## Data Source

This corpus was generated through synthesis by:
- **ChatGPT 5.2**
- **Gemini 3 Fast**
- **Gemini 3 Pro**

The synthetic data was then carefully annotated with named entity labels to create a high-quality training dataset for Thai NER tasks.

## Data Quality

- AI-synthesized sentences across diverse domains
- Consistent tagging across domains
- Handles multi-domain overlaps
- Supports Thai language specifics
- Includes minority languages and regional dialects of Thailand, though accuracy may vary due to limited available reference data for these languages (data for minority languages may not be 100% accurate or may be completely inaccurate)

## License

This dataset is licensed under the [Creative Commons Attribution 3.0 Unported (CC BY 3.0)](https://creativecommons.org/licenses/by/3.0/) license.

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

See the full license at: https://creativecommons.org/licenses/by/3.0/

## Citation

If you use this corpus in your research, please cite:

```bibtex
@dataset{thainer_corpus_2025,
  title={Thai Named Entity Recognition Corpus},
  year={2025},
  howpublished={https://github.com/[username]/Natural-Language-Processing}
}
```

## Contact & Contributions

For issues, questions, or contributions, please create an issue or pull request on GitHub.

---

**Last Updated:** December 26, 2025  
**Total Samples:** 10,345  
**Total Domains:** 441
