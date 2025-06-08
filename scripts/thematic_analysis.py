#!/usr/bin/env python3
"""
Thematic Analysis Module for Ethiopian Banking App Reviews
Extracts keywords and groups them into actionable themes
"""

import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import re
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThematicAnalyzer:
    """
    Advanced thematic analysis for banking app reviews
    """
    
    def __init__(self, language_model='en_core_web_sm'):
        """
        Initialize thematic analyzer
        
        Args:
            language_model (str): spaCy language model to use
        """
        self._load_spacy_model(language_model)
        self._define_banking_themes()
        
    def _load_spacy_model(self, model_name):
        """Load spaCy language model"""
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"✅ Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"❌ spaCy model '{model_name}' not found. Please install with:")
            logger.error(f"python -m spacy download {model_name}")
            self.nlp = None
    
    def _define_banking_themes(self):
        """
        Define predefined banking themes with associated keywords
        """
        self.theme_keywords = {
            'Account Access Issues': [
                'login', 'password', 'authentication', 'access', 'locked', 'activation',
                'pin', 'forgot', 'reset', 'cannot', 'login', 'sign', 'register',
                'unlock', 'block', 'disable', 'account', 'inactive'
            ],
            
            'Transaction Performance': [
                'transfer', 'payment', 'transaction', 'money', 'send', 'receive',
                'balance', 'withdraw', 'deposit', 'failed', 'pending', 'delay',
                'slow', 'fast', 'quick', 'instant', 'processing', 'successful'
            ],
            
            'User Interface & Experience': [
                'ui', 'interface', 'design', 'layout', 'navigation', 'menu',
                'button', 'screen', 'display', 'easy', 'difficult', 'simple',
                'complex', 'user', 'friendly', 'confusing', 'intuitive', 'beautiful'
            ],
            
            'Technical Issues': [
                'crash', 'bug', 'error', 'problem', 'issue', 'glitch', 'freeze',
                'loading', 'connection', 'network', 'server', 'down', 'maintenance',
                'update', 'version', 'compatibility', 'performance'
            ],
            
            'Customer Support': [
                'support', 'help', 'service', 'staff', 'customer', 'assistance',
                'response', 'solve', 'complaint', 'feedback', 'contact', 'call',
                'branch', 'representative', 'professional', 'helpful'
            ],
            
            'Feature Requests': [
                'feature', 'add', 'include', 'missing', 'need', 'want', 'wish',
                'suggest', 'improve', 'enhancement', 'upgrade', 'new', 'additional',
                'option', 'functionality', 'capability'
            ]
        }
        
        # Create reverse mapping for quick lookup
        self.keyword_to_theme = {}
        for theme, keywords in self.theme_keywords.items():
            for keyword in keywords:
                self.keyword_to_theme[keyword.lower()] = theme
    
    def preprocess_text(self, text, remove_stopwords=True, lemmatize=True):
        """
        Preprocess text for thematic analysis
        
        Args:
            text (str): Input text
            remove_stopwords (bool): Whether to remove stop words
            lemmatize (bool): Whether to lemmatize tokens
            
        Returns:
            str: Preprocessed text
        """
        if not text or not self.nlp:
            return ""
        
        # Convert to lowercase and clean
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Process with spaCy
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            # Skip if stop word (optional)
            if remove_stopwords and token.is_stop:
                continue
            
            # Skip if not alphabetic or too short
            if not token.is_alpha or len(token.text) < 3:
                continue
            
            # Add lemma or original token
            if lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)
        
        return ' '.join(tokens)
    
    def extract_keywords_tfidf(self, texts, max_features=1000, ngram_range=(1, 3)):
        """
        Extract keywords using TF-IDF
        
        Args:
            texts (list): List of preprocessed texts
            max_features (int): Maximum number of features
            ngram_range (tuple): N-gram range for TF-IDF
            
        Returns:
            tuple: (tfidf_matrix, feature_names, vectorizer)
        """
        logger.info("Extracting keywords using TF-IDF...")
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        
        if not valid_texts:
            logger.warning("No valid texts for TF-IDF analysis")
            return None, [], None
        
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            stop_words='english'
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        logger.info(f"Extracted {len(feature_names)} keywords/phrases")
        
        return tfidf_matrix, feature_names, vectorizer
    
    def extract_keywords_spacy(self, texts, min_freq=2):
        """
        Extract keywords using spaCy NER and linguistic features
        
        Args:
            texts (list): List of texts to analyze
            min_freq (int): Minimum frequency for keyword inclusion
            
        Returns:
            dict: Keywords with frequencies and types
        """
        if not self.nlp:
            return {}
        
        logger.info("Extracting keywords using spaCy...")
        
        keyword_freq = Counter()
        keyword_types = defaultdict(set)
        
        for text in tqdm(texts, desc="Processing texts with spaCy"):
            if not text.strip():
                continue
                
            doc = self.nlp(text)
            
            # Extract entities
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'MONEY', 'TIME']:
                    keyword_freq[ent.text.lower()] += 1
                    keyword_types[ent.text.lower()].add('entity')
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit phrase length
                    keyword_freq[chunk.text.lower()] += 1
                    keyword_types[chunk.text.lower()].add('noun_phrase')
            
            # Extract important single tokens
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                    not token.is_stop and 
                    len(token.text) > 3):
                    keyword_freq[token.lemma_.lower()] += 1
                    keyword_types[token.lemma_.lower()].add(token.pos_)
        
        # Filter by minimum frequency
        filtered_keywords = {
            keyword: freq for keyword, freq in keyword_freq.items() 
            if freq >= min_freq
        }
        
        logger.info(f"Extracted {len(filtered_keywords)} keywords with spaCy")
        
        return {
            'frequencies': filtered_keywords,
            'types': dict(keyword_types)
        }
    
    def assign_themes_rule_based(self, keywords_data):
        """
        Assign themes to keywords using rule-based approach
        
        Args:
            keywords_data (dict): Keywords with frequencies
            
        Returns:
            dict: Theme assignments with confidence scores
        """
        logger.info("Assigning themes using rule-based approach...")
        
        theme_assignments = defaultdict(list)
        keyword_themes = {}
        
        keywords = keywords_data.get('frequencies', {})
        
        for keyword, frequency in keywords.items():
            assigned_themes = []
            
            # Check against predefined theme keywords
            for theme, theme_keywords in self.theme_keywords.items():
                # Direct match
                if keyword in theme_keywords:
                    assigned_themes.append((theme, 1.0))
                    continue
                
                # Partial match (keyword contains theme keyword)
                for theme_keyword in theme_keywords:
                    if theme_keyword in keyword or keyword in theme_keyword:
                        assigned_themes.append((theme, 0.7))
                        break
            
            # If no theme assigned, try contextual assignment
            if not assigned_themes:
                assigned_themes = self._contextual_theme_assignment(keyword)
            
            # Store assignments
            if assigned_themes:
                # Take the highest confidence theme
                best_theme, confidence = max(assigned_themes, key=lambda x: x[1])
                keyword_themes[keyword] = {
                    'theme': best_theme,
                    'confidence': confidence,
                    'frequency': frequency
                }
                theme_assignments[best_theme].append(keyword)
        
        logger.info(f"Assigned themes to {len(keyword_themes)} keywords")
        
        return {
            'keyword_themes': keyword_themes,
            'theme_keywords': dict(theme_assignments)
        }
    
    def _contextual_theme_assignment(self, keyword):
        """
        Assign theme based on contextual rules
        
        Args:
            keyword (str): Keyword to assign theme to
            
        Returns:
            list: List of (theme, confidence) tuples
        """
        assignments = []
        
        # Technical words
        if any(word in keyword for word in ['app', 'system', 'software', 'technical']):
            assignments.append(('Technical Issues', 0.6))
        
        # Money/transaction related
        if any(word in keyword for word in ['money', 'birr', 'amount', 'cost', 'fee']):
            assignments.append(('Transaction Performance', 0.6))
        
        # Time related
        if any(word in keyword for word in ['time', 'hour', 'minute', 'day', 'week']):
            assignments.append(('Transaction Performance', 0.5))
        
        # Quality adjectives
        if any(word in keyword for word in ['good', 'bad', 'excellent', 'poor', 'best', 'worst']):
            assignments.append(('User Interface & Experience', 0.5))
        
        return assignments
    
    def analyze_themes_by_bank(self, df, text_column='review'):
        """
        Analyze themes separately for each bank
        
        Args:
            df (pd.DataFrame): DataFrame with reviews
            text_column (str): Column containing review text
            
        Returns:
            dict: Thematic analysis results by bank
        """
        logger.info("Starting thematic analysis by bank...")
        
        results = {}
        
        for bank in df['bank'].unique():
            logger.info(f"Analyzing themes for {bank}...")
            
            bank_data = df[df['bank'] == bank].copy()
            bank_texts = bank_data[text_column].fillna('').astype(str).tolist()
            
            # Preprocess texts
            preprocessed_texts = [
                self.preprocess_text(text) for text in bank_texts
            ]
            
            # Extract keywords using both methods
            tfidf_results = self.extract_keywords_tfidf(preprocessed_texts)
            spacy_results = self.extract_keywords_spacy(bank_texts)
            
            # Combine keyword extraction results
            combined_keywords = self._combine_keyword_results(
                tfidf_results, spacy_results
            )
            
            # Assign themes
            theme_assignments = self.assign_themes_rule_based(combined_keywords)
            
            # Calculate theme statistics
            theme_stats = self._calculate_theme_statistics(
                theme_assignments, bank_data
            )
            
            results[bank] = {
                'total_reviews': len(bank_data),
                'keywords': combined_keywords,
                'theme_assignments': theme_assignments,
                'theme_statistics': theme_stats,
                'top_themes': self._get_top_themes(theme_stats, top_n=5)
            }
        
        return results
    
    def _combine_keyword_results(self, tfidf_results, spacy_results):
        """
        Combine TF-IDF and spaCy keyword extraction results
        
        Args:
            tfidf_results (tuple): TF-IDF results
            spacy_results (dict): spaCy results
            
        Returns:
            dict: Combined keyword frequencies
        """
        combined = {'frequencies': {}}
        
        # Add spaCy results
        if spacy_results and 'frequencies' in spacy_results:
            combined['frequencies'].update(spacy_results['frequencies'])
        
        # Add top TF-IDF results
        if tfidf_results[0] is not None:
            tfidf_matrix, feature_names, vectorizer = tfidf_results
            
            # Get mean TF-IDF scores
            mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # Get top features
            top_indices = mean_scores.argsort()[-100:][::-1]  # Top 100 TF-IDF terms
            
            for idx in top_indices:
                term = feature_names[idx]
                score = mean_scores[idx]
                
                # Convert TF-IDF score to frequency-like measure
                frequency = int(score * 1000)  # Scale for comparison
                
                if term in combined['frequencies']:
                    combined['frequencies'][term] += frequency
                else:
                    combined['frequencies'][term] = frequency
        
        return combined
    
    def _calculate_theme_statistics(self, theme_assignments, bank_data):
        """
        Calculate statistics for each theme
        
        Args:
            theme_assignments (dict): Theme assignment results
            bank_data (pd.DataFrame): Bank-specific data
            
        Returns:
            dict: Theme statistics
        """
        theme_stats = {}
        
        for theme, keywords in theme_assignments['theme_keywords'].items():
            # Count mentions across reviews
            mentions = 0
            total_frequency = 0
            
            for keyword in keywords:
                if keyword in theme_assignments['keyword_themes']:
                    freq = theme_assignments['keyword_themes'][keyword]['frequency']
                    total_frequency += freq
                    mentions += 1
            
            # Calculate coverage (reviews mentioning this theme)
            theme_coverage = self._calculate_theme_coverage(
                keywords, bank_data['review'].fillna('').astype(str).tolist()
            )
            
            theme_stats[theme] = {
                'keyword_count': len(keywords),
                'total_frequency': total_frequency,
                'mention_count': mentions,
                'coverage_percentage': theme_coverage,
                'top_keywords': sorted(keywords, key=lambda k: 
                    theme_assignments['keyword_themes'].get(k, {}).get('frequency', 0),
                    reverse=True)[:10]
            }
        
        return theme_stats
    
    def _calculate_theme_coverage(self, theme_keywords, reviews):
        """
        Calculate what percentage of reviews mention this theme
        
        Args:
            theme_keywords (list): Keywords for this theme
            reviews (list): List of review texts
            
        Returns:
            float: Coverage percentage
        """
        if not reviews:
            return 0.0
        
        reviews_with_theme = 0
        
        for review in reviews:
            review_lower = review.lower()
            if any(keyword in review_lower for keyword in theme_keywords):
                reviews_with_theme += 1
        
        return (reviews_with_theme / len(reviews)) * 100
    
    def _get_top_themes(self, theme_stats, top_n=5):
        """
        Get top themes by total frequency
        
        Args:
            theme_stats (dict): Theme statistics
            top_n (int): Number of top themes to return
            
        Returns:
            list: Top themes with statistics
        """
        sorted_themes = sorted(
            theme_stats.items(),
            key=lambda x: x[1]['total_frequency'],
            reverse=True
        )
        
        return sorted_themes[:top_n]
    
    def create_theme_assignment_dataframe(self, df, thematic_results, text_column='review'):
        """
        Create DataFrame with theme assignments for each review
        
        Args:
            df (pd.DataFrame): Original DataFrame
            thematic_results (dict): Results from analyze_themes_by_bank
            text_column (str): Column containing review text
            
        Returns:
            pd.DataFrame: DataFrame with theme assignments
        """
        logger.info("Creating theme assignment DataFrame...")
        
        results_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Assigning themes to reviews"):
            bank = row['bank']
            review_text = str(row[text_column]) if pd.notna(row[text_column]) else ""
            
            # Get themes for this bank
            bank_themes = thematic_results.get(bank, {}).get('theme_assignments', {})
            
            # Find themes mentioned in this review
            mentioned_themes = self._find_themes_in_review(
                review_text, bank_themes.get('keyword_themes', {})
            )
            
            # Create result row
            result_row = {
                'review_id': idx,
                'review_text': review_text,
                'bank': bank,
                'rating': row.get('rating', 0),
                'date': row.get('date', ''),
                'identified_themes': ', '.join(mentioned_themes['themes']),
                'theme_count': len(mentioned_themes['themes']),
                'primary_theme': mentioned_themes['primary_theme'],
                'theme_confidence': mentioned_themes['confidence']
            }
            
            # Add individual theme flags
            for theme in self.theme_keywords.keys():
                result_row[f'theme_{theme.lower().replace(" ", "_").replace("&", "and")}'] = \
                    theme in mentioned_themes['themes']
            
            results_list.append(result_row)
        
        return pd.DataFrame(results_list)
    
    def _find_themes_in_review(self, review_text, keyword_themes):
        """
        Find themes mentioned in a specific review
        
        Args:
            review_text (str): Review text
            keyword_themes (dict): Keyword to theme mappings
            
        Returns:
            dict: Themes found in the review
        """
        review_lower = review_text.lower()
        mentioned_themes = defaultdict(float)
        
        # Check for keyword mentions
        for keyword, theme_data in keyword_themes.items():
            if keyword in review_lower:
                theme = theme_data['theme']
                confidence = theme_data['confidence']
                mentioned_themes[theme] += confidence
        
        # Get themes sorted by confidence
        sorted_themes = sorted(
            mentioned_themes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        themes = [theme for theme, _ in sorted_themes]
        primary_theme = themes[0] if themes else 'Unclassified'
        confidence = sorted_themes[0][1] if sorted_themes else 0.0
        
        return {
            'themes': themes,
            'primary_theme': primary_theme,
            'confidence': confidence
        }

def generate_thematic_summary(thematic_results):
    """
    Generate a comprehensive summary of thematic analysis results
    
    Args:
        thematic_results (dict): Results from thematic analysis
        
    Returns:
        dict: Summary statistics and insights
    """
    summary = {
        'overall_statistics': {},
        'bank_comparisons': {},
        'theme_insights': {}
    }
    
    # Overall statistics
    total_reviews = sum(bank_data['total_reviews'] for bank_data in thematic_results.values())
    all_themes = set()
    
    for bank_data in thematic_results.values():
        themes = bank_data.get('theme_statistics', {}).keys()
        all_themes.update(themes)
    
    summary['overall_statistics'] = {
        'total_reviews_analyzed': total_reviews,
        'total_banks': len(thematic_results),
        'unique_themes_identified': len(all_themes),
        'themes_per_bank_avg': len(all_themes) / len(thematic_results) if thematic_results else 0
    }
    
    # Bank comparisons
    for bank, bank_data in thematic_results.items():
        top_themes = bank_data.get('top_themes', [])
        summary['bank_comparisons'][bank] = {
            'total_reviews': bank_data['total_reviews'],
            'themes_identified': len(bank_data.get('theme_statistics', {})),
            'top_theme': top_themes[0][0] if top_themes else 'None',
            'top_theme_frequency': top_themes[0][1]['total_frequency'] if top_themes else 0
        }
    
    # Theme insights across all banks
    theme_totals = defaultdict(int)
    for bank_data in thematic_results.values():
        for theme, stats in bank_data.get('theme_statistics', {}).items():
            theme_totals[theme] += stats['total_frequency']
    
    summary['theme_insights'] = dict(sorted(
        theme_totals.items(),
        key=lambda x: x[1],
        reverse=True
    ))
    
    return summary

def main():
    """Main function for testing thematic analysis"""
    # Load data
    try:
        df = pd.read_csv('../data/processed/cleaned_banking_reviews.csv')
        logger.info(f"Loaded {len(df)} reviews for thematic analysis")
    except FileNotFoundError:
        logger.error("Preprocessed data not found. Please run data preprocessing first.")
        return
    
    # Initialize analyzer
    analyzer = ThematicAnalyzer()
    
    # Run thematic analysis
    thematic_results = analyzer.analyze_themes_by_bank(df.head(300))  # Test with subset
    
    # Create theme assignments DataFrame
    theme_df = analyzer.create_theme_assignment_dataframe(df.head(300), thematic_results)
    
    # Generate summary
    summary = generate_thematic_summary(thematic_results)
    
    # Display results
    print("\n🎯 Thematic Analysis Results:")
    print(f"Total Reviews Analyzed: {summary['overall_statistics']['total_reviews_analyzed']}")
    print(f"Unique Themes Identified: {summary['overall_statistics']['unique_themes_identified']}")
    
    print("\n🏦 Top Themes by Bank:")
    for bank, stats in summary['bank_comparisons'].items():
        print(f"  {bank}: {stats['top_theme']} (frequency: {stats['top_theme_frequency']})")
    
    # Save results
    output_path = '../data/processed/thematic_analysis_results.csv'
    theme_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main() 