#!/usr/bin/env python3
"""
Sentiment Analysis Module for Ethiopian Banking App Reviews
Supports multiple sentiment analysis approaches with comparison
"""

import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import torch
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Multi-method sentiment analysis for banking app reviews
    """
    
    def __init__(self, use_gpu=False):
        """
        Initialize sentiment analysis models
        
        Args:
            use_gpu (bool): Whether to use GPU for DistilBERT (if available)
        """
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1
        logger.info(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")
        
        # Initialize models
        self._init_distilbert()
        self._init_vader()
        
    def _init_distilbert(self):
        """Initialize DistilBERT sentiment classifier"""
        try:
            self.distilbert_classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device,
                return_all_scores=True
            )
            logger.info("✅ DistilBERT model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load DistilBERT: {e}")
            self.distilbert_classifier = None
    
    def _init_vader(self):
        """Initialize VADER sentiment analyzer"""
        try:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("✅ VADER analyzer loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load VADER: {e}")
            self.vader_analyzer = None
    
    def analyze_distilbert(self, text):
        """
        Analyze sentiment using DistilBERT
        
        Args:
            text (str): Review text to analyze
            
        Returns:
            dict: Sentiment scores and label
        """
        if not self.distilbert_classifier or not text.strip():
            return {
                'distilbert_label': 'NEUTRAL',
                'distilbert_positive': 0.5,
                'distilbert_negative': 0.5,
                'distilbert_confidence': 0.0
            }
        
        try:
            # DistilBERT has a token limit, truncate if necessary
            text = text[:512]
            
            results = self.distilbert_classifier(text)[0]
            
            # Convert to standardized format
            positive_score = 0.0
            negative_score = 0.0
            
            for result in results:
                if result['label'] == 'POSITIVE':
                    positive_score = result['score']
                elif result['label'] == 'NEGATIVE':
                    negative_score = result['score']
            
            # Determine primary label and confidence
            if positive_score > negative_score:
                label = 'POSITIVE'
                confidence = positive_score
            else:
                label = 'NEGATIVE'
                confidence = negative_score
            
            return {
                'distilbert_label': label,
                'distilbert_positive': positive_score,
                'distilbert_negative': negative_score,
                'distilbert_confidence': confidence
            }
            
        except Exception as e:
            logger.warning(f"DistilBERT analysis failed for text: {e}")
            return {
                'distilbert_label': 'NEUTRAL',
                'distilbert_positive': 0.5,
                'distilbert_negative': 0.5,
                'distilbert_confidence': 0.0
            }
    
    def analyze_vader(self, text):
        """
        Analyze sentiment using VADER
        
        Args:
            text (str): Review text to analyze
            
        Returns:
            dict: Sentiment scores and label
        """
        if not self.vader_analyzer or not text.strip():
            return {
                'vader_label': 'NEUTRAL',
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 1.0,
                'vader_compound': 0.0
            }
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine label based on compound score
            if scores['compound'] >= 0.05:
                label = 'POSITIVE'
            elif scores['compound'] <= -0.05:
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'
            
            return {
                'vader_label': label,
                'vader_positive': scores['pos'],
                'vader_negative': scores['neg'],
                'vader_neutral': scores['neu'],
                'vader_compound': scores['compound']
            }
            
        except Exception as e:
            logger.warning(f"VADER analysis failed for text: {e}")
            return {
                'vader_label': 'NEUTRAL',
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 1.0,
                'vader_compound': 0.0
            }
    
    def analyze_textblob(self, text):
        """
        Analyze sentiment using TextBlob
        
        Args:
            text (str): Review text to analyze
            
        Returns:
            dict: Sentiment scores and label
        """
        if not text.strip():
            return {
                'textblob_label': 'NEUTRAL',
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0
            }
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine label
            if polarity > 0.1:
                label = 'POSITIVE'
            elif polarity < -0.1:
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'
            
            return {
                'textblob_label': label,
                'textblob_polarity': polarity,
                'textblob_subjectivity': subjectivity
            }
            
        except Exception as e:
            logger.warning(f"TextBlob analysis failed for text: {e}")
            return {
                'textblob_label': 'NEUTRAL',
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0
            }
    
    def analyze_review(self, text):
        """
        Comprehensive sentiment analysis using all methods
        
        Args:
            text (str): Review text to analyze
            
        Returns:
            dict: Combined sentiment analysis results
        """
        results = {}
        
        # Run all analysis methods
        results.update(self.analyze_distilbert(text))
        results.update(self.analyze_vader(text))
        results.update(self.analyze_textblob(text))
        
        # Create ensemble sentiment
        results.update(self._create_ensemble_sentiment(results))
        
        return results
    
    def _create_ensemble_sentiment(self, results):
        """
        Create ensemble sentiment from multiple methods
        
        Args:
            results (dict): Individual sentiment analysis results
            
        Returns:
            dict: Ensemble sentiment scores and label
        """
        # Weight the different methods
        weights = {
            'distilbert': 0.5,  # Primary method
            'vader': 0.3,       # Good for social media text
            'textblob': 0.2     # Simple baseline
        }
        
        # Calculate weighted positive score
        positive_score = (
            results['distilbert_positive'] * weights['distilbert'] +
            results['vader_positive'] * weights['vader'] +
            max(0, results['textblob_polarity']) * weights['textblob']
        )
        
        # Calculate weighted negative score
        negative_score = (
            results['distilbert_negative'] * weights['distilbert'] +
            results['vader_negative'] * weights['vader'] +
            max(0, -results['textblob_polarity']) * weights['textblob']
        )
        
        # Determine ensemble label
        if positive_score > negative_score + 0.1:
            ensemble_label = 'POSITIVE'
        elif negative_score > positive_score + 0.1:
            ensemble_label = 'NEGATIVE'
        else:
            ensemble_label = 'NEUTRAL'
        
        return {
            'ensemble_label': ensemble_label,
            'ensemble_positive': positive_score,
            'ensemble_negative': negative_score,
            'ensemble_confidence': max(positive_score, negative_score)
        }
    
    def analyze_dataframe(self, df, text_column='review', batch_size=32):
        """
        Analyze sentiment for entire dataframe
        
        Args:
            df (pd.DataFrame): DataFrame with reviews
            text_column (str): Name of text column
            batch_size (int): Batch size for processing
            
        Returns:
            pd.DataFrame: DataFrame with sentiment analysis results
        """
        logger.info(f"Starting sentiment analysis for {len(df)} reviews...")
        
        results_list = []
        
        # Process in batches for memory efficiency
        for i in tqdm(range(0, len(df), batch_size), desc="Analyzing sentiment"):
            batch_df = df.iloc[i:i+batch_size].copy()
            
            for idx, row in batch_df.iterrows():
                text = str(row[text_column]) if pd.notna(row[text_column]) else ""
                
                # Get sentiment results
                sentiment_results = self.analyze_review(text)
                
                # Add review metadata
                result_row = {
                    'review_id': idx,
                    'review_text': text,
                    'bank': row.get('bank', ''),
                    'rating': row.get('rating', 0),
                    'date': row.get('date', ''),
                }
                
                # Add sentiment scores
                result_row.update(sentiment_results)
                
                results_list.append(result_row)
        
        results_df = pd.DataFrame(results_list)
        
        logger.info(f"✅ Sentiment analysis completed for {len(results_df)} reviews")
        
        return results_df

def calculate_sentiment_aggregations(sentiment_df):
    """
    Calculate sentiment aggregations by bank and rating
    
    Args:
        sentiment_df (pd.DataFrame): DataFrame with sentiment results
        
    Returns:
        dict: Aggregated sentiment statistics
    """
    aggregations = {}
    
    # Overall statistics
    aggregations['overall'] = {
        'total_reviews': len(sentiment_df),
        'sentiment_coverage': (sentiment_df['ensemble_label'] != 'NEUTRAL').mean() * 100,
        'positive_ratio': (sentiment_df['ensemble_label'] == 'POSITIVE').mean() * 100,
        'negative_ratio': (sentiment_df['ensemble_label'] == 'NEGATIVE').mean() * 100,
        'neutral_ratio': (sentiment_df['ensemble_label'] == 'NEUTRAL').mean() * 100
    }
    
    # By bank
    aggregations['by_bank'] = {}
    for bank in sentiment_df['bank'].unique():
        bank_data = sentiment_df[sentiment_df['bank'] == bank]
        aggregations['by_bank'][bank] = {
            'total_reviews': len(bank_data),
            'avg_positive_score': bank_data['ensemble_positive'].mean(),
            'avg_negative_score': bank_data['ensemble_negative'].mean(),
            'positive_ratio': (bank_data['ensemble_label'] == 'POSITIVE').mean() * 100,
            'negative_ratio': (bank_data['ensemble_label'] == 'NEGATIVE').mean() * 100
        }
    
    # By rating
    aggregations['by_rating'] = {}
    for rating in sorted(sentiment_df['rating'].unique()):
        rating_data = sentiment_df[sentiment_df['rating'] == rating]
        aggregations['by_rating'][rating] = {
            'total_reviews': len(rating_data),
            'avg_positive_score': rating_data['ensemble_positive'].mean(),
            'avg_negative_score': rating_data['ensemble_negative'].mean(),
            'avg_distilbert_confidence': rating_data['distilbert_confidence'].mean()
        }
    
    return aggregations

def main():
    """Main function for testing sentiment analysis"""
    # Load data
    try:
        df = pd.read_csv('../data/processed/cleaned_banking_reviews.csv')
        logger.info(f"Loaded {len(df)} reviews for sentiment analysis")
    except FileNotFoundError:
        logger.error("Preprocessed data not found. Please run data preprocessing first.")
        return
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(use_gpu=False)
    
    # Run sentiment analysis
    sentiment_results = analyzer.analyze_dataframe(df.head(100))  # Test with first 100 reviews
    
    # Calculate aggregations
    aggregations = calculate_sentiment_aggregations(sentiment_results)
    
    # Display results
    print("\n📊 Sentiment Analysis Results:")
    print(f"Total Reviews Analyzed: {aggregations['overall']['total_reviews']}")
    print(f"Sentiment Coverage: {aggregations['overall']['sentiment_coverage']:.1f}%")
    print(f"Positive: {aggregations['overall']['positive_ratio']:.1f}%")
    print(f"Negative: {aggregations['overall']['negative_ratio']:.1f}%")
    print(f"Neutral: {aggregations['overall']['neutral_ratio']:.1f}%")
    
    # Save results
    output_path = '../data/processed/sentiment_analysis_results.csv'
    sentiment_results.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main() 