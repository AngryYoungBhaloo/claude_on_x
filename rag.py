# rag.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os

class TweetStore:
    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = storage_dir
        self.tweets_file = os.path.join(storage_dir, "tweets.json")
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
            
        # Initialize or load tweets
        if os.path.exists(self.tweets_file):
            with open(self.tweets_file, 'r') as f:
                self.tweets = json.load(f)
        else:
            self.tweets = []
            
        # Initialize embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_size = 384  # Size of embeddings from all-MiniLM-L6-v2
        
        # Initialize FAISS index
        if self.tweets:
            self.index = faiss.IndexFlatL2(self.embedding_size)
            embeddings = [self.model.encode(tweet['text']) for tweet in self.tweets]
            self.index.add(np.array(embeddings))
        else:
            self.index = faiss.IndexFlatL2(self.embedding_size)

    def _save_tweets(self):
        """Save tweets to JSON file"""
        with open(self.tweets_file, 'w') as f:
            json.dump(self.tweets, f)

    def store_tweet(self, tweet_data: Dict):
        """Store a tweet and its embedding"""
        # Ensure required fields exist
        required_fields = ['tweet_id', 'text']
        if not all(field in tweet_data for field in required_fields):
            raise ValueError(f"Tweet data missing required fields: {required_fields}")
        
        # Check if tweet already exists
        existing_tweet = next(
            (t for t in self.tweets if t['tweet_id'] == tweet_data['tweet_id']), 
            None
        )
        
        if existing_tweet:
            # Update existing tweet
            existing_idx = self.tweets.index(existing_tweet)
            self.tweets[existing_idx] = tweet_data
            
            # Update embedding
            embedding = self.model.encode(tweet_data['text'])
            if self.index.ntotal > existing_idx:
                # Remove old embedding and add new one
                old_embeddings = [self.model.encode(t['text']) for t in self.tweets[:existing_idx]]
                new_embeddings = [self.model.encode(t['text']) for t in self.tweets[existing_idx + 1:]]
                self.index = faiss.IndexFlatL2(self.embedding_size)
                if old_embeddings:
                    self.index.add(np.array(old_embeddings))
                self.index.add(np.array([embedding]))
                if new_embeddings:
                    self.index.add(np.array(new_embeddings))
        else:
            # Add new tweet
            self.tweets.append(tweet_data)
            
            # Add new embedding
            embedding = self.model.encode(tweet_data['text'])
            self.index.add(np.array([embedding]))
        
        # Save to file
        self._save_tweets()

    def retrieve_context(self, query: str, k: int = 5) -> List[Dict]:
        """Find most relevant tweets for a query"""
        if not self.tweets:
            return []
            
        query_embedding = self.model.encode(query)
        
        # Search FAISS index
        D, I = self.index.search(np.array([query_embedding]), min(k, len(self.tweets)))
        
        # Return relevant tweets
        return [self.tweets[i] for i in I[0] if i < len(self.tweets)]

    def get_thread(self, tweet_id: str) -> List[Dict]:
        """Get all tweets in a thread"""
        thread = []
        
        # Find the root tweet (traverse up)
        current_id = tweet_id
        while current_id:
            tweet = next(
                (t for t in self.tweets if t['tweet_id'] == current_id), 
                None
            )
            if tweet:
                thread.insert(0, tweet)
                current_id = tweet.get('in_reply_to_status_id')
            else:
                break
                
        # Find all replies (traverse down)
        replies = [
            t for t in self.tweets 
            if t.get('in_reply_to_status_id') == tweet_id
        ]
        thread.extend(replies)
        
        return thread

    def get_tweet(self, tweet_id: str) -> Optional[Dict]:
        """Get a specific tweet by ID"""
        return next(
            (t for t in self.tweets if t['tweet_id'] == tweet_id), 
            None
        )

    def get_recent_tweets(self, limit: int = 10) -> List[Dict]:
        """Get most recent tweets"""
        # Sort by created_at if available, otherwise return last N tweets
        if self.tweets and 'created_at' in self.tweets[0]:
            sorted_tweets = sorted(
                self.tweets,
                key=lambda x: x['created_at'],
                reverse=True
            )
            return sorted_tweets[:limit]
        return self.tweets[-limit:]