# model_integration.py
import os
from anthropic import Anthropic
from typing import Dict, List
import json

class ModelInterface:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Missing Anthropic API key in .env file")
        self.client = Anthropic(api_key=api_key)
        
    def format_tweet_context(self, tweets: List[Dict]) -> str:
        """Format tweets into a context string"""
        context = []
        for tweet in tweets:
            # Support both 'id' and 'tweet_id' fields
            tweet_id = tweet.get('id') or tweet.get('tweet_id', 'unknown')
            context.append(
                f"Tweet ID: {tweet_id}\n"
                f"From User ID: {tweet.get('author_id', 'unknown')}\n"
                f"Content: {tweet.get('text', '')}\n"
                f"Time: {tweet.get('created_at', 'unknown')}\n"
                f"---"
            )
        return "\n".join(context)
        
    def get_response(self, notifications: List[Dict], relevant_context: List[Dict]) -> Dict:
        """Get model's response to notifications with context"""
        
        # Format the context
        notifications_text = self.format_tweet_context(notifications)
        context_text = self.format_tweet_context(relevant_context) if relevant_context else "No relevant past context."
        
        prompt = f"""You're having casual conversations on Twitter. Just be yourself - no need to be preachy or overly formal. Think of this as hanging out and chatting about interesting ideas.

Here are some recent tweets you've received:

{notifications_text}

Some context from earlier:
{context_text}

How would you like to respond? Just give me the JSON in this format:
{{
    "action": "reply" or "post" or "like" or "retweet",
    "tweet_id": "ID of tweet to reply to/like/retweet (if needed)",
    "text": "Your response (if posting/replying)"
}}

Just the JSON please, no other text."""

        # Get model response
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        try:
            result = json.loads(response.content[0].text)
            print(f"Parsed response: {result}")  # Debug output
            return result
        except json.JSONDecodeError as e:
            print(f"Error parsing model response: {response.content[0].text}")
            print(f"JSON error: {str(e)}")
            return {"error": "Failed to parse model response"}