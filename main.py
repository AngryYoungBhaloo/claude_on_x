# main.py
import schedule
import time
from twitter_client import TwitterClientV2
from rag import TweetStore
from model_integration import ModelInterface
from datetime import datetime, timezone

class TwitterBot:
    def __init__(self):
        self.twitter = TwitterClientV2()
        self.store = TweetStore()
        self.model = ModelInterface()
        
    def run_cycle(self):
        """Main bot cycle - aware of free tier limits"""
        try:
            print(f"\nStarting bot cycle at {datetime.now(timezone.utc).isoformat()}")
            
            # Step 1: Check notifications (1/15min limit)
            notifications = self.twitter.check_notifications()
            if notifications:
                print(f"Found {len(notifications)} new notifications")
                
                # Store new notifications
                for notification in notifications:
                    self.store.store_tweet(notification)
                
                # Get relevant context
                context_query = " ".join([n['text'] for n in notifications])
                relevant_context = self.store.retrieve_context(context_query)
                
                print("Getting model response...")
                # Get model response
                response = self.model.get_response(notifications, relevant_context)
                print(f"Model response: {response}")
                
                # Execute model's decision
                if response.get("action") == "reply":
                    print("Attempting to reply to a tweet...")
                    result = self.twitter.reply_tweet(
                        response["tweet_id"],
                        response["text"]
                    )
                    if result:
                        print("Successfully replied")
                        self.store.store_tweet(result)
                    
                elif response.get("action") == "post":
                    print("Attempting to post a new tweet...")
                    result = self.twitter.post_tweet(response["text"])
                    if result:
                        print("Successfully posted")
                        self.store.store_tweet(result)
            else:
                print("No new notifications")
                
        except Exception as e:
            print(f"Error in bot cycle: {str(e)}")

def main():
    bot = TwitterBot()
    
    # Schedule bot to run every 15 minutes
    schedule.every(15).minutes.do(bot.run_cycle)
    
    print("Bot started! Running first cycle...")
    # Run immediately on start
    bot.run_cycle()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()