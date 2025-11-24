 import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataLoader:
    def __init__(self):
        self.products = {
            1: "Milk", 2: "Bread", 3: "Eggs", 4: "Butter", 5: "Cheese",
            6: "Yogurt", 7: "Apple", 8: "Banana", 9: "Coffee", 10: "Tea",
            11: "Rice", 12: "Pasta", 13: "Chicken", 14: "Fish", 15: "Juice"
        }
        
        self.categories = {
            "Milk": "Dairy", "Bread": "Bakery", "Eggs": "Dairy", "Butter": "Dairy",
            "Cheese": "Dairy", "Yogurt": "Dairy", "Apple": "Fruits", "Banana": "Fruits",
            "Coffee": "Beverages", "Tea": "Beverages", "Rice": "Grains", "Pasta": "Grains",
            "Chicken": "Meat", "Fish": "Meat", "Juice": "Beverages"
        }
    
    def generate_sample_data(self, num_customers=50, num_transactions=1000):
        """Generate sample purchase data"""
        purchases = []
        
        for i in range(num_transactions):
            customer_id = random.randint(1, num_customers)
            product_id = random.randint(1, len(self.products))
            quantity = random.randint(1, 3)
            
            # Generate random date in last 90 days
            days_ago = random.randint(1, 90)
            purchase_date = datetime.now() - timedelta(days=days_ago)
            
            purchases.append({
                'customer_id': customer_id,
                'product_id': product_id,
                'product_name': self.products[product_id],
                'category': self.categories[self.products[product_id]],
                'quantity': quantity,
                'purchase_date': purchase_date
            })
        
        return pd.DataFrame(purchases)
    
    def get_customer_purchases(self, customer_id, purchase_data):
        """Get purchase history for specific customer"""
        return purchase_data[purchase_data['customer_id'] == customer_id]
