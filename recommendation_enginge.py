import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

class SimpleRecommendationEngine:
    def __init__(self):
        self.customer_product_matrix = None
        self.product_similarity = None
        self.customer_similarity = None
        self.frequent_patterns = None
    
    def fit(self, purchase_data):
        """Train the recommendation models"""
        print("Training recommendation models...")
        
        # Create customer-product matrix
        self._create_customer_product_matrix(purchase_data)
        
        # Calculate product similarity
        self._calculate_product_similarity()
        
        # Calculate customer similarity
        self._calculate_customer_similarity()
        
        # Find frequent patterns
        self._find_frequent_patterns(purchase_data)
        
        print("Model training completed!")
    
    def _create_customer_product_matrix(self, purchase_data):
        """Create matrix of customers vs products"""
        # Group by customer and product, sum quantities
        customer_product = purchase_data.groupby(['customer_id', 'product_name'])['quantity'].sum().unstack(fill_value=0)
        self.customer_product_matrix = customer_product
    
    def _calculate_product_similarity(self):
        """Calculate similarity between products"""
        product_matrix = self.customer_product_matrix.T
        self.product_similarity = cosine_similarity(product_matrix)
        self.product_names = product_matrix.index.tolist()
    
    def _calculate_customer_similarity(self):
        """Calculate similarity between customers"""
        self.customer_similarity = cosine_similarity(self.customer_product_matrix)
        self.customer_ids = self.customer_product_matrix.index.tolist()
    
    def _find_frequent_patterns(self, purchase_data):
        """Find frequently bought together items"""
        self.frequent_patterns = defaultdict(int)
        
        # Group purchases by customer and date to find items bought together
        customer_dates = purchase_data.groupby(['customer_id', 'purchase_date'])['product_name'].apply(list)
        
        for items in customer_dates:
            if len(items) > 1:
                # Count pairs of items bought together
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        pair = tuple(sorted([items[i], items[j]]))
                        self.frequent_patterns[pair] += 1
    
    def recommend_products(self, customer_id, top_n=5):
        """Generate product recommendations for a customer"""
        if customer_id not in self.customer_ids:
            return ["New customer - try popular items: Milk, Bread, Eggs"]
        
        recommendations = []
        
        # Method 1: Based on customer's purchase history
        customer_idx = self.customer_ids.index(customer_id)
        customer_purchases = self.customer_product_matrix.iloc[customer_idx]
        
        # Get products customer hasn't bought much
        not_purchased = customer_purchases[customer_purchases == 0].index
        
        for product in not_purchased:
            if product in self.product_names:
                product_idx = self.product_names.index(product)
                
                # Find similar products that customer likes
                similarity_scores = []
                for purchased_product in customer_purchases[customer_purchases > 0].index:
                    if purchased_product in self.product_names:
                        purchased_idx = self.product_names.index(purchased_product)
                        similarity = self.product_similarity[product_idx][purchased_idx]
                        similarity_scores.append(similarity * customer_purchases[purchased_product])
                
                if similarity_scores:
                    score = max(similarity_scores)
                    recommendations.append((product, score))
        
        # Method 2: Based on similar customers
        similar_customers = []
        for idx, other_customer_id in enumerate(self.customer_ids):
            if other_customer_id != customer_id:
                similarity = self.customer_similarity[customer_idx][idx]
                similar_customers.append((other_customer_id, similarity))
        
        # Sort by similarity and get top 3 similar customers
        similar_customers.sort(key=lambda x: x[1], reverse=True)
        
        for similar_customer_id, similarity in similar_customers[:3]:
            similar_customer_purchases = self.customer_product_matrix.loc[similar_customer_id]
            liked_products = similar_customer_purchases[similar_customer_purchases > 0].index
            
            for product in liked_products:
                if product in not_purchased:
                    recommendations.append((product, similarity))
        
        # Method 3: Frequent patterns
        customer_products = set(customer_purchases[customer_purchases > 0].index)
        
        for (item1, item2), count in self.frequent_patterns.items():
            if count > 2:  # Minimum threshold
                if item1 in customer_products and item2 not in customer_products:
                    recommendations.append((item2, count * 0.1))
                elif item2 in customer_products and item1 not in customer_products:
                    recommendations.append((item1, count * 0.1))
        
        # Remove duplicates and sort by score
        unique_recommendations = {}
        for product, score in recommendations:
            if product in unique_recommendations:
                unique_recommendations[product] += score
            else:
                unique_recommendations[product] = score
        
        # Sort by score and return top N
        sorted_recommendations = sorted(unique_recommendations.items(), 
                                      key=lambda x: x[1], reverse=True)
        
        return [product for product, score in sorted_recommendations[:top_n]]
    
    def get_customer_insights(self, customer_id, purchase_data):
        """Get insights about customer's purchase behavior"""
        if customer_id not in self.customer_ids:
            return {"error": "Customer not found"}
        
        customer_purchases = purchase_data[purchase_data['customer_id'] == customer_id]
        
        insights = {
            'total_purchases': len(customer_purchases),
            'unique_products': customer_purchases['product_name'].nunique(),
            'favorite_category': customer_purchases['category'].mode().iloc[0] if not customer_purchases.empty else 'None',
            'avg_quantity': customer_purchases['quantity'].mean(),
            'purchase_frequency_days': self._calculate_purchase_frequency(customer_purchases)
        }
        
        return insights
    
    def _calculate_purchase_frequency(self, customer_purchases):
        """Calculate average days between purchases"""
        if len(customer_purchases) < 2:
            return "Insufficient data"
        
        customer_purchases = customer_purchases.sort_values('purchase_date')
        time_diffs = customer_purchases['purchase_date'].diff().dt.days.dropna()
        
        return f"{time_diffs.mean():.1f} days"
