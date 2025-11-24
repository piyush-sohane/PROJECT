import pandas as pd
from data_loader import DataLoader
from recommendation_engine import SimpleRecommendationEngine

def main():
    print("️ AI Purchase Recommendation System")
    print("=" * 50)
    
    # Initialize components
    data_loader = DataLoader()
    recommender = SimpleRecommendationEngine()
    
    # Generate sample data
    print("\n Generating sample purchase data...")
    purchase_data = data_loader.generate_sample_data()
    print(f"Generated {len(purchase_data)} purchase records for 50 customers")
    
    # Train the recommendation engine
    recommender.fit(purchase_data)
    
    while True:
        print("\n" + "=" * 50)
        print("MENU:")
        print("1. Get recommendations for customer")
        print("2. View customer insights")
        print("3. Show sample data")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            try:
                customer_id = int(input("Enter customer ID (1-50): "))
                recommendations = recommender.recommend_products(customer_id)
                
                print(f"\n Recommendations for Customer {customer_id}:")
                for i, product in enumerate(recommendations, 1):
                    print(f"  {i}. {product}")
                    
            except ValueError:
                print(" Please enter a valid customer ID!")
        
        elif choice == '2':
            try:
                customer_id = int(input("Enter customer ID (1-50): "))
                insights = recommender.get_customer_insights(customer_id, purchase_data)
            
                print(f"\n Insights for Customer {customer_id}:")
                for key, value in insights.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                    
            except ValueError:
                print(" Please enter a valid customer ID!")
        
        elif choice == '3':
            print(f"\n Sample Purchase Data (first 10 records):")
            print(purchase_data[['customer_id', 'product_name', 'category', 'quantity']].head(10).to_string(index=False))
            
            print(f"\n Available Products:")
            for pid, name in data_loader.products.items():
                print(f"  {pid}. {name} ({data_loader.categories[name]})")
        
        elif choice == '4':
            print(" Thank you for using the AI Recommendation System!")
            break
        
        else:
            print(" Invalid choice! Please enter 1-4.")

if __name__ == "__main__":
    main()
