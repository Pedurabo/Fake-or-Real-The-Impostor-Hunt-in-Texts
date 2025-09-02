#!/usr/bin/env python3
"""
Production Deployment Script for Phase 7
Deploys the optimized model as a production API
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def deploy_production():
    """Deploy the production pipeline"""
    try:
        logging.info("Starting Phase 7 Production Deployment")
        
        # Import production pipeline
        from src.modules.production_pipeline import ProductionPipeline
        
        # Initialize production pipeline
        pipeline = ProductionPipeline(data_path="src/temp_data/data")
        
        # Load optimized models
        if not pipeline.load_optimized_models():
            raise Exception("Failed to load optimized models")
        
        # Create production API
        if not pipeline.create_production_api():
            raise Exception("Failed to create production API")
        
        # Start the API server
        logging.info("Starting production API server...")
        pipeline.api_app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
        
    except Exception as e:
        logging.error(f"Production deployment failed: {e}")
        raise

if __name__ == "__main__":
    deploy_production()
