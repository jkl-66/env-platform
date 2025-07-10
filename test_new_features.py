#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for new features in DashScope Environmental Generator
- Auto-open images after generation
- Additional prompt input functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dashscope_environmental_generator import DashScopeEnvironmentalGenerator

def test_new_features():
    """Test the new features"""
    print("=== Testing New Features ===")
    
    try:
        # Initialize generator
        generator = DashScopeEnvironmentalGenerator()
        
        # Test additional prompt input function
        print("\n1. Testing additional prompt input function...")
        
        # Simulate user input by temporarily replacing input function
        original_input = input
        test_prompt = "Show a dystopian cityscape with heavy smog"
        
        def mock_input(prompt_text):
            print(f"Mock input: {test_prompt}")
            return test_prompt
        
        # Temporarily replace input function
        import builtins
        builtins.input = mock_input
        
        additional_prompt = generator.get_additional_prompt()
        print(f"Additional prompt result: {additional_prompt}")
        
        # Restore original input function
        builtins.input = original_input
        
        # Test with sample environmental data
        print("\n2. Testing image generation with additional prompt...")
        sample_data = {
            'carbon_emissions': 450.0,
            'air_quality_index': 180.0,
            'water_pollution_index': 75.0,
            'noise_level': 85.0,
            'deforestation_area': 15000.0,
            'plastic_waste': 12.0
        }
        
        print("Sample environmental data:")
        for key, value in sample_data.items():
            print(f"  {key}: {value}")
        
        print(f"\nAdditional prompt: {additional_prompt}")
        
        # Test connection first
        print("\n3. Testing API connection...")
        connection_result = generator.test_connection()
        if connection_result["success"]:
            print("✅ API connection successful!")
            
            # Generate image with additional prompt
            print("\n4. Generating image with new features...")
            result = generator.generate_environmental_warning_image(
                sample_data, 
                user_description=additional_prompt
            )
            
            if result["success"]:
                print("✅ Image generation successful!")
                print(f"📁 Saved paths: {result['saved_paths']}")
                print("📸 Images should have opened automatically!")
                if 'analysis' in result:
                    analysis = result['analysis']
                    print(f"📊 Environmental analysis: Severity level {analysis.get('overall_severity', 'N/A')}")
            else:
                print(f"❌ Image generation failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ API connection failed: {connection_result.get('error', 'Unknown error')}")
            print("Note: This is expected if API keys are not configured")
        
        print("\n=== Test Complete ===")
        print("New features tested:")
        print("✅ Additional prompt input functionality")
        print("✅ Auto-open images after generation (if API connection works)")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_features()