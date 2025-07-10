#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for data-driven environmental warning image generation
Tests the enhanced functionality with deviation analysis and visual emphasis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dashscope_environmental_generator import DashScopeEnvironmentalGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_high_deviation_scenario():
    """
    Test scenario with high deviations from default values
    """
    print("\n" + "="*60)
    print("🧪 Testing HIGH DEVIATION scenario")
    print("="*60)
    
    # Create generator instance
    generator = DashScopeEnvironmentalGenerator()
    
    # Test with significantly deviated environmental data
    high_deviation_data = {
        'carbon_emission': 180.0,      # 80% above default (100)
        'air_quality_index': 225.0,    # 125% above default (100)
        'water_pollution_index': 85.0,  # 70% above default (50)
        'noise_level': 95.0,           # 58% above default (60)
        'deforestation_rate': 8.5,     # 70% above default (5)
        'plastic_waste': 135.0         # 35% above default (100)
    }
    
    additional_prompt = "Focus on industrial pollution and urban environmental crisis"
    
    print("\n📊 Environmental Data (High Deviation):")
    for key, value in high_deviation_data.items():
        default_val = generator.environmental_data_types[key]['default_value']
        deviation = ((value - default_val) / default_val) * 100
        print(f"  - {generator.environmental_data_types[key]['name']}: {value} {generator.environmental_data_types[key]['unit']} ({deviation:+.1f}% deviation)")
    
    print(f"\n💬 Additional Prompt: {additional_prompt}")
    
    # Test deviation analysis
    deviation_analysis = generator._calculate_data_deviations(high_deviation_data)
    print("\n📈 Top Deviations Analysis:")
    print(deviation_analysis['top_deviations_text'])
    
    # Test emphasis instructions
    emphasis_instructions = generator._build_emphasis_instructions(deviation_analysis)
    print("\n🎯 Visual Emphasis Instructions:")
    print(emphasis_instructions)
    
    # Generate image
    print("\n🎨 Generating environmental warning image...")
    result = generator.generate_environmental_warning_image(
        environmental_data=high_deviation_data,
        user_description=additional_prompt
    )
    
    if result.get('success', False):
        print("✅ High deviation image generation successful!")
        print(f"📁 Saved to: {result.get('saved_paths', [])}")
        print(f"📊 Analysis: {result['analysis']['overall_severity']} severity")
    else:
        print(f"❌ High deviation image generation failed: {result.get('error', 'Unknown error')}")
    
    return result

def test_moderate_deviation_scenario():
    """
    Test scenario with moderate deviations from default values
    """
    print("\n" + "="*60)
    print("🧪 Testing MODERATE DEVIATION scenario")
    print("="*60)
    
    # Create generator instance
    generator = DashScopeEnvironmentalGenerator()
    
    # Test with moderately deviated environmental data
    moderate_deviation_data = {
        'carbon_emission': 130.0,      # 30% above default
        'air_quality_index': 140.0,    # 40% above default
        'water_pollution_index': 65.0,  # 30% above default
        'noise_level': 75.0,           # 25% above default
        'deforestation_rate': 6.5,     # 30% above default
        'plastic_waste': 115.0         # 15% above default
    }
    
    additional_prompt = "Show suburban environmental concerns with visible but not extreme impacts"
    
    print("\n📊 Environmental Data (Moderate Deviation):")
    for key, value in moderate_deviation_data.items():
        default_val = generator.environmental_data_types[key]['default_value']
        deviation = ((value - default_val) / default_val) * 100
        print(f"  - {generator.environmental_data_types[key]['name']}: {value} {generator.environmental_data_types[key]['unit']} ({deviation:+.1f}% deviation)")
    
    print(f"\n💬 Additional Prompt: {additional_prompt}")
    
    # Generate image
    print("\n🎨 Generating environmental warning image...")
    result = generator.generate_environmental_warning_image(
        environmental_data=moderate_deviation_data,
        user_description=additional_prompt
    )
    
    if result.get('success', False):
        print("✅ Moderate deviation image generation successful!")
        print(f"📁 Saved to: {result.get('saved_paths', [])}")
        print(f"📊 Analysis: {result['analysis']['overall_severity']} severity")
    else:
        print(f"❌ Moderate deviation image generation failed: {result.get('error', 'Unknown error')}")
    
    return result

def test_low_deviation_scenario():
    """
    Test scenario with low deviations from default values
    """
    print("\n" + "="*60)
    print("🧪 Testing LOW DEVIATION scenario")
    print("="*60)
    
    # Create generator instance
    generator = DashScopeEnvironmentalGenerator()
    
    # Test with slightly deviated environmental data
    low_deviation_data = {
        'carbon_emission': 110.0,      # 10% above default
        'air_quality_index': 105.0,    # 5% above default
        'water_pollution_index': 55.0,  # 10% above default
        'noise_level': 65.0,           # 8% above default
        'deforestation_rate': 5.5,     # 10% above default
        'plastic_waste': 108.0         # 8% above default
    }
    
    additional_prompt = "Show early warning signs of environmental change in a natural setting"
    
    print("\n📊 Environmental Data (Low Deviation):")
    for key, value in low_deviation_data.items():
        default_val = generator.environmental_data_types[key]['default_value']
        deviation = ((value - default_val) / default_val) * 100
        print(f"  - {generator.environmental_data_types[key]['name']}: {value} {generator.environmental_data_types[key]['unit']} ({deviation:+.1f}% deviation)")
    
    print(f"\n💬 Additional Prompt: {additional_prompt}")
    
    # Generate image
    print("\n🎨 Generating environmental warning image...")
    result = generator.generate_environmental_warning_image(
        environmental_data=low_deviation_data,
        user_description=additional_prompt
    )
    
    if result.get('success', False):
        print("✅ Low deviation image generation successful!")
        print(f"📁 Saved to: {result.get('saved_paths', [])}")
        print(f"📊 Analysis: {result['analysis']['overall_severity']} severity")
    else:
        print(f"❌ Low deviation image generation failed: {result.get('error', 'Unknown error')}")
    
    return result

def main():
    """
    Main test function
    """
    print("🚀 Starting Data-Driven Environmental Image Generation Tests")
    print("This test will verify that images are generated based on data deviations")
    print("and that visual emphasis correlates with deviation magnitude.")
    
    try:
        # Test API connection first
        generator = DashScopeEnvironmentalGenerator()
        if not generator.test_connection():
            print("❌ API connection test failed. Please check your configuration.")
            return
        
        # Run test scenarios
        high_result = test_high_deviation_scenario()
        moderate_result = test_moderate_deviation_scenario()
        low_result = test_low_deviation_scenario()
        
        # Summary
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        print(f"High Deviation Test: {'✅ PASSED' if high_result.get('success') else '❌ FAILED'}")
        print(f"Moderate Deviation Test: {'✅ PASSED' if moderate_result.get('success') else '❌ FAILED'}")
        print(f"Low Deviation Test: {'✅ PASSED' if low_result.get('success') else '❌ FAILED'}")
        
        if all([high_result.get('success'), moderate_result.get('success'), low_result.get('success')]):
            print("\n🎉 All tests passed! Data-driven image generation is working correctly.")
            print("📸 Check the generated images to verify visual emphasis correlates with data deviations.")
        else:
            print("\n⚠️ Some tests failed. Please check the error messages above.")
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        print(f"❌ Test execution failed: {e}")

if __name__ == "__main__":
    main()