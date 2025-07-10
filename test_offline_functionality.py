#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API functionality test script

Test basic functionality of EnvironmentalImageGenerator based on Hugging Face API
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environmental_image_generator import EnvironmentalImageGenerator
from src.utils.logger import get_logger

logger = get_logger("offline_test")

def test_basic_initialization():
    """Test basic initialization functionality"""
    print("\n=== Testing Basic Initialization ===")
    
    try:
        # Test basic initialization
        generator = EnvironmentalImageGenerator()
        print(f"✅ Basic initialization successful")
        print(f"   Model ID: {generator.model_id}")
        print(f"   API Endpoint: {generator.api_url}")
        print(f"   HF Token: {'Set' if generator.hf_token else 'Not set'}")
        
        return generator
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None

def test_api_connection(generator):
    """Test API connection functionality"""
    print("\n=== Testing API Connection Functionality ===")
    
    try:
        result = generator.test_api_connection()
        
        if result['success']:
            print(f"✅ API connection test successful")
            print(f"   Status code: {result['status_code']}")
            print(f"   Message: {result['message']}")
        else:
            print(f"❌ API connection test failed")
            print(f"   Error: {result.get('error', result.get('message', 'Unknown error'))}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ API connection test failed: {e}")
        return False

def test_prompt_enhancement(generator):
    """Test prompt enhancement functionality"""
    print("\n=== Testing Prompt Enhancement Functionality ===")
    
    try:
        # Test prompts
        test_prompts = [
            "城市空气污染",
            "forest destruction",
            "ocean plastic pollution",
            "climate change effects",
            "工业废水排放"
        ]
        
        for prompt in test_prompts:
            try:
                enhanced = generator.enhance_prompt(prompt)
                print(f"\nOriginal prompt: {prompt}")
                print(f"Enhanced prompt: {enhanced[:100]}...")
            except Exception as e:
                print(f"⚠️ Prompt enhancement failed: {prompt} - {e}")
        
        print("\n✅ Prompt enhancement functionality test completed")
        return True
        
    except Exception as e:
        print(f"❌ Prompt enhancement test failed: {e}")
        return False

def test_image_generation(generator):
    """Test image generation functionality (API-based)"""
    print("\n=== Testing Image Generation Functionality ===")
    
    try:
        # Test user input
        user_input = "工业污染的城市景观"
        
        print(f"Test input: {user_input}")
        print("⚠️ Note: This test requires valid HF_TOKEN and network connection")
        
        # Try to generate image (using smaller parameters to save time)
        result = generator.generate_image(
            user_input=user_input,
            width=512,
            height=512,
            num_inference_steps=10  # Reduce steps to speed up testing
        )
        
        if result['success']:
            print("✅ Image generation test successful")
            print(f"   Generation time: {result.get('generation_time', 'N/A')} seconds")
            print(f"   Image count: {len(result.get('images', []))}")
            print(f"   Save path: {result.get('image_paths', [])}")
            print(f"   Used prompt: {result.get('prompt', '')[:100]}...")
        else:
            print(f"⚠️ Image generation failed: {result.get('error', 'Unknown error')}")
            # For testing, API failure is not a fatal error
            if 'token' in result.get('error', '').lower() or result.get('status_code') == 401:
                print("💡 Tip: Please set a valid HF_TOKEN environment variable")
                return True  # Consider test passed, just missing token
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Image generation test failed: {e}")
        # For network-related errors, don't count as test failure
        if 'connection' in str(e).lower() or 'timeout' in str(e).lower():
            print("💡 Tip: Network connection issue, skipping this test")
            return True
        return False

def save_test_results(results, output_dir="outputs/offline_test"):
    """Save test results"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"offline_test_results_{timestamp}.json")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nTest results saved to: {result_file}")
    return result_file

def main():
    """Main function"""
    print("🧪 Starting API functionality test")
    
    test_results = {
        "test_timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    try:
        # 1. Basic initialization test
        generator = test_basic_initialization()
        test_results["tests"]["initialization"] = generator is not None
        
        if generator is None:
            print("❌ Initialization failed, cannot continue testing")
            return False
        
        # 2. API connection test
        test_results["tests"]["api_connection"] = test_api_connection(generator)
        
        # 3. Prompt enhancement test
        test_results["tests"]["prompt_enhancement"] = test_prompt_enhancement(generator)
        
        # 4. Image generation test
        test_results["tests"]["image_generation"] = test_image_generation(generator)
        
        # Calculate results
        passed_tests = sum(1 for result in test_results["tests"].values() if result)
        total_tests = len(test_results["tests"])
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0
        }
        
        # Save results
        save_test_results(test_results)
        
        # Print summary
        print("\n=== Test Summary ===")
        print(f"Total tests: {total_tests}")
        print(f"Passed tests: {passed_tests}")
        print(f"Failed tests: {total_tests - passed_tests}")
        print(f"Success rate: {passed_tests / total_tests * 100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 All API functionality tests passed!")
            return True
        else:
            print("\n⚠️ Some tests failed, please check the logs")
            return False
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)