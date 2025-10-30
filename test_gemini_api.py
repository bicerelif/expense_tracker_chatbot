"""
Test script to diagnose Gemini API connection issues
"""

import os
import sys

# Suppress warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

def test_gemini_connection():
    """Test connection to Gemini API and show detailed diagnostics"""
    
    print("=" * 60)
    print("GEMINI API CONNECTION TEST")
    print("=" * 60)
    
    # Step 1: Check API key
    print("\n1. Checking API Key...")
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("   ❌ GEMINI_API_KEY environment variable not set!")
        print("\n   To fix this:")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        return False
    
    print(f"   ✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Step 2: Test import
    print("\n2. Testing google.generativeai import...")
    try:
        import google.generativeai as genai
        print("   ✅ Module imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import: {e}")
        print("\n   To fix this:")
        print("   pip install google-generativeai")
        return False
    
    # Step 3: Configure API
    print("\n3. Configuring Gemini API...")
    try:
        genai.configure(api_key=api_key)
        print("   ✅ API configured")
    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")
        return False
    
    # Step 4: List available models
    print("\n4. Fetching available models...")
    try:
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(m.name)
                print(f"   ✅ Found: {m.name}")
        
        if not models:
            print("   ⚠️  No models with 'generateContent' found")
            print("\n   All available models:")
            for m in genai.list_models():
                print(f"      - {m.name}: {m.supported_generation_methods}")
        else:
            print(f"\n   ✅ Total models available: {len(models)}")
            
    except Exception as e:
        print(f"   ❌ Failed to list models: {e}")
        print(f"\n   Error type: {type(e).__name__}")
        print(f"   Error details: {str(e)}")
        
        if "API key not valid" in str(e):
            print("\n   🔧 SOLUTION: Your API key appears to be invalid")
            print("      - Get a new key from: https://makersuite.google.com/app/apikey")
            print("      - Make sure you're using a Gemini API key, not a different Google API key")
        elif "403" in str(e):
            print("\n   🔧 SOLUTION: Permission denied")
            print("      - Your API key may not have access to Gemini models")
            print("      - Verify your API key at: https://makersuite.google.com/app/apikey")
        elif "timeout" in str(e).lower() or "connection" in str(e).lower():
            print("\n   🔧 SOLUTION: Network connection issue")
            print("      - Check your internet connection")
            print("      - Check if you're behind a firewall/proxy")
            print("      - Try: curl https://generativelanguage.googleapis.com/")
        
        return False
    
    # Step 5: Test simple generation
    print("\n5. Testing content generation with a simple prompt...")
    try:
        # Use Flash models which have better free tier limits
        preferred_models = [
            'models/gemini-2.5-flash',
            'models/gemini-2.0-flash', 
            'models/gemini-flash-latest',
            'models/gemini-2.5-flash-lite',
        ]
        
        model_name = None
        for pref in preferred_models:
            if pref in models:
                model_name = pref
                break
        
        if not model_name:
            model_name = models[0] if models else "gemini-pro"
            
        print(f"   Using model: {model_name}")
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'Hello, API test successful!'")
        
        print(f"   ✅ Response received: {response.text[:50]}...")
        
    except Exception as e:
        error_str = str(e)
        print(f"   ❌ Generation failed: {error_str[:200]}...")
        print(f"   Error type: {type(e).__name__}")
        
        if "429" in error_str or "quota" in error_str.lower():
            print("\n   🔧 SOLUTION: Quota exceeded")
            print("      - You've hit your free tier limit for this model")
            print("      - Wait 12-24 hours for quota to reset")
            print("      - Or try upgrading to a paid plan")
            print("      - The receipt processing will use Flash models with higher limits")
        
        return False
    
    # Step 6: Test vision capability
    print("\n6. Testing vision model (receipt processing capability)...")
    vision_models = [m for m in models if 'vision' in m.lower() or 'flash' in m.lower() or '1.5' in m or '2.0' in m]
    
    if not vision_models:
        print("   ⚠️  No vision-capable models found")
        print("   Models available:", models[:3])
    else:
        print(f"   ✅ Vision-capable models found:")
        for vm in vision_models[:3]:
            print(f"      - {vm}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYour Gemini API is working correctly.")
    print("The receipt processing should work now.")
    return True


if __name__ == "__main__":
    try:
        success = test_gemini_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)