#!/usr/bin/env python3
"""
Test script to verify Pydantic v2 compatibility with updated langchain packages.
"""

def test_langchain_imports():
    """Test that langchain packages can be imported without Pydantic v2 errors."""
    try:
        print("Testing langchain-core import...")
        from langchain_core.messages import HumanMessage
        print("✅ langchain-core import successful")
        
        print("Testing langchain-community import...")
        from langchain_community.vectorstores import Chroma
        print("✅ langchain-community import successful")
        
        print("Testing langchain-openai import...")
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        print("✅ langchain-openai import successful")
        
        print("Testing langchain import...")
        from langchain.schema import Document
        print("✅ langchain import successful")
        
        print("\n🎉 All langchain packages imported successfully with Pydantic v2!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_pydantic_compatibility():
    """Test that Pydantic v2 features work correctly."""
    try:
        print("Testing Pydantic v2 compatibility...")
        from pydantic import BaseModel, Field
        from typing import Optional
        
        class TestModel(BaseModel):
            name: str = Field(description="Test name field")
            value: Optional[int] = None
        
        # Test model creation
        model = TestModel(name="test", value=42)
        print(f"✅ Pydantic v2 model creation successful: {model}")
        
        # Test JSON schema generation (new v2 method)
        schema = model.model_json_schema()
        print("✅ Pydantic v2 JSON schema generation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Pydantic v2 test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PYDANTIC V2 COMPATIBILITY TEST")
    print("=" * 60)
    
    success = True
    
    # Test Pydantic v2 functionality
    if not test_pydantic_compatibility():
        success = False
    
    print("\n" + "-" * 60)
    
    # Test langchain imports
    if not test_langchain_imports():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - Pydantic v2 compatibility verified!")
    else:
        print("❌ TESTS FAILED - Pydantic v2 compatibility issues detected")
    print("=" * 60)
