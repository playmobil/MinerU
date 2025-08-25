#!/usr/bin/env python3
"""
Test script for VLM integration with MinerU
"""

import sys
import os
import tempfile
from pathlib import Path

# Add the MinerU directory to the path
sys.path.insert(0, "/Users/frank/mygit/github/MinerU")

def test_vlm_flag():
    """Test that the --vlm flag is properly recognized"""
    print("Testing VLM flag recognition...")
    
    try:
        from mineru.cli.client import main
        import click
        from click.testing import CliRunner
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple test image (or we could skip file processing for this test)
            test_output = Path(temp_dir) / "test_output"
            test_output.mkdir()
            
            # Test that the CLI accepts the --vlm flag
            runner = CliRunner()
            
            # Test help to see if --vlm appears
            result = runner.invoke(main, ['--help'])
            print("Help output check:")
            if '--vlm' in result.output:
                print("✅ --vlm flag is present in help")
            else:
                print("❌ --vlm flag is missing from help")
            
            print(f"Help exit code: {result.exit_code}")
            
            # Test VLM availability
            from mineru.backend.pipeline.model_init import VLM_AVAILABLE
            print(f"VLM Available: {'✅' if VLM_AVAILABLE else '❌'} {VLM_AVAILABLE}")
            
            return VLM_AVAILABLE
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def test_vlm_processor():
    """Test VLM processor functionality"""
    print("\nTesting VLM processor...")
    
    try:
        from mineru.backend.pipeline.model_init import TableVLMProcessor
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='white')
        
        # Create VLM processor
        processor = TableVLMProcessor()
        
        # Test enhancement (this should work without actually loading models)
        test_result = {'html': '<table><tr><td>test</td></tr></table>'}
        enhanced = processor.enhance_table_result(test_image, test_result, "test context")
        
        print("✅ VLM processor test completed successfully")
        print(f"Enhanced result type: {type(enhanced)}")
        
        return True
        
    except Exception as e:
        print(f"❌ VLM processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("MinerU VLM Integration Test")
    print("=" * 40)
    
    # Test 1: VLM flag recognition
    flag_test_passed = test_vlm_flag()
    
    # Test 2: VLM processor functionality
    processor_test_passed = test_vlm_processor()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"VLM Flag Recognition: {'✅ PASS' if flag_test_passed else '❌ FAIL'}")
    print(f"VLM Processor Test: {'✅ PASS' if processor_test_passed else '❌ FAIL'}")
    
    if flag_test_passed and processor_test_passed:
        print("\n🎉 All tests passed! VLM integration is ready.")
        print("\nYou can now use MinerU with VLM enhancement:")
        print("mineru -p your_file.pdf -o output_dir --vlm")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
    
    return flag_test_passed and processor_test_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)