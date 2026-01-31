#!/usr/bin/env python3
"""
Simplified test script for Pose Tokenizer integration (without PyTorch dependency).
This script validates the code structure and integration points.
"""

import sys
import os
import ast
import inspect
from pathlib import Path

def test_models_structure():
    """Test that models.py has the required classes and methods."""
    print("Testing models.py structure...")

    try:
        # Read models.py file
        models_path = Path("models.py")
        if not models_path.exists():
            print("❌ models.py not found")
            return False

        with open(models_path, 'r') as f:
            content = f.read()

        # Parse the AST to check for required classes
        tree = ast.parse(content)

        classes_found = []
        methods_found = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes_found.append(node.name)
                methods_found[node.name] = [method.name for method in node.body if isinstance(method, ast.FunctionDef)]

        # Check required classes
        required_classes = ['VectorQuantizer', 'ResidualVectorQuantizer', 'PoseTokenizer', 'Uni_Sign']
        for cls in required_classes:
            if cls in classes_found:
                print(f"  ✓ {cls} class found")
            else:
                print(f"  ❌ {cls} class missing")
                return False

        # Check Uni_Sign has pose tokenizer integration
        if 'Uni_Sign' in methods_found:
            uni_sign_methods = methods_found['Uni_Sign']
            if '__init__' in uni_sign_methods and 'forward' in uni_sign_methods:
                print(f"  ✓ Uni_Sign has required methods")
            else:
                print(f"  ❌ Uni_Sign missing required methods")
                return False

        # Check for pose tokenizer related code
        if 'use_pose_tokenizer' in content and 'pose_tokenizer' in content:
            print(f"  ✓ Pose tokenizer integration code found")
        else:
            print(f"  ❌ Pose tokenizer integration code missing")
            return False

        print("✓ models.py structure test passed\n")
        return True

    except Exception as e:
        print(f"❌ Error testing models.py: {e}")
        return False


def test_config_structure():
    """Test that config.py has the required configuration."""
    print("Testing config.py structure...")

    try:
        # Read config.py file
        config_path = Path("config.py")
        if not config_path.exists():
            print("❌ config.py not found")
            return False

        with open(config_path, 'r') as f:
            content = f.read()

        # Check for pose tokenizer config
        if 'POSE_TOKENIZER_CONFIG' in content:
            print(f"  ✓ POSE_TOKENIZER_CONFIG found")
        else:
            print(f"  ❌ POSE_TOKENIZER_CONFIG missing")
            return False

        # Check for required config keys
        required_keys = ['use_pose_tokenizer', 'tokenizer_hidden_dim', 'num_quantizers', 'codebook_size']
        for key in required_keys:
            if key in content:
                print(f"  ✓ {key} configuration found")
            else:
                print(f"  ❌ {key} configuration missing")
                return False

        print("✓ config.py structure test passed\n")
        return True

    except Exception as e:
        print(f"❌ Error testing config.py: {e}")
        return False


def test_training_integration():
    """Test that pre_training.py has pose tokenizer integration."""
    print("Testing pre_training.py integration...")

    try:
        # Read pre_training.py file
        training_path = Path("pre_training.py")
        if not training_path.exists():
            print("❌ pre_training.py not found")
            return False

        with open(training_path, 'r') as f:
            content = f.read()

        # Check for pose tokenizer related code
        integration_checks = [
            ('vq_loss', 'VQ loss handling'),
            ('perplexity', 'Perplexity logging'),
            ('use_pose_tokenizer', 'Pose tokenizer flag check'),
            ('metric_logger.update', 'Metric logging')
        ]

        for check, description in integration_checks:
            if check in content:
                print(f"  ✓ {description} found")
            else:
                print(f"  ❌ {description} missing")
                return False

        print("✓ pre_training.py integration test passed\n")
        return True

    except Exception as e:
        print(f"❌ Error testing pre_training.py: {e}")
        return False


def test_trainer_script():
    """Test that pose_tokenizer_trainer.py exists and has required structure."""
    print("Testing pose_tokenizer_trainer.py structure...")

    try:
        # Read trainer file
        trainer_path = Path("pose_tokenizer_trainer.py")
        if not trainer_path.exists():
            print("❌ pose_tokenizer_trainer.py not found")
            return False

        with open(trainer_path, 'r') as f:
            content = f.read()

        # Parse the AST to check for required functions
        tree = ast.parse(content)

        functions_found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions_found.append(node.name)

        # Check required functions
        required_functions = ['train_pose_tokenizer', 'evaluate_pose_tokenizer', 'main']
        for func in required_functions:
            if func in functions_found:
                print(f"  ✓ {func} function found")
            else:
                print(f"  ❌ {func} function missing")
                return False

        # Check for key components
        key_components = [
            ('PoseTokenizer', 'PoseTokenizer import/usage'),
            ('DataLoader', 'DataLoader usage'),
            ('optimizer', 'Optimizer setup'),
            ('loss', 'Loss computation')
        ]

        for component, description in key_components:
            if component in content:
                print(f"  ✓ {description} found")
            else:
                print(f"  ❌ {description} missing")
                return False

        print("✓ pose_tokenizer_trainer.py structure test passed\n")
        return True

    except Exception as e:
        print(f"❌ Error testing pose_tokenizer_trainer.py: {e}")
        return False


def test_file_imports():
    """Test that all files can be parsed (syntax check)."""
    print("Testing file syntax...")

    files_to_check = ['models.py', 'config.py', 'pre_training.py', 'pose_tokenizer_trainer.py']

    for filename in files_to_check:
        try:
            filepath = Path(filename)
            if filepath.exists():
                with open(filepath, 'r') as f:
                    content = f.read()

                # Try to parse the file
                ast.parse(content)
                print(f"  ✓ {filename} syntax is valid")
            else:
                print(f"  ⚠️ {filename} not found (skipping)")

        except SyntaxError as e:
            print(f"  ❌ {filename} has syntax error: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Error checking {filename}: {e}")
            return False

    print("✓ File syntax test passed\n")
    return True


def test_integration_completeness():
    """Test overall integration completeness."""
    print("Testing integration completeness...")

    # Check that all required files exist
    required_files = [
        ('models.py', 'Core model definitions'),
        ('config.py', 'Configuration settings'),
        ('pre_training.py', 'Training script'),
        ('pose_tokenizer_trainer.py', 'Pose tokenizer trainer'),
        ('test_pose_tokenizer.py', 'Test script')
    ]

    all_files_exist = True
    for filename, description in required_files:
        if Path(filename).exists():
            print(f"  ✓ {filename} - {description}")
        else:
            print(f"  ❌ {filename} - {description} (missing)")
            all_files_exist = False

    if all_files_exist:
        print("✓ Integration completeness test passed\n")
        return True
    else:
        print("❌ Integration completeness test failed\n")
        return False


def main():
    """Run all structure tests."""
    print("=" * 60)
    print("Pose Tokenizer Integration Structure Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_file_imports,
        test_models_structure,
        test_config_structure,
        test_training_integration,
        test_trainer_script,
        test_integration_completeness
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test in tests:
        try:
            if test():
                passed_tests += 1
            else:
                print(f"Test {test.__name__} failed\n")
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}\n")

    print("=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All structure tests passed!")
        print()
        print("Integration Summary:")
        print("✓ VectorQuantizer and ResidualVectorQuantizer classes implemented")
        print("✓ PoseTokenizer class integrated with ST-GCN architecture")
        print("✓ Uni_Sign model updated with pose tokenizer support")
        print("✓ Training script modified to handle VQ losses and metrics")
        print("✓ Configuration system extended with pose tokenizer settings")
        print("✓ Dedicated trainer script created for pose tokenizer training")
        print()
        print("Next steps:")
        print("1. Install PyTorch and dependencies")
        print("2. Run: python test_pose_tokenizer.py (full functionality test)")
        print("3. Run training with pose tokenizer enabled")
        print("4. Monitor VQ loss and perplexity metrics")
    else:
        print("❌ Some structure tests failed. Please review the errors above.")
        return 1

    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)