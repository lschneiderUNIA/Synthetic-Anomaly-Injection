import pytest 

from test_modules.test_generated_data import TestGeneratedData

def test_all():
    TestGeneratedData.test_generated_dataset()


# Run all tests with pytest
if __name__ == "__main__":
    pytest.main()