import unittest
import torch
from flcore.models.basic import HARSModel

class TestHARSModel(unittest.TestCase):

    def test_import_export(self):
        model = HARSModel(device='cpu')
        bin1: bytes = model.export_binary()

        model2 = HARSModel(device='cpu')
        model2.import_binary(bin1)
        #bin2 = model2.export_binary()

        state_dict1 = model.state_dict()
        state_dict2 = model2.state_dict()
        for key in state_dict1.keys():
            self.assertTrue(
                torch.equal(state_dict1[key], state_dict2[key]),
                f"Mismatch found in key '{key}'"
        )


if __name__ == "__main__":
    unittest.main()