import unittest
import key_generation as kg


class TestKeyGen(unittest.TestCase):

    def test_keygen(self):
        key_path = "/home/marcuswrrn/Projects/Federated-Learning/src/client/instance/client_hash.txt"
        generated_key = kg.get_key(key_path)
    
    def test_randomness(self):
        gen_key1 = kg.generate_random_key()
        gen_key2 = kg.generate_random_key()

        assert gen_key1 != gen_key2


if __name__ == "__main__":
    unittest.main()