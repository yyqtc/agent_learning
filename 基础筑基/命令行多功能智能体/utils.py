import os
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization


def generate_ed25519_keypair():
    """生成Ed25519私钥和公钥对"""
    # 生成私钥
    private_key = ed25519.Ed25519PrivateKey.generate()
    
    # 从私钥生成公钥
    public_key = private_key.public_key()
    
    # 将私钥序列化为PEM格式
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # 将公钥序列化为PEM格式
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return private_pem.decode('utf-8'), public_pem.decode('utf-8')
    """从私钥生成对应的公钥"""
    # 加载私钥
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode('utf-8'),
        password=None
    )
    
    # 生成公钥
    public_key = private_key.public_key()
    
    # 序列化公钥
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return public_pem.decode('utf-8')


def init_jwt_token() -> str:
    """使用Ed25519算法进行签名"""
    import jwt
    import time
    import sys

    private_key = ""

    payload  = {
        'iat': int(time.time()) - 30,
        'exp': int(time.time()) + 900,
    }


if __name__ == "__main__":
    private_key, public_key = generate_ed25519_keypair()
    print(private_key)
    print(public_key)