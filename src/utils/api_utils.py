def read_token(token_path):
    """
    Reads the token from a file.

    Args:
        token_path (str): Path to the file containing the token.

    Returns:
        str: The token read from the file.
    """
    with open(token_path, 'r') as file:
        token = file.read().strip()
    return token