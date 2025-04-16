from unittest.mock import AsyncMock, MagicMock
    

def get_async_magic_mock():
    async_mock = AsyncMock()
    magic_mock = MagicMock()

    async_mock.return_value = magic_mock
    
    return async_mock
