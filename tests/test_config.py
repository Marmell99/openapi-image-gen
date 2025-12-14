import os
from unittest.mock import patch


def test_image_base_url_from_env():
    """
    Test IMAGE_BASE_URL is loaded from environment.
    """
    with patch.dict(
        os.environ,
        {
            "IMAGE_BASE_URL": "http://image-api:8000",
        },
        clear=True,
    ):
        from importlib import reload

        import app.core.config

        reload(app.core.config)
        settings = app.core.config.Settings()

        assert settings.IMAGE_BASE_URL == "http://image-api:8000"


def test_image_base_url_default():
    """
    Test default IMAGE_BASE_URL when not set.
    """
    with patch.dict(os.environ, {}, clear=True):
        from importlib import reload

        import app.core.config

        reload(app.core.config)
        settings = app.core.config.Settings()

        assert settings.IMAGE_BASE_URL == "http://localhost:8000"


def test_default_model_config():
    """
    Test DEFAULT_MODEL configuration.
    """
    with patch.dict(
        os.environ,
        {
            "DEFAULT_MODEL": "gemini/gemini-2.0-flash-exp-image-generation",
        },
        clear=True,
    ):
        from importlib import reload

        import app.core.config

        reload(app.core.config)
        settings = app.core.config.Settings()

        assert settings.DEFAULT_MODEL == "gemini/gemini-2.0-flash-exp-image-generation"


def test_filter_image_models_default():
    """
    Test FILTER_IMAGE_MODELS defaults to True.
    """
    with patch.dict(os.environ, {}, clear=True):
        from importlib import reload

        import app.core.config

        reload(app.core.config)
        settings = app.core.config.Settings()

        assert settings.FILTER_IMAGE_MODELS is True
