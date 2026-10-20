"""Local model directory override, or public model ID for explicit first download."""
import os
MODEL_PATH = os.environ.get('CARDASSIST_MODEL_PATH', 'google/flan-t5-small')
