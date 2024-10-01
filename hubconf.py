# hubconf.py
dependencies = ['torch']


def simple_model(input_size=10, hidden_size=50, num_classes=10, **kwargs):
    """Load the SimpleModel."""
    model = SimpleModel(input_size, hidden_size, num_classes)
    return model