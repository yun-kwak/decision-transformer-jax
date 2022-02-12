import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.field1 = 1
    config.field2 = "tom"
    config.nested = ml_collections.ConfigDict()
    config.nested.field = 2.23
    config.tuple = (1, 2, 3)
    return config
