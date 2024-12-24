openai_function_schemas = [
    {
        "name": "get_guitar_info",
        "description": "Get specs about a specific guitar from MelodyCraft Guitars database",
        "parameters": {
            "type": "object",
            "properties": {
                "guitar_name": {
                    "type": "string",
                    "description": "The name of the guitar model to look up",
                },
            },
            "required": ["guitar_name"],
        },
    }
]