import dashscope
from dashscope.api_entities.dashscope_response import Role

def convert_nl_to_sql(natural_language_query: str, api_key: str, table_schema: str) -> str:
    """
    Converts a natural language query to a SQL query using the qwen7b model.
    """
    dashscope.api_key = api_key
    messages = [
        {
            "role": Role.SYSTEM,
            "content": f"You are a helpful assistant that converts natural language questions to SQL queries. You are given the following table schema:\n\n{table_schema}\n\nPlease only output the SQL query and nothing else."
        },
        {
            "role": Role.USER,
            "content": natural_language_query
        }
    ]

    response = dashscope.Generation.call(
        model='qwen-turbo',
        messages=messages,
        result_format='message',
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        raise Exception(f"Error calling Dashscope API: {response.message}")