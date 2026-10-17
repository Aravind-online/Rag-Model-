import ast
import json
import math
import operator
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from rag import NOT_AVAILABLE_ANSWER, build_context, generate_answer


SYSTEM_PROMPT = """You are a helpful assistant. Use the knowledge-base search for questions about the user's documents. Use the calculator for arithmetic and the current-time tool for date or time questions. For time questions, use UTC unless the user gives a location or timezone. Do not expose internal retrieval details in user-facing answers."""
MAX_TOOL_ROUNDS = 5

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the user's indexed documents for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise search query for the user's documents.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic using numbers, parentheses, +, -, *, /, //, %, and **.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in an IANA timezone, such as UTC or Asia/Tokyo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name. Defaults to UTC.",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
]

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate(expression: str) -> int | float:
    """Evaluate arithmetic without allowing names, calls, or other Python syntax."""
    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression is too complex.")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent must be between -100 and 100.")
            result = _BINARY_OPERATORS[type(node.op)](left, right)
            if not math.isfinite(result):
                raise ValueError("Result is outside the supported numeric range.")
            return result
        raise ValueError("Only basic arithmetic is allowed.")

    return evaluate(tree.body)


def execute_tool(name: str, arguments: dict, knowledge_base=None) -> str:
    if name == "search_knowledge_base":
        if knowledge_base is None:
            return "The knowledge base is unavailable."
        return knowledge_base.search(arguments["query"])
    if name == "calculate":
        return str(calculate(arguments["expression"]))
    if name == "get_current_time":
        timezone = arguments.get("timezone", "UTC")
        current_time = datetime.now(ZoneInfo(timezone))
        return current_time.isoformat(timespec="seconds")
    return f"Unknown tool: {name}"


def run_agent(
    client,
    model: str,
    messages: list[dict[str, str]],
    on_tool: Callable[[str, dict, str], None] | None = None,
    knowledge_base=None,
    top_k: int = 5,
) -> str:
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages[-20:],
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=model,
            messages=conversation,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        assistant_message = response.choices[0].message
        if not assistant_message.tool_calls:
            return assistant_message.content or "I couldn't generate a response."

        conversation.append(assistant_message.model_dump(exclude_none=True))
        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            arguments = {}
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
                if name == "search_knowledge_base":
                    if knowledge_base is None:
                        return NOT_AVAILABLE_ANSWER
                    documents = knowledge_base.retrieve_documents(
                        arguments["query"], limit=top_k
                    )
                    context = build_context(documents)
                    if not context:
                        return NOT_AVAILABLE_ANSWER
                    query = next(
                        (
                            message["content"]
                            for message in reversed(messages)
                            if message.get("role") == "user"
                        ),
                        "",
                    )
                    history = [
                        message
                        for message in messages[:-1]
                        if message.get("role") in {"user", "assistant"}
                    ]
                    return generate_answer(
                        client, model, query, context, history=history
                    )
                result = execute_tool(name, arguments, knowledge_base)
            except (ArithmeticError, KeyError, SyntaxError, TypeError, ValueError) as error:
                result = f"Tool error: {error}"
            if on_tool:
                on_tool(name, arguments, result)
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return "I reached the tool-call limit before finishing. Please try a simpler request."