#!/usr/bin/env python3
"""
MCP Server for Local Vector RAG Tool
Provides document search capabilities for the LLM VTuber system
"""

import asyncio
import json
from typing import Any, Sequence
from loguru import logger

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    TextContent,
    Tool,
)

from .vector_search import LocalVectorRAG


class LocalVectorRAGServer:
    """MCP Server for Local Vector RAG functionality"""

    def __init__(self, docs_folder: str = "data/company_docs"):
        """
        Initialize the Local Vector RAG MCP Server

        Args:
            docs_folder: Path to the documents folder
        """
        self.server = Server("local-vector-rag")
        self.rag = LocalVectorRAG(docs_folder)
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools():
            """List available tools"""
            return [
                Tool(
                    name="search_documents",
                    description="Search for documents",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            """Handle tool calls"""
            if name == "search_documents":
                return await self._handle_search_documents(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _handle_search_documents(self, arguments: dict):
        """Handle document search requests"""
        query = arguments.get("query")
        if not query:
            raise ValueError("Missing required parameter: query")
        
        top_k = arguments.get("top_k", 3)
        
        results = self.rag.search(query, top_k)
        
        # Format results for response
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = {
                "rank": i,
                "source": result["source"],
                "content": result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"],
                "score": result.get("score", 0),
                "search_type": result.get("search_type", "unknown")
            }
            formatted_results.append(formatted_result)
        
        response_text = f"Found {len(results)} relevant documents for query: '{query}'\n\n"
        
        for result in formatted_results:
            response_text += f"**Document {result['rank']}: {result['source']}**\n"
            response_text += f"Score: {result['score']:.4f} ({result['search_type']} search)\n"
            response_text += f"Content: {result['content']}\n\n"
        
        return [TextContent(type="text", text=response_text)]

    async def run(self, transport_type: str = "stdio") -> None:
        """Run the MCP server"""
        if transport_type == "stdio":
            from mcp.server.stdio import stdio_server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="local-vector-rag",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )


async def main():
    """Main entry point for the MCP server"""
    import sys
    docs_folder = sys.argv[1] if len(sys.argv) > 1 else "data/company_docs"
    
    server = LocalVectorRAGServer(docs_folder)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
